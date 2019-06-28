import os
import pickle
import numpy as np

import json

import torch
import cv2

from itertools import chain

from tqdm import tqdm

from preprocess_data.crop_dress import get_image_as_tensor, get_instance_segmentation_model

from neural_network import DressFindingNN

def filter_prediction(prediction, img_class_name, class_names2id):

    img_class_id = class_names2id[img_class_name]

    for k, v in prediction.items():
        prediction[k] = v.to("cpu").numpy()

    # filter only current class
    class_filter = np.where(prediction["labels"] == img_class_id)[0]

    if len(class_filter)>0:
        # filter class instances by score
        score_filter = class_filter[np.where(prediction["scores"][class_filter]>0.05)[0]]
        if len(score_filter)>0:
            for k,v in prediction.items():
                prediction[k] = v[score_filter]
            return prediction

    return None


def get_dress_from_image(image, model):

    # get only dress
    with torch.no_grad():
        prediction = model([image.to("cuda")])
    model_pred = filter_prediction(prediction[0], "dress", class_names2id)

    # crop dress images
    if not model_pred is None:
        num_dresses = len(model_pred["masks"])
        dresses = list()
        for i in range(num_dresses):
            dress = image.numpy() * (model_pred["masks"][i] > 0.5).astype(int)
            x0, y0, x1, y1 = model_pred["boxes"][i].astype(int)
            dress = np.transpose(dress, (1, 2, 0))[y0:y1, x0:x1, :]
            dresses.append(dress)

        return dresses
    return None

def get_dress2vec_model(weights_path):
    model = DressFindingNN()
    model.load_state_dict(torch.load(weights_path))
    model = model.to("cuda")
    model.eval()
    return model


def image2vec(image_path, dress_fixed_size):

    # dress search
    segmentaion_model = get_instance_segmentation_model("/home/ubuntu/imaterialist/model_firts_weigths.pytorch")

    image = get_image_as_tensor(image_path)
    if image is None:
        print("\nНевозможно прочитать изображение ", image_path)
        return None

    dresses = get_dress_from_image(image, segmentaion_model)
    if dresses is None:
        print("\nНа изображении нет платьев:", image_path)
        return None
    dresses = [cv2.resize(dress.astype(float), dsize=dress_fixed_size) for dress in dresses ]
    dresses = np.concatenate([np.transpose(dress, (2,1,0))[None,...] for dress in dresses], axis=0)

    # dress vectorization
    dress2vec_model = get_dress2vec_model("/home/ubuntu/street2shop/base_model_20ppb.pmw")
    dress_vectors = dress2vec_model(torch.as_tensor(dresses, dtype=torch.float32).to("cuda"))

    return dress_vectors.to("cpu")


with open("/data/kaggle-fashion/label_descriptions.json") as f:
    desc = json.load(f)
class_names2id =  { categ["name"]:categ["id"] for categ in desc["categories"]}
del desc

if __name__ == "__main__":

    os.chdir("/home/ubuntu/street2shop")
    # get all test examples
    with open("street_id2product_id_test.pkl", "rb") as f:
        street_id2product_id_test = pickle.load(f)

    with open("id2image_link.pkl", "rb") as f:
        id2image_link = pickle.load(f)

    # transform them to vectors
    # test_image_dress_vectors = dict()

    image_ids = sorted(street_id2product_id_test.keys())

    # add other image ids (shop ids)
    with open("test_product_id2id.pkl", "rb") as f:
        test_product_id2id = pickle.load(f)

    all_image_ids = set(chain(*test_product_id2id.values()))
    image_ids = sorted(all_image_ids-set(image_ids))

    for image_id in tqdm(image_ids):
        image_name = id2image_link[image_id].split("/")[-1]
        image_path = "/data/street2shop/"+image_name
        image_vec = image2vec(image_path, dress_fixed_size=(900, 500))

        # save vectors as dict (keys: image_ids)
        with open("test_dress_vectors/{}.pkl".format(image_name), "wb") as f:
            pickle.dump(image_vec, f)