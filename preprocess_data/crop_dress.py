import os
from tqdm import tqdm

import numpy as np

import json
import torch

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import cv2


with open("/data/kaggle-fashion/label_descriptions.json") as f:
    desc = json.load(f)
class_names2id =  { categ["name"]:categ["id"] for categ in desc["categories"]}
del desc


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


def get_instance_segmentation_model(weights_path ,num_classes=46):

    # load pretrained model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    # load weights pretrained on imaterialist dataset
    model.load_state_dict(torch.load(weights_path))
    model = model.to("cuda")
    model.eval()

    return model


def correct_image_size(image, max_size):
    h, w, _ = image.shape
    if max(h,w) > max_size:
        k = max_size/max(h,w)
        new_h, new_w = int(h*k), int(w*k)
        image = cv2.resize(image, dsize=(new_w, new_h))
    return image

def get_image_as_tensor(image_path):
    # load image from file and prepare for model
    image = cv2.imread(image_path)
    if not image is None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = correct_image_size(image, max_size=1500)
        image = np.transpose(image, (2, 0, 1)) / 255
        image = torch.as_tensor(image, dtype=torch.float32)
        return image
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
            dress = (dress * 255).astype(int)[:, :, ::-1]
            dresses.append(dress)

        return dresses
    return None

if __name__ == "__main__":

    images = sorted(os.listdir("/data/street2shop"))[(1500+1415+7590+12900):]

    model = get_instance_segmentation_model("/home/ubuntu/imaterialist/model_firts_weigths.pytorch")

    for image_name in tqdm(images):

        image = get_image_as_tensor("/data/street2shop/" + image_name)
        if not image is None:

            try:
                dresses = get_dress_from_image(image, model)
            except:
                print(image.size())
                x = y

            if not dresses is None:
                for i in range(len(dresses)):
                    result = cv2.imwrite("/data/street2shop_dresses/"+image_name+"_dress_"+str(i)+".png",
                                dresses[i])
                    if not result:
                        print(image_name)
        else:
            print("None: "+image_name)