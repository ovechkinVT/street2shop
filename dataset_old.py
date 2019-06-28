import json
import pickle
from itertools import chain

import requests
import numpy as np

import cv2
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler, BatchSampler



class Street2ShopDataset(Dataset):

    def __init__(self, train=True, transform=None):

        # load images info
        with open("id2image_link.pkl", "rb") as f:
            self.id2image_link = pickle.load(f)

        if train:
            street_ids_file_name = "street_id2product_id_train.pkl"
        else:
            street_ids_file_name = "street_id2product_id_test.pkl"
        with open(street_ids_file_name, "rb") as f:
            self.street_id2product_id = pickle.load(f)

        with open("product_id2shop_id.pkl", "rb") as f:
            self.product_id2shop_id =pickle.load(f)


    def __len__(self):
        return len(self.id2image_link)

    def __getitem__(self, item):

        # get indexes
        street_id = item
        products_id = self.street_id2product_id[item]
        shop_ids = [self.product_id2shop_id[id] for id in products_id]
        shop_ids = list(set(chain(*shop_ids)))

        #get images
        street_image = self.image_link2array(self.id2image_link[street_id])
        shop_images = [self.image_link2array(self.id2image_link[id]) for id in shop_ids]
        shop_images = [img for img in shop_images if img is not None]
        # shop_images = np.concatenate(shop_images , axis=3)
        if (len(shop_images) > 0) and (not street_image is None):
            return street_image, shop_images
        else:
            return None


    def image_link2array(self, image_link):
        """
        Return:
        ------
        array: np.ndarray
            image [H,W,CH]
        """

        response = requests.get(image_link)
        if response.status_code == 200:
            image = np.asarray(bytearray(response.content), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        else:
            return None


class ContrastiveSampler(BatchSampler):
    def __init__(self, batch_size, num_classes, labels):
        self.num_classes = num_classes
        self.imgs_per_class = labels.size()[0] // (2 * num_classes)
        self.labels = labels
        self.batch_size = batch_size

    def __iter__(self):
        num_yielded = 0
        while num_yielded < (self.num_classes * self.imgs_per_class):
            current_label = torch.randint(10, size=(1,)).item()

            positive_indexes = torch.arange(0, self.labels.size()[0])[self.labels == current_label]
            negative_indexes = torch.arange(0, self.labels.size()[0])[self.labels != current_label]

            positive_indexes = positive_indexes[torch.randint(positive_indexes.size()[0],
                                                              size=(self.imgs_per_class,)).long()]
            negative_indexes = negative_indexes[torch.randint(negative_indexes.size()[0],
                                                              size=(self.imgs_per_class,)).long()]
            batch = torch.cat([positive_indexes, negative_indexes], dim=0).tolist()

            num_yielded += self.batch_size
            yield batch

if __name__ == "__main__":
    data = Street2ShopDataset()

    with open("street_id2product_id.pkl", "rb") as f:
        street_id2product_id = pickle.load(f)

    id = list(street_id2product_id.keys())[4]
    street_image, shop_images = data[id]
    print("Размер изображения пользователя: ",street_image.shape)
    print("Размер изображений магазина: ", [img.shape for img in shop_images])