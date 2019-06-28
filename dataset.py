import pickle
from itertools import chain

import requests
import numpy as np

import os

import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler, BatchSampler



class Street2ShopDataset(Dataset):

    def __init__(self, images_path, train=True, fixed_size=None):

        # load images info
        self.images_path = images_path
        self.dress_id2dress_img_name = sorted(os.listdir(images_path))
        self.fixed_size = fixed_size

        if train:
            _file_name = "train_product_id2dress_id.pkl"
        else:
            _file_name = "test_product_id2dress_id.pkl"

        with open(_file_name , "rb") as f:
            self.product_id2dress_id = pickle.load(f)

        with open("dress_id2product_id.pkl", "rb") as f:
            self.dress_id2product_id = pickle.load(f)


    def __len__(self):
        return len(list(chain(*self.product_id2dress_id.values())))

    def __getitem__(self, item):
        image = self.image_link2tensor(self.dress_id2dress_img_name[item], fixed_size=self.fixed_size)
        product_id = self.dress_id2product_id[item][0]
        return image, product_id

    def image_link2tensor(self, image_link, fixed_size=None):
        """
        Return: array: np.ndarray image [H,W,CH]
        """
        image = cv2.imread(self.images_path+image_link)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if not fixed_size is None:
            image = cv2.resize(image, dsize=fixed_size)
        image = np.transpose(image, (2, 0, 1)) / 255
        image = torch.as_tensor(image, dtype=torch.float32)
        return image


class ContrastiveSampler(BatchSampler):
    def __init__(self, product_id2dress_id, labels_in_batch):
        self.labels_in_batch = labels_in_batch
        self.product_id2dress_id = product_id2dress_id
        self.labels = list(product_id2dress_id.keys())
        self.num_iters = int(len(self.labels)/self.labels_in_batch/5)

    def __len__(self):
        return self.num_iters

    def __iter__(self):
        for i in range(self.num_iters):
            cur_labs = np.random.choice(self.labels, self.labels_in_batch)
            batch = list(chain(*[ self.product_id2dress_id[pr_id] for pr_id in cur_labs]))
            yield batch



if __name__ == "__main__":
    data = Street2ShopDataset(images_path="/data/street2shop_dresses/",train=True, fixed_size=(900, 500))
    id = data.product_id2dress_id[list(data.product_id2dress_id.keys())[0]][0]
    img, pr_id = data[id]
    print("Пример размера одного изображения: ", img.size(), "Id продукта: ", pr_id)

    samples = ContrastiveSampler(data.product_id2dress_id, 3)
    print("Количество изображений за эпоху: ",len(set(chain(*samples.__iter__()))))