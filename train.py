import pickle
import json

from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
from neural_network import DressFindingNN
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x, y, device):
        x_diff = ((x[:, None] - x[None, :]) ** 2).sum(dim=-1)
        y = (y[:, None] != y[None, :]).float()
        loss = (1 - y) * x_diff + y * torch.max(self.margin - x_diff, torch.tensor(0.).to(device))
        return loss.mean() / 2

def train_model(model, device, optimizer, train_data_loader, test_data_loader, contrastive_loss, epoch, log_writer):
    for ep in range(epoch):
        print("Epoch: {}".format(str(ep)))
        train_epoch(model, device, optimizer, train_data_loader, test_data_loader, contrastive_loss, ep,  log_writer=log_writer)


def train_epoch(model, device, optimizer, train_data_loader, test_data_loader, contrastive_loss, epoch, log_writer):

    batch_i = 0
    for images, products in tqdm(train_data_loader):
        model.train()
        images= images.to(device)
        products=products.to(device)

        images_vectors = model(images)
        loss = contrastive_loss(images_vectors, products, device)

        loss.backward()
        optimizer.step()

        layer_grads = dict()
        for name, layer in model.named_parameters():
            layer_grads[name] = torch.norm(layer.grad).item()
        log_writer.add_scalars("layers' gradients", layer_grads, (epoch+1)*len(train_data_loader)+batch_i)

        optimizer.zero_grad()

        log_writer.add_scalar("contrastive loss train", loss.item(), (epoch+1)*len(train_data_loader)+batch_i)
        batch_i += 1

    test_losses = list()
    for images, products in tqdm(test_data_loader):
        model.eval()
        images = images.to(device)
        products = products.to(device)

        with torch.no_grad():
            images_vectors = model(images)
            loss = contrastive_loss(images_vectors, products, device)
            test_losses.append(loss.item())

    log_writer.add_scalar("contrastive loss test", np.mean(test_losses), (epoch+1)*len(train_data_loader)+batch_i)

if __name__ == "__main__":

    from dataset import Street2ShopDataset, ContrastiveSampler

    device = "cuda"
    experiment_name = "_without_MRCNN"


    train_data = Street2ShopDataset(images_path="/data/street2shop_dresses/", train=True, fixed_size=(900, 500))
    test_data = Street2ShopDataset(images_path="/data/street2shop_dresses/", train=False, fixed_size=(900, 500))
    train_data_loader = DataLoader(dataset=train_data,
                                  batch_sampler=ContrastiveSampler(train_data.product_id2dress_id, 20))
    test_data_loader = DataLoader(dataset=train_data,
                                  batch_sampler=ContrastiveSampler(test_data.product_id2dress_id, 20))

    contrastive_loss = ContrastiveLoss()
    model = DressFindingNN()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    log_writer = SummaryWriter(comment=experiment_name)

    num_epoch = 10
    train_model(model, device, optimizer, train_data_loader, test_data_loader, contrastive_loss, num_epoch, log_writer=log_writer)
