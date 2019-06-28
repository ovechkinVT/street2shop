import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class DressFindingNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.images_embs = None

        self.img2vec = nn.Sequential()
        self.img2vec.add_module("Conv1", nn.Conv2d(3, 3, 10))
        self.img2vec.add_module("MaxPool1", nn.MaxPool2d(4))
        self.img2vec.add_module("Conv2", nn.Conv2d(3, 3, 10))
        self.img2vec.add_module("MaxPool2", nn.MaxPool2d(4))
        self.img2vec.add_module("Conv3", nn.Conv2d(3, 10, 10))
        self.img2vec.add_module("MaxPool3", nn.MaxPool2d(4))
        self.img2vec.add_module("Conv4", nn.Conv2d(10, 20, 3))
        self.img2vec.add_module("MaxPool4", nn.AdaptiveAvgPool2d(output_size=1))
        self.img2vec.add_module("Flatten", Flatten())

        self.img2vec.add_module("Sigmoid", nn.Sigmoid())
        self.img2vec.add_module("Linear", nn.Linear(20, 10))
        # self.img2vec.add_module("Flatten", Flatten())


    def forward(self, x):
        images_vectors = self.img2vec(x)
        return images_vectors



if __name__ == "__main__":

    model = DressFindingNN()
    model.eval()
    x = torch.rand(5,3,900,500)
    print(model(x).size())