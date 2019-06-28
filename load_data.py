import os
import sys

with open("photos/photos.txt") as f:
    img_links = f.readlines()

img_links = [row.split(",")[1] for row in img_links]


os.chdir("/data/street2shop")

from tqdm import tqdm

for link in tqdm(img_links):
    try:
        os.system("wget -q " + link)
    except:
        print(link)