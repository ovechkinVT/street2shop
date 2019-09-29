import os
import json
from tqdm import tqdm

# load photo links
with open("/home/ubuntu/clothes/data/street2shop/photos.txt") as f:
    img_links = f.readlines()

with open("/home/ubuntu/clothes/data/street2shop/useful_img_id.json", "r") as f:
    useful_img_id = set(json.load(f)) # list of images' ids

# get id and link
img_links = map(lambda row: row.split(","),  img_links)

# only useful images
img_links = filter(lambda x: int(x[0]) in useful_img_id, img_links)

# only links with jpg endings (~90%)
img_links = list(filter(lambda x: x[1].split(".")[-1][:3] == "jpg", img_links))

os.chdir("/home/ubuntu/clothes/data/street2shop/photos")


for img_id, link in tqdm(img_links, total=len(img_links)):
    try:
        os.system("curl -s -o {img_id}.jpg  {link} ".format(link=link, img_id=img_id))
    except:
        print(link)
