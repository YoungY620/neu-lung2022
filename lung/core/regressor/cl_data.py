import os

import torch
from torchvision.transforms import transforms
import numpy as np
from PIL import Image

from lung.core.regressor.transforms import GaussianBlur, AddCannyEdgeLayer, get_simclr_pipeline_transform


def get_filelist(dir, filter=lambda filename:(filename.split("-")[1] == "100")):
  flist = []
  for home, dirs, files in os.walk(dir):
    for filename in files:
      if filter(filename):
        flist.append(os.path.join(home, filename))
  return flist



class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


class ContrastiveLearningDataset(torch.utils.data.Dataset):

  def __init__(self, data_root, n_views=4) -> None:
    transforms = get_simclr_pipeline_transform(80)
    self.transforms = ContrastiveLearningViewGenerator(transforms, n_views)

    flist = get_filelist(data_root)
    self.img_boxes = []
    h, w, _ = np.array(Image.open(flist[0])).shape
    for f in flist:
      for xmin, ymin in zip(range(0, h, h//8), range(0, w, w//8)):
        img = Image.open(f).crop((xmin, ymin, xmin+h/8, ymin+w/8)).convert("RGB")
        self.img_boxes.append((f, img))
  
  def __len__(self):
    return len(self.img_boxes)

  def __getitem__(self, idx):
    imfile, img = self.img_boxes[idx]

    if self.transforms != None:
      img = self.transforms(img)

    return img


