import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms
from PIL import Image
import cv2 as cv

np.random.seed(0)

class GaussianBlur(object):
  """
  blur a single image on CPU
  """
  def __init__(self, kernel_size):
    radias = kernel_size // 2
    kernel_size = radias * 2 + 1
    self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                            stride=1, padding=0, bias=False, groups=3)
    self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                            stride=1, padding=0, bias=False, groups=3)
    self.k = kernel_size
    self.r = radias

    self.blur = nn.Sequential(
        nn.ReflectionPad2d(radias),
        self.blur_h,
        self.blur_v
    )

    self.pil_to_tensor = transforms.ToTensor()
    self.tensor_to_pil = transforms.ToPILImage()

  def __call__(self, img):
    img = self.pil_to_tensor(img).unsqueeze(0)

    sigma = np.random.uniform(0.1, 2.0)
    x = np.arange(-self.r, self.r + 1)
    x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
    x = x / x.sum()
    x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

    self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
    self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

    with torch.no_grad():
        img = self.blur(img)
        img = img.squeeze()

    img = self.tensor_to_pil(img)

    return img


class AddCannyEdgeLayer(object):
  def __init__(self, threadhold1, threadhold2):
    self.th1 = threadhold1
    self.th2 = threadhold2

  def __call__(self, img: Image):
    _img = np.array(img.convert('L'))
    edge = cv.Canny(_img, 100, 200)[:, :, None]
    # print(img.shape, _img.shape, edge.shape)
    img = np.append(img, edge, axis=-1)
    return img


def get_simclr_pipeline_transform(size=80, s=1):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)

    trans = [
        transforms.RandomResizedCrop(size=size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(kernel_size=int(0.1 * size)),
        AddCannyEdgeLayer(50, 200),
        transforms.ToTensor()
    ]
    
    data_transforms = transforms.Compose(trans)
    return data_transforms


def get_simclr_encoding_transform(size=80):
    trans = [
        transforms.Resize(size),
        AddCannyEdgeLayer(50, 200), 
        transforms.ToTensor()
    ]
    # converts the image, a PIL image, into a PyTorch Tensor
    return transforms.Compose(trans)
