import os.path as osp

import cv2
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from .utils import scandir


class Resize(object):
    def __init__(self, resize, resize_small=False):
        self.resize = resize
        self.resize_small = resize_small

    def __call__(self, img):
        w, h = img.width, img.height
        if max(h, w) > self.resize or self.resize_small:
            ratio = self.resize / max(h, w)
            h_new, w_new = round(h * ratio), round(w * ratio)
            img = img.resize((w_new, h_new), Image.Resampling.BICUBIC)
        return img


class CustomDataset(Dataset):
    def __init__(self,
                 dataroot_gt,
                 meta_info_file=None,
                 enlarge_ratio=1,
                 mode="RGB",
                 hflip=None,
                 resize=None,
                 crop_size=None):
        super(CustomDataset, self).__init__()
        self.gt_folder = dataroot_gt
        self.meta_info_file = meta_info_file
        self.enlarge_ratio = enlarge_ratio
        self.mode = mode
        self.hflip = hflip
        self.resize = resize
        self.crop_size = crop_size

        if meta_info_file:
            with open(meta_info_file, "r") as fin:
                self.gt_paths = [
                    osp.join(self.gt_folder,
                             line.strip().split(" ")[0]) for line in fin
                ]
        else:
            self.gt_paths = sorted(
                list(scandir(self.gt_folder, full_path=True)))
        self.gt_paths = self.gt_paths * self.enlarge_ratio

        if self.hflip:
            self.fn_hflip = transforms.RandomHorizontalFlip()
        if self.resize:
            if self.resize["type"] == "long":
                self.fn_resize = Resize(self.resize["size"])
            elif self.resize["type"] == "short":
                self.fn_resize = transforms.Resize(self.resize["size"])
            else:
                raise NotImplementedError
        if self.crop_size:
            self.fn_crop = transforms.RandomCrop(self.crop_size)
        self.fn_totensor = transforms.ToTensor()

    def __getitem__(self, index):
        gt_path = self.gt_paths[index]
        img = cv2.imread(gt_path)
        if self.mode == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img, "RGB")
        elif self.mode == "GRAY":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = Image.fromarray(img, "L")
        else:
            raise NotImplementedError
        if self.hflip:
            img = self.fn_hflip(img)
        if self.resize:
            img = self.fn_resize(img)
        if self.crop_size:
            img = self.fn_crop(img)
        img = 2 * self.fn_totensor(img) - 1

        return {"image": img}

    def __len__(self):
        return len(self.gt_paths)
