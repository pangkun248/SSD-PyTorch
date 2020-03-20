import torch.utils.data
import numpy as np
from PIL import Image
from torchvision import transforms as tvtsf
import torch.nn.functional as F
from utils.box_tools import wh2xy, assign_anchors, xy2wh, box2loc
from utils.box_tools import create_anchors
from torch.utils.data import Dataset
import glob


class ListDataset(Dataset):
    def __init__(self, path=None, is_train=True):
        with open(path) as f:
            self.img_paths = f.readlines()
        self.is_train = is_train
        self.label_paths = [path.replace('JPGImages', 'labels').replace('.jpg', '.txt') for path in self.img_paths]
        self.anchors_xywh = create_anchors()
        self.anchors_xyxy = wh2xy(self.anchors_xywh)

    def __getitem__(self, index):
        img_path = self.img_paths[index].rstrip()
        label_path = self.label_paths[index].rstrip()
        label_data = (np.loadtxt(label_path).reshape(-1, 5))
        labels = label_data[:, 0].astype(np.int64)
        boxes = label_data[:, 1:].astype(np.float32)
        img = tvtsf.ToTensor()(Image.open(img_path))
        c, h, w = img.shape
        # 将坐标修改为相对形式的坐标
        boxes[:, 0::2] /= w
        boxes[:, 1::2] /= h
        img = F.interpolate(img.unsqueeze(0), size=(300, 300), mode="nearest").squeeze(0)
        img = tvtsf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        gt_boxes = torch.from_numpy(boxes)
        gt_labels = torch.from_numpy(labels)
        anchor_targets, anchor_labels = assign_anchors(gt_boxes, gt_labels, self.anchors_xyxy)
        anchor_targets = xy2wh(anchor_targets)
        anchor_locs = box2loc(anchor_targets, self.anchors_xywh)
        return img, anchor_locs, anchor_labels, img_path

    def __len__(self):
        return len(self.img_paths)

    def _get_annotation(self, image_name):
        w, h = Image.open(image_name).size
        label_path = image_name.replace('JPGImages', 'labels').replace('.jpg', '.txt')
        label_data = (np.loadtxt(label_path).reshape(-1, 5))
        labels = label_data[:, 0].astype(np.int64)
        boxes = label_data[:, 1:].astype(np.float32)
        return boxes, labels, w, h


# 为测试图片准备
class ImageFolder(Dataset):
    def __init__(self, folder_path):
        self.files = glob.glob("%s/*.*" % folder_path)

    def __getitem__(self, index):
        img_path = self.files[index]
        # 这里使用convert是防止使用png图片或其他格式时会有多个通道而引起的报错,
        img = tvtsf.ToTensor()(Image.open(img_path).convert("RGB"))
        img = F.interpolate(img.unsqueeze(0), size=(300, 300), mode="nearest").squeeze(0)
        img = tvtsf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        return img, img_path

    def __len__(self):
        return len(self.files)
