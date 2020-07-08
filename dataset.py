import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from utils import get_random_data, preprocess_true_boxes


class MyDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, anchors, num_classes, is_random):
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.anchors = anchors
        self.num_classes = num_classes
        self.is_random = is_random

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, idx):
        image, box = get_random_data(self.annotation_lines[idx], self.input_shape, random=self.is_random)
        image = torch.from_numpy(np.array(image).transpose(2, 0, 1))
        box = np.array([box])
        label = torch.from_numpy(preprocess_true_boxes(box, self.input_shape, self.anchors, self.num_classes)[0]).squeeze(0)
        return image.float(), label.float()