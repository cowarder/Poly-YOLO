import os
import pandas as pd
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import MyDataset
from utils import parse_cfg
from utils import get_classes, get_anchors, data_generator_wrapper

np.set_printoptions(precision=3, suppress=True)
MAX_VERTICES = 1000  # that allows the labels to have 1000 vertices per polygon at max. They are reduced for training
ANGLE_STEP = 15  # that means Poly-YOLO will detect 360/15=24 vertices per polygon at max
NUM_ANGLES3 = int(360 // ANGLE_STEP * 3)
NUM_ANGLES = int(360 // ANGLE_STEP)
anchors_per_level = 9


class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class UpSample(nn.Module):

    def __init__(self, scale_factor=2, mode="nearest"):
        super(UpSample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        assert (x.dim() == 4)
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)


class EmptyLayer(nn.Module):

    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class YoloLayer(nn.Module):

    def __init__(self, name):
        super(YoloLayer, self).__init__()
        self.name = name

    def forward(self, x):
        return {self.name: x}


class Conv2D_BN_Leaky(nn.Module):
    def __init__(self, in_c, out_c, kernel_size):
        super(Conv2D_BN_Leaky, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_size = kernel_size

    def forward(self, x):
        if isinstance(self.kernel_size, int):
            padding = (self.kernel_size - 1) // 2
        elif isinstance(self.kernel_size, tuple):
            padding = (self.kernel_size[0] - 1) // 2
        Conv2D = nn.Conv2d(self.in_c, self.out_c, self.kernel_size, stride=1, padding=padding, bias=False)
        BN = nn.BatchNorm2d(self.out_c)
        Leaky = nn.LeakyReLU(0.1, inplace=True)
        x = Conv2D(x)
        x = BN(x)
        x = Leaky(x)
        return x


class Darknet(nn.Module):

    def __init__(self, cfgfile='./yolo_v3.cfg'):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        # print(len(self.blocks))
        self.models = self.create_modules(self.blocks)
        # print(self.modules)

    def get_yolo_layers(self):
        yolo_layers = {}
        for module in self.models:
            if isinstance(module[0], YoloLayer):
                yolo_layers[module[0].name] = module[0]
        return yolo_layers

    def yolo_body(self, yolo_layers, num_anchors=9, num_classes=13):
        """Create Poly-YOLo model CNN body in Pytorch."""
        tiny = yolo_layers['tiny']
        small = yolo_layers['small']
        medium = yolo_layers['medium']
        big = yolo_layers['big']


        base = 6
        tiny = Conv2D_BN_Leaky(tiny.shape[1], base*32, (1, 1))(tiny)
        small = Conv2D_BN_Leaky(small.shape[1], base*32, (1, 1))(small)
        medium = Conv2D_BN_Leaky(medium.shape[1], base*32, (1, 1))(medium)
        big = Conv2D_BN_Leaky(big.shape[1], base*32, (1, 1))(big)

        up = UpSample(scale_factor=2, mode='bilinear')
        all = medium + up(big)
        all = small + up(all)
        all = tiny + up(all)

        num_filters = base * 32
        all = Conv2D_BN_Leaky(all.shape[1], num_filters, (1, 1))(all)
        all = Conv2D_BN_Leaky(all.shape[1], num_filters * 2, (3, 3))(all)
        all = Conv2D_BN_Leaky(all.shape[1], num_filters, (1, 1))(all)
        all = Conv2D_BN_Leaky(all.shape[1], num_filters*2, (3, 3))(all)
        all = nn.Conv2d(all.shape[1], num_anchors * (num_classes + 5 + NUM_ANGLES3), (1, 1))(all)

        return all
        # print(tiny.shape, small.shape, medium.shape, big.shape)

    def darknet_body(self, x):
        yolo_layers = {}
        output = {}
        for index, block in enumerate(self.blocks[1:]):
            if block['type'] == 'convolutional':
                x = self.models[index](x)
                output[index] = x
            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                from_layer = int(from_layer) if int(from_layer) > 0 else int(from_layer) + index
                x1 = output[from_layer]
                x2 = output[index - 1]
                x = x1 + x2
                output[index] = x
            elif block['type'] == 'yolo':
                output[index] = x
                name = block['name']
                yolo_layers[name] = x

            else:
                print("Unknown type {0}".format(block['type']))
        return yolo_layers

    def forward(self, x):
        yolo_layers = self.darknet_body(x)
        x = self.yolo_body(yolo_layers, num_classes=1)
        return x

    def create_modules(self, blocks):
        self.net_info = blocks[0]
        self.width = int(self.net_info['width'])
        self.height = int(self.net_info['height'])
        prev_filters = int(self.net_info['channels'])
        out_filters = []
        models = nn.ModuleList()

        # iterate all blocks
        for index, block in enumerate(blocks[1:]):
            module = nn.Sequential()

            if block["type"] == 'convolutional':
                activation_func = block['activation']
                kernel_size = int(block['size'])
                pad = int(block['pad'])
                filters = int(block['filters'])
                stride = int(block['stride'])

                try:
                    batch_normalize = int(block['batch_normalize'])
                    bias = False
                except KeyError:
                    # no BN
                    batch_normalize = 0
                    bias = True

                if pad:
                    padding = (kernel_size - 1) // 2
                else:
                    padding = 0

                conv_layer = nn.Conv2d(prev_filters, filters, kernel_size, stride, padding, bias=bias)
                module.add_module('conv_{0}'.format(index), conv_layer)

                if batch_normalize:
                    module.add_module('batch_norm_{}'.format(index), nn.BatchNorm2d(filters))

                if activation_func == 'leaky':
                    activation = nn.LeakyReLU(0.1, inplace=True)
                    module.add_module('leaky_{0}'.format(index), activation)

            elif block['type'] == 'shortcut':
                module.add_module('shortcut_{0}'.format(index), EmptyLayer())

            elif block['type'] == 'yolo':
                name = block['name']
                module.add_module('feature_{0}'.format(index), YoloLayer(name))

            models.append(module)
            prev_filters = filters
            out_filters.append(prev_filters)

        return models


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters.
    feats: nB, nC, nH, nW
    """

    nA = anchors_per_level
    nB, nC, nH, nW = feats.shape
    anchors = torch.from_numpy(anchors).view([1, 1, 1, nA, 2])
    grid_y = torch.linspace(0, nH-1, nH).view(-1, 1, 1, 1).repeat([1, nW, 1, 1])
    grid_x = torch.linspace(0, nW-1, nW).view(1, -1, 1, 1).repeat([nH, 1, 1, 1])
    grid = torch.cat([grid_x, grid_y], dim=-1)
    feats = feats.view([-1, nH, nW, nA, num_classes + 5 + NUM_ANGLES3])
    ix = torch.LongTensor(range(5))

    box_xy = (feats[:, :, :, :, :2].sigmoid() + grid) / torch.tensor([nW, nH])
    box_wh = (feats[:, :, :, :, 2:4].exp() * anchors) / torch.tensor([input_shape[1], input_shape[0]])
    box_confidence = feats[:, :, :, :, 4:5].sigmoid()
    box_class_probs = feats[:, :, :, :,  5:5 + num_classes]
    polygons_confidence = feats[:, :, :, :, 5 + num_classes + 2:5 + num_classes + NUM_ANGLES3:3].sigmoid()
    polygons_x = feats[:, :, :, :, 5 + num_classes:num_classes + 5 + NUM_ANGLES3:3].exp()
    # print(box_confidence.shape, box_class_probs.shape,
    #       box_class_probs.shape, polygons_confidence.shape, polygons_x.shape)

    dx = (anchors[:, :, :, :, 0:1] / 2).square()
    dy = (anchors[:, :, :, :, 1:2] / 2).square()
    d = torch.sqrt(dx + dy)
    a = torch.pow(torch.from_numpy(np.array(input_shape[::-1])), 2).float()
    b = torch.sum(a)
    diagonal = torch.sqrt(b)
    polygons_x = polygons_x * d / diagonal
    polygons_y = feats[..., 5 + num_classes + 1:num_classes + 5 + NUM_ANGLES3:3]
    polygons_y = polygons_y.sigmoid()
    if calc_loss:
        return grid, feats, box_xy, box_wh, polygons_confidence
    return box_xy, box_wh, box_confidence, box_class_probs, polygons_x, polygons_y, polygons_confidence


def yolo_loss(yolo_outputs, y_true, anchors, num_classes, ignore_thresh=.5):
    """Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: tensor, the output of yolo_body or tiny_yolo_body (N, C, H, W)
    y_true: tensor, the output of preprocess_true_boxes (N, H, W, A, (4 + 1 + num_class + NUM_ANGLES3))
    anchors: array, shape=(N, 2)
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    """
    g_y_true = y_true
    input_shape = np.array(yolo_outputs.shape[2:])
    grid_shapes = np.array(y_true[1:3])
    # batch size
    N = yolo_outputs.shape[0]
    ix = torch.LongTensor(range(5+num_classes+NUM_ANGLES3))
    object_mask = y_true.index_select(4, ix[4:5])   # confidence
    vertices_mask = y_true.index_select(4, ix[5 + num_classes + 2:5 + num_classes + NUM_ANGLES3:3])
    true_class_probs = y_true.index_select(4, ix[5:5 + num_classes])
    yolo_head(yolo_outputs, anchors, num_classes, input_shape, calc_loss=True)
    pass


def main():
    model = Darknet()
    # x = torch.randn(4, 3, 416, 832)
    # res = model(x)
    # print(res.shape)

    annotation_path = './data/simulator_dataset/simulator-train.txt'
    validation_path = './data/simulator_dataset/simulator-val.txt'
    log_dir = 'models/'
    classes_path = './data/yolo_classes.txt'
    anchors_path = './data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416, 832)  # multiple of 32, hw

    with open(annotation_path) as f:
        lines = f.readlines()

    with open(validation_path) as f:
        lines_val = f.readlines()

    for i in range(0, len(lines)):
        lines[i] = lines[i].split()
        for element in range(1, len(lines[i])):
            for symbol in range(lines[i][element].count(',') - 4, MAX_VERTICES * 2, 2):
                lines[i][element] = lines[i][element] + ',0,0'

    for i in range(0, len(lines_val)):
        lines_val[i] = lines_val[i].split()
        for element in range(1, len(lines_val[i])):
            for symbol in range(lines_val[i][element].count(',') - 4, MAX_VERTICES * 2, 2):
                lines_val[i][element] = lines_val[i][element] + ',0,0'

    num_val = int(len(lines_val))
    num_train = len(lines)
    batch_size = 2  # decrease/increase batch size according to your memory of your GPU
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    gene = data_generator_wrapper(lines, batch_size, input_shape, anchors, num_classes, True)

    train_dataset = MyDataset(lines, input_shape, anchors, num_classes, True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = MyDataset(lines_val, input_shape, anchors, num_classes, False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for i, (image, labels) in enumerate(train_loader):
        yolo_outputs = model(image)
        yolo_loss(yolo_outputs, labels, anchors, num_classes, ignore_thresh=0.5)
        break


if __name__=='__main__':
    main()
