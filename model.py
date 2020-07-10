import os
import pandas as pd
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
grid_size_multiplier = 4


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


def box_iou(b1, b2):
    """Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    """
    # Expand dim to apply broadcasting.
    b1 = torch.unsqueeze(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = torch.unsqueeze(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = torch.max(b1_mins, b2_mins)
    intersect_maxes = torch.min(b1_maxes, b2_maxes)
    intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.tensor(0.))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters.
    feats: nB, nC, nH, nW
    """

    nA = anchors_per_level
    nB, nC, nH, nW = feats.shape
    anchors = anchors.view([1, 1, 1, nA, 2])
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
    polygons_y = feats[:, :, :, :, 5 + num_classes + 1:num_classes + 5 + NUM_ANGLES3:3]
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
    anchors = torch.from_numpy(anchors).float()
    g_y_true = y_true
    input_shape = np.array(yolo_outputs.shape[2:]) * grid_size_multiplier

    grid_shapes = np.array(y_true.shape[1:3])
    # batch size
    nB = yolo_outputs.shape[0]
    ix = torch.LongTensor(range(5+num_classes+NUM_ANGLES3))
    object_mask = y_true.index_select(4, ix[4:5])   # confidence
    vertices_mask = y_true.index_select(4, ix[5 + num_classes + 2:5 + num_classes + NUM_ANGLES3:3])
    true_class_probs = y_true.index_select(4, ix[5:5 + num_classes])
    yolo_head(yolo_outputs, anchors, num_classes, input_shape, calc_loss=True)

    grid, raw_pred, pred_xy, pred_wh, pol_cnf = yolo_head(yolo_outputs, anchors, num_classes, input_shape, calc_loss=True)
    pred_box = torch.cat([pred_xy.float(), pred_wh.float()], -1)
    raw_true_xy = y_true[:, :, :, :, :2] * grid_shapes[::-1] - grid
    raw_true_polygon0 = y_true[:, :, :, :, 5 + num_classes: 5 + num_classes + NUM_ANGLES3]

    raw_true_wh = torch.log(y_true[..., 2:4] / anchors * input_shape[::-1]).float()
    raw_true_wh[raw_true_wh==-float('inf')] = 0
    # raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf

    raw_true_polygon_x = raw_true_polygon0[..., ::3]
    raw_true_polygon_y = raw_true_polygon0[..., 1::3]

    dx = torch.square(anchors[..., 0:1] / 2)
    dy = torch.square(anchors[..., 1:2] / 2)
    d = torch.sqrt(dx + dy)
    # a = torch.pow(torch.from_numpy(np.array(input_shape[::-1][0])), 2).float()
    a = torch.pow(torch.tensor(input_shape[::-1][0]), 2).float()
    # b = torch.pow(torch.from_numpy(np.array(input_shape[::-1][1])), 2).float()
    b = torch.pow(torch.tensor(input_shape[::-1][1]), 2).float()
    diagonal = torch.sqrt(a + b)
    raw_true_polygon_x = torch.log(raw_true_polygon_x / d * diagonal)
    raw_true_polygon_x[raw_true_polygon_x == -float('inf')] = 0
    # raw_true_polygon_x = K.switch(vertices_mask, raw_true_polygon_x, K.zeros_like(raw_true_polygon_x))
    box_loss_scale = 2 - y_true[..., 2:3] * y_true[..., 3:4]

    # Find ignore mask, iterate over each of batch.
    ignore_mask = []
    object_mask_bool = object_mask.bool()
    for b in range(nB):
        true_box = y_true[b, ..., 0:4][object_mask_bool[b, ..., 0]]
        iou = box_iou(pred_box[b], true_box)    # H,W, nA, num_gt
        best_iou, _ = torch.max(iou, -1)
        ignore_mask.append(best_iou < ignore_thresh)

    ignore_mask = torch.stack(ignore_mask)
    ignore_mask = ignore_mask.unsqueeze(-1)
    # K.binary_crossentropy is helpful to avoid exp overflow.
    xy_loss = object_mask * box_loss_scale * F.binary_cross_entropy_with_logits(raw_pred[..., 0:2], raw_true_xy)
    wh_loss = object_mask * box_loss_scale * 0.5 * torch.square(raw_true_wh - raw_pred[..., 2:4])

    # print((1-object_mask).shape)
    # print(F.binary_cross_entropy_with_logits(object_mask, raw_pred[..., 4:5]))
    # print(((1 - object_mask) * F.binary_cross_entropy_with_logits(object_mask, raw_pred[..., 4:5]) * ignore_mask).shape)

    confidence_loss = object_mask * F.binary_cross_entropy_with_logits(raw_pred[..., 4:5], object_mask)
    # + (1 - object_mask) * F.binary_cross_entropy_with_logits(raw_pred[..., 4:5], object_mask) * ignore_mask

    class_loss = object_mask * F.binary_cross_entropy_with_logits(raw_pred[..., 5:5 + num_classes], true_class_probs)
    polygon_loss_x = object_mask * vertices_mask * box_loss_scale * 0.5 * torch.square(
        raw_true_polygon_x - raw_pred[..., 5 + num_classes:5 + num_classes + NUM_ANGLES3:3])
    polygon_loss_y = object_mask * vertices_mask * box_loss_scale * F.binary_cross_entropy_with_logits(
        raw_pred[..., 5 + num_classes + 1:5 + num_classes + NUM_ANGLES3:3], raw_true_polygon_y)
    vertices_confidence_loss = object_mask * F.binary_cross_entropy_with_logits(
        raw_pred[..., 5 + num_classes + 2:5 + num_classes + NUM_ANGLES3:3], vertices_mask)

    xy_loss = torch.sum(xy_loss) / nB
    wh_loss = torch.sum(wh_loss) / nB
    class_loss = torch.sum(class_loss) / nB
    confidence_loss = torch.sum(confidence_loss) / nB
    vertices_confidence_loss = torch.sum(vertices_confidence_loss) / nB
    polygon_loss = torch.sum(polygon_loss_x) / nB + torch.sum(polygon_loss_y) / nB

    loss = (xy_loss + wh_loss + confidence_loss + class_loss + 0.2 * polygon_loss + 0.2 * vertices_confidence_loss) / (
            torch.sum(object_mask) + 1) * nB
    return loss


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

    train_dataset = MyDataset(lines, input_shape, anchors, num_classes, True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = MyDataset(lines_val, input_shape, anchors, num_classes, False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    epochs = 10
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (image, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            yolo_outputs = model(image)
            loss = yolo_loss(yolo_outputs, labels, anchors, num_classes, ignore_thresh=0.5)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print("epoch_{},iter_{}, loss_{}".format(epoch, i, running_loss / 100))
                running_loss = 0.0


if __name__=='__main__':
    main()
