import numpy as np
import cv2 as cv
from PIL import Image
import math
import random as rd

MAX_VERTICES = 1000  # that allows the labels to have 1000 vertices per polygon at max. They are reduced for training
ANGLE_STEP = 15  # that means Poly-YOLO will detect 360/15=24 vertices per polygon at max
NUM_ANGLES3 = int(360 // ANGLE_STEP * 3)
NUM_ANGLES = int(360 // ANGLE_STEP)
anchor_mask = [[0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7,8], [0,1,2,3,4,5,6,7,8]]  # that should be optimized
grid_size_multiplier = 4  # that is resolution of the output scale compared with input. So it is 1/4



def parse_cfg(cfgfile):
    """
    Parse cfg file, and retur a bloc dictionary.
    :param cfgfile: cfg file path
    :return: blocks
    """
    with open(cfgfile, 'r') as f:
        lines = f.readlines()

    blocks = []     # store info of all blocks
    block = {}      # store info of single block

    for line in lines:
        line = line.strip()
        if len(line) == 0 or line[0] == '#':
            continue
        if line[0] == '[':
            if block:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].strip()
        else:
            key, value = line.split('=')
            block[key.strip()] = value.strip()
    blocks.append(block)
    return blocks


def get_classes(classes_path):
    """loads the classes"""
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    """loads the anchors from a file"""
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def get_random_data(line, input_shape, random=True, max_boxes=80, hue_alter=20, sat_alter=30, val_alter=30, proc_img=True):
    # load data
    # the color conversion is later. it is not necessary to realize bgr->rgb->hsv->rgb
    image = cv.imread("./data/simulator_dataset/imgs/"+line[0])
    iw = image.shape[1]
    ih = image.shape[0]
    h, w = input_shape
    box = np.array([np.array(list(map(float, box.split(','))))
                    for box in line[1:]])

    if not random:
        # resize image
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        image_data = 0
        if proc_img:
            # image = image.resize((nw, nh), Image.BICUBIC)
            image = cv.cvtColor(
                cv.resize(image, (nw, nh), interpolation=cv.INTER_CUBIC), cv.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image) / 255.
        # correct boxes
        box_data = np.zeros((max_boxes, 5 + NUM_ANGLES3))
        if len(box) > 0:
            np.random.shuffle(box)
            if len(box) > max_boxes:
                box = box[:max_boxes]
            box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
            box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
            box_data[:len(box), 0:5] = box[:, 0:5]
            for b in range(0, len(box)):
                for i in range(5, MAX_VERTICES * 2, 2):
                    if box[b,i] == 0 and box[b, i + 1] == 0:
                        continue
                    box[b, i] = box[b, i] * scale + dx
                    box[b, i + 1] = box[b, i + 1] * scale + dy

            box_data[:, i:NUM_ANGLES3 + 5] = 0

            for i in range(0, len(box)):
                boxes_xy = (box[i, 0:2] + box[i, 2:4]) // 2

                for ver in range(5, MAX_VERTICES * 2, 2):
                    if box[i, ver] == 0 and box[i, ver + 1] == 0:
                        break
                    dist_x = boxes_xy[0] - box[i, ver]
                    dist_y = boxes_xy[1] - box[i, ver + 1]
                    dist = np.sqrt(np.power(dist_x, 2) + np.power(dist_y, 2))
                    if (dist < 1): dist = 1 #to avoid inf or nan in log in loss

                    angle = np.degrees(np.arctan2(dist_y, dist_x))
                    if (angle < 0): angle += 360
                    iangle = int(angle) // ANGLE_STEP
                    relative_angle = (angle - (iangle * int(ANGLE_STEP))) / ANGLE_STEP

                    if dist > box_data[i, 5 + iangle * 3]:  # check for vertex existence. only the most distant is taken
                        box_data[i, 5 + iangle * 3] = dist
                        box_data[i, 5 + iangle * 3 + 1] = relative_angle
                        box_data[i, 5 + iangle * 3 + 2] = 1
        return image_data, box_data


    # resize image
    random_scale = rd.uniform(.6, 1.4)
    scale = min(w / iw, h / ih)
    nw = int(iw * scale * random_scale)
    nh = int(ih * scale * random_scale)

    # force nw a nh to be an even
    if (nw % 2) == 1:
        nw = nw + 1
    if (nh % 2) == 1:
        nh = nh + 1

    # jitter for slight distort of aspect ratio
    if np.random.rand() < 0.3:
        if np.random.rand() < 0.5:
            nw = int(nw*rd.uniform(.8, 1.0))
        else:
            nh = int(nh*rd.uniform(.8, 1.0))

    image = cv.resize(image, (nw, nh), interpolation=cv.INTER_CUBIC)
    nwiw = nw/iw
    nhih = nh/ih

    # clahe. applied on resized image to save time. but before placing to avoid
    # the influence of homogenous background
    clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    cl = clahe.apply(l)
    limg = cv.merge((cl, a, b))
    image = cv.cvtColor(limg, cv.COLOR_LAB2BGR)

    # place image
    dx = rd.randint(0, max(w - nw, 0))
    dy = rd.randint(0, max(h - nh, 0))

    new_image = np.full((h, w, 3), 128, dtype='uint8')
    new_image, crop_coords, new_img_coords = random_crop(
        image, new_image)

    # flip image or not
    flip = rd.random() < .5
    if flip:
        new_image = cv.flip(new_image, 1)

    # distort image
    hsv = np.int32(cv.cvtColor(new_image, cv.COLOR_BGR2HSV))

    # linear hsv distortion
    hsv[..., 0] += rd.randint(-hue_alter, hue_alter)
    hsv[..., 1] += rd.randint(-sat_alter, sat_alter)
    hsv[..., 2] += rd.randint(-val_alter, val_alter)

    # additional non-linear distortion of saturation and value
    if np.random.rand() < 0.5:
        hsv[..., 1] = hsv[..., 1]*rd.uniform(.7, 1.3)
        hsv[..., 2] = hsv[..., 2]*rd.uniform(.7, 1.3)

    hsv[..., 0][hsv[..., 0] > 179] = 179
    hsv[..., 0][hsv[..., 0] < 0] = 0
    hsv[..., 1][hsv[..., 1] > 255] = 255
    hsv[..., 1][hsv[..., 1] < 0] = 0
    hsv[..., 2][hsv[..., 2] > 255] = 255
    hsv[..., 2][hsv[..., 2] < 0] = 0

    image_data = cv.cvtColor(
        np.uint8(hsv), cv.COLOR_HSV2RGB).astype('float32') / 255.0

    # add noise
    if np.random.rand() < 0.15:
        image_data = np.clip(image_data + np.random.rand() *
                             image_data.std() * np.random.random(image_data.shape), 0, 1)

    # correct boxes
    box_data = np.zeros((max_boxes, 5 + NUM_ANGLES3))

    if len(box) > 0:
        np.random.shuffle(box)
        # rescaling separately because 5-th element is class
        box[:, [0, 2]] = box[:, [0, 2]] * nwiw
        # rescale polygon vertices
        box[:, 5::2] = box[:, 5::2] * nwiw
        # rescale polygon vertices
        box[:, [1, 3]] = box[:, [1, 3]] * nhih
        box[:, 6::2] = box[:, 6::2] * nhih

        # mask out boxes that lies outside of croping window
        mask = (box[:, 1] >= crop_coords[0]) & (box[:, 3] < crop_coords[1]) & (
            box[:, 0] >= crop_coords[2]) & (box[:, 2] < crop_coords[3])
        box = box[mask]

        # transform boxes to new coordinate system w.r.t new_image
        box[:, :2] = box[:, :2] - [crop_coords[2], crop_coords[0]] + [new_img_coords[2], new_img_coords[0]]
        box[:, 2:4] = box[:, 2:4] - [crop_coords[2], crop_coords[0]] + [new_img_coords[2], new_img_coords[0]]
        if flip:
            box[:, [0, 2]] = (w-1) - box[:, [2, 0]]

        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] >= w] = w-1
        box[:, 3][box[:, 3] >= h] = h-1
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

        if len(box) > max_boxes:
            box = box[:max_boxes]

        box_data[:len(box), 0:5] = box[:, 0:5]

    #-------------------------------start polygon vertices processing-------------------------------#
    for b in range(0, len(box)):
        boxes_xy = (box[b, 0:2] + box[b, 2:4]) // 2
        for i in range(5, MAX_VERTICES * 2, 2):
            if box[b, i] == 0 and box[b, i + 1] == 0:
                break
            box[b, i:i+2] = box[b, i:i+2] - [crop_coords[2], crop_coords[0]] + [new_img_coords[2], new_img_coords[0]]
            if flip: box[b, i] = (w - 1) - box[b, i]
            dist_x = boxes_xy[0] - box[b, i]
            dist_y = boxes_xy[1] - box[b, i + 1]
            dist = np.sqrt(np.power(dist_x, 2) + np.power(dist_y, 2))
            if (dist < 1): dist = 1

            angle = np.degrees(np.arctan2(dist_y, dist_x))
            if (angle < 0): angle += 360
            iangle = int(angle) // ANGLE_STEP
            if iangle>=NUM_ANGLES: iangle = NUM_ANGLES-1

            if dist > box_data[b, 5 + iangle * 3]: # check for vertex existence. only the most distant is taken
                box_data[b, 5 + iangle * 3]     = dist
                box_data[b, 5 + iangle * 3 + 1] = (angle - (iangle * int(ANGLE_STEP))) / ANGLE_STEP #relative angle
                box_data[b, 5 + iangle * 3 + 2] = 1
    #---------------------------------end polygon vertices processing-------------------------------#
    return image_data, box_data


def random_crop(img, new_img):
    """Creates random crop from img and insert it into new_img

    Args:
        img (numpy array): Image to be cropped
        new_img (numpy array): Image to which the crop will be inserted into.

    Returns:
        tuple: Tuple of image containing the crop, list of coordinates used to crop img and list of coordinates where the crop
        has been inserted into in new_img
    """
    h, w = img.shape[:2]
    crop_shape = new_img.shape[:2]
    crop_coords = [0, 0, 0, 0]
    new_pos = [0, 0, 0, 0]
    # if image height is smaller than cropping window
    if h < crop_shape[0]:
        # cropping whole image [0,h]
        crop_coords[1] = h
        # randomly position whole img along height dimension
        val = rd.randint(0, crop_shape[0]-h)
        new_pos[0:2] = [val, val + h]
    else:
        # if image height is bigger than cropping window
        # randomly position cropping window on image
        crop_h_shift = rd.randint(crop_shape[0], h)
        crop_coords[0:2] = [crop_h_shift - crop_shape[0], crop_h_shift]
        new_pos[0:2] = [0, crop_shape[0]]

    # same as above for image width
    if w < crop_shape[1]:
        crop_coords[3] = w
        val = rd.randint(0, crop_shape[1] - w)
        new_pos[2:4] = [val, val + w]
    else:
        crop_w_shift = rd.randint(crop_shape[1], w)
        crop_coords[2:4] = [crop_w_shift - crop_shape[1], crop_w_shift]
        new_pos[2:4] = [0, crop_shape[1]]

    # slice, insert and return image including crop and coordinates used for cropping and inserting
    # coordinates are later used for boxes adjustments.
    new_img[new_pos[0]:new_pos[1], new_pos[2]:new_pos[3],
            :] = img[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3], :]
    return new_img, crop_coords, new_pos


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5+69)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

    true_boxes[:,:, 5:NUM_ANGLES3 + 5:3] /= np.clip(np.expand_dims(np.sqrt(np.power(boxes_wh[:, :, 0], 2) + np.power(boxes_wh[:, :, 1], 2)), -1), 0.0001, 9999999)
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: grid_size_multiplier}[l] for l in range(1)]
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes + NUM_ANGLES3),
                       dtype='float32') for l in range(1)]


    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0


    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)
        for t, n in enumerate(best_anchor):
            l = 0
            if n in anchor_mask[l]:
                i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                k = anchor_mask[l].index(n)
                c = true_boxes[b, t, 4].astype('int32')

                y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                y_true[l][b, j, i, k, 4] = 1
                y_true[l][b, j, i, k, 5 + c] = 1
                y_true[l][b, j, i, k, 5 + num_classes:5 + num_classes + NUM_ANGLES3] = true_boxes[b, t, 5: 5 + NUM_ANGLES3]
    return y_true


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, is_random):
    """data generator for fit_generator"""
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=is_random)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data.transpose(0, 3, 1, 2), *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes, random):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, random)
