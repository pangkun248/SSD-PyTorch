import math
import torch
import torchvision
import torch.nn.functional as F
from config import cfg
import numpy as np

def loc2box(loc, box):
    # 这里由于可以使用torch的广播机制,所以可以忽略locations比priors多一维
    return torch.cat([
        loc[..., :2] * cfg.center_variance * box[..., 2:] + box[..., :2],
        torch.exp(loc[..., 2:] * cfg.size_variance) * box[..., 2:]
    ], dim=-1)



def box2loc(anchor_targets, anchors_xywh):
    return torch.cat([(
        anchor_targets[..., :2] - anchors_xywh[..., :2]) / anchors_xywh[..., 2:] / cfg.center_variance,
        torch.log(anchor_targets[..., 2:] / anchors_xywh[..., 2:]) / cfg.size_variance], dim=-1)


def box_iou(box_a, box_b, eps=1e-5,is_tensor=True):
    # 计算 N个box与M个box的iou需要使用到torch与numpy的广播特性
    if is_tensor:
        # lt为交叉部分左上角坐标最大值, lt.shape -> (N,M,2), br为交叉部分右下角坐标最小值
        lt = torch.max(box_a[..., :2], box_b[..., :2])
        rb = torch.min(box_a[..., 2:], box_b[..., 2:])
        # 第一个axis是指定某一个box内宽高进行相乘,第二个axis是筛除那些没有交叉部分的box
        # 这个 < 和 all(axis=2) 是为了保证右下角的xy坐标必须大于左上角的xy坐标,否则最终没有重合部分的box公共面积为0
        area_overlap = torch.prod(rb - lt, dim=2) * (lt < rb).all(dim=2)
        # 分别计算bbox_a,bbox_b的面积,以及最后的iou
        area_a = torch.prod(box_a[..., 2:] - box_a[..., :2], dim=2)
        area_b = torch.prod(box_b[..., 2:] - box_b[..., :2], dim=2)
        iou = area_overlap / (area_a + area_b - area_overlap + eps)
    else:
        tl = np.maximum(box_a[..., :2], box_b[..., :2])
        br = np.minimum(box_a[..., 2:], box_b[..., 2:])
        area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
        area_a = np.prod(box_a[..., 2:] - box_a[..., :2], axis=2)
        area_b = np.prod(box_b[..., 2:] - box_b[..., :2], axis=2)
        iou = area_i / (area_a + area_b - area_i)
    return iou


def assign_anchors(target_boxes, target_labels, anchors_xyxy):
    """
    target_boxes:       [target_nums,4]         真实框
    target_labels:      [target_nums]           真实标签
    anchors_xyxy:       [anchor_nums,4]         预测坐标
    anchor_targets:     [anchor_nums,4]         基础anchors要拟合的target_boxes
    anchor_labels:     [anchor_nums]            基础anchors要拟合的target_labels
    """
    #               [1,target_nums,4]      [anchor_nums,1,4]
    # 顺便说一句,这里box_iou中前两个参数,谁在前在后是没有关系的.只要最终利用广播机制扩维到 [anchor_nums,target_nums,4]即可
    ious = box_iou(target_boxes[None, :], anchors_xyxy[:, None],is_tensor=True)
    # 每个anchor与所有target的最大iou,最大iou对应的target索引
    anchor_maxious, anchor_argmaxious = ious.max(1)
    # 每个target与所有anchor的最大iou,最大iou对应的anchor索引
    target_maxious, target_argmaxious = ious.max(0)
    # anchor与target的匹配策略 下面这个for循环就是为了解决冲突问题而存在的
    # 1.每个anchor只能匹配一个target,但每个target可以匹配多个anchor
    # 2.anchor_maxious > iou_threshold的anchor就可以被视为正样本.该anchor的label也匹配为target的label
    # 3.和target IOU最高的那个anchor也被视为正样本,该anchor的label也匹配为target的label
    # 4.如果某个anchor在2与3中label匹配出现冲突的话,则该anchor的label以第3步的为准(主要是为了确保每个target都有一个anchor与其匹配)
    for target_index, anchor_index in enumerate(target_argmaxious):
        anchor_argmaxious[anchor_index] = target_index
    # 填充2是为了确保每个和target有最大IOU的anchor的IOU即使小于iou_threshold也算作正样本
    anchor_maxious.index_fill_(0, target_argmaxious, 2)
    anchor_labels = target_labels[anchor_argmaxious]
    # IOU值小于iou_threshold的被视为背景(和target有最大IOU的anchor除外)
    anchor_labels[anchor_maxious < cfg.iou_threshold] = 0
    anchor_targets = target_boxes[anchor_argmaxious]
    return anchor_targets, anchor_labels


def wh2xy(box):
    # [x, y, w, h] -> [x_min, y_min, x_max, y_max]
    # box.shape torch.Size([8732, 4])
    return torch.cat([box[..., :2] - box[..., 2:] / 2, box[..., :2] + box[..., 2:] / 2], dim=-1)


def xy2wh(box):
    # [x_min, y_min, x_max, y_max] -> [x, y, w, h]
    return torch.cat([(box[..., :2] + box[..., 2:]) / 2, box[..., 2:] - box[..., :2]], dim=-1)


def create_anchors():
    image_size = 300  # 模型输入图片大小
    prior_boxex = []
    feature_maps = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
    min_sizes = [30, 60, 111, 162, 213, 264]
    max_sizes = [60, 111, 162, 213, 264, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    for k, (feature_map_w, feature_map_h) in enumerate(feature_maps):
        for i in range(feature_map_w):
            for j in range(feature_map_h):
                # 每个prior_box中对应的最小正方形box
                size = min_sizes[k]
                cx = (j + 0.5) / feature_map_w
                cy = (i + 0.5) / feature_map_h
                h = w = size / image_size
                prior_boxex.append([cx, cy, w, h])

                # 每个prior_box中对应的最大正方形box
                size = math.sqrt(min_sizes[k] * max_sizes[k])
                h = w = size / image_size
                prior_boxex.append([cx, cy, w, h])

                # 不同特征图对应的不同prior_box种类 从前往后4,6,6,6,4,4
                # 每个prior_box中对应的长宽比为 1/2,1/3,2,3 或者 1/2,2的长方形box
                size = min_sizes[k]
                h = w = size / image_size
                for ratio in aspect_ratios[k]:
                    ratio = math.sqrt(ratio)
                    prior_boxex.append([cx, cy, w * ratio, h / ratio])
                    prior_boxex.append([cx, cy, w / ratio, h * ratio])
    prior_boxex = torch.tensor(prior_boxex)
    # 限制prior_box的范围 幻想不灭次元不倒
    prior_boxex = wh2xy(prior_boxex)
    prior_boxex.clamp_(max=1, min=0)
    prior_boxex = xy2wh(prior_boxex)
    return prior_boxex


def filter_box(pred_cls, pred_locs):
    """
    该方法实际上是一个对网络预测结果的处理(筛选),符合条件的即可以作为输出结果显示
    pred_cls :  SSD网络预测的一个batch中所有anchor的各个类别置信度    [batch_size, num_anchors, num_classes]
    pred_locs : SSD网络预测的一个batch中所有anchor的修正系数         [batch_size, num_anchors, 4]
    priors :    SSD网络一个batch中所有anchor的xywh相对坐标形式       [batch_size, num_anchors, 4]
    """
    anchor_xywh = create_anchors().cuda()
    batches_scores = F.softmax(pred_cls, dim=2)
    pred_boxes = loc2box(pred_locs, anchor_xywh)
    pred_boxes = wh2xy(pred_boxes)
    batch_size = batches_scores.size(0)
    results = []
    for batch_id in range(batch_size):
        processed_boxes = []
        processed_scores = []
        processed_labels = []
        # [n_anchor, n_cls] [n_anchor, 4]   一张图片中所有anchor预测的所有类别置信度,以及坐标
        per_img_scores, per_img_boxes = batches_scores[batch_id], pred_boxes[batch_id]
        for class_id in range(1, per_img_scores.size(1)):  # 从1开始循环是为了跳过背景类
            # 一张图片中所有anchor在某一个类别下的预测情况
            scores = per_img_scores[:, class_id]
            # 过滤掉类别得分小于socre_threshold的pred_box
            mask = scores > cfg.score_threshold
            scores = scores[mask]
            # 如果没有置信度大于 socre_threshold 的pred_box则跳过此类别
            if scores.size(0) == 0:
                continue
            # 获取一张图片中某个类别下所有score满足条件的pred_box,并且对其坐标进行绝对化
            boxes = per_img_boxes[mask, :]

            # 这里使用的是pytorch内置的nms方法,具体原理过程可以参考以下NMS的注释内容
            # ops的其他方法 参考 https://blog.csdn.net/shanglianlm/article/details/102002844
            # 这里输入的scores是没有排序过的,keep为nms之后的排过序的保留的boxes索引
            keep = torchvision.ops.nms(boxes, scores, cfg.iou_nms)
            nmsed_boxes = boxes[keep, :]  # 利用keep获取nms后的pred_boxes
            nmsed_labels = torch.tensor([class_id] * keep.shape[0]).cuda()  # 根据keep的长度创建相应数量的label值
            nmsed_scores = scores[keep]  # 利用keep获取nms后的pred_scores

            processed_boxes.append(nmsed_boxes)
            processed_scores.append(nmsed_scores)
            processed_labels.append(nmsed_labels)

        # 如果一张图片中没有符合条件的pred_box,则返回空
        if len(processed_boxes) == 0:
            processed_boxes = torch.empty(0, 4)
            processed_labels = torch.empty(0)
            processed_scores = torch.empty(0)
        # 将一张图片中所有的box,label,score合并起来
        else:
            processed_boxes = torch.cat(processed_boxes, 0)
            processed_labels = torch.cat(processed_labels, 0)
            processed_scores = torch.cat(processed_scores, 0)
        results.append([processed_boxes, processed_labels, processed_scores])
    return results