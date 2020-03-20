from model import SSD
from dataset import ImageFolder
from config import cfg
import time
import cv2
import numpy as np
import colorsys
import torch
from PIL import Image, ImageFont, ImageDraw
from utils.box_tools import filter_box

model = SSD().cuda()
model.load_state_dict(torch.load(cfg.load_path))
model.eval()
dection_imgs = ImageFolder(cfg.test_dir)

# 为每个类名配置不同的颜色
hsv_tuples = [(x / len(cfg.class_name), 1., 1.)for x in range(len(cfg.class_name))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
with torch.no_grad():
    for img, img_path in dection_imgs:
        images_tensor = img[None].cuda()
        ac = time.time()
        pred_cls, pred_locs = model(images_tensor)
        pred_boxes, pred_labels, pred_scores = filter_box(pred_cls, pred_locs)[0]
        PIL_img = Image.open(img_path)
        w, h = PIL_img.size
        # 根据每张图片的原始尺寸情况,把相对坐标转换成绝对坐标,方便后面绘制预测框框
        pred_boxes[:, 0::2] *= w
        pred_boxes[:, 1::2] *= h
        print("Detect {} object, inference cost {:.2f} ms".format(len(pred_scores),(time.time()-ac)*1000))

        content_font = ImageFont.truetype(font='FiraMono-Medium.otf', size=16)
        draw = ImageDraw.Draw(PIL_img)
        for (x1, y1, x2, y2), l, s in zip(pred_boxes, pred_labels, pred_scores):
            content = '{} {:.2f}'.format(cfg.class_name[l], s)
            label_w, label_h = draw.textsize(content, content_font)
            draw.rectangle([x1 , y1, x2, y2], outline=colors[l], width=3)
            draw.rectangle([x1, y1 - label_h, x1 + label_w, y1], fill=colors[l])
            draw.text((x1, y1 - label_h), content, fill=(0, 0, 0),font=content_font)
        PIL_img = np.array(PIL_img)[...,::-1]
        cv2.imshow('result',PIL_img)
        cv2.waitKey(0)
