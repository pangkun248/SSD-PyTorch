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
from torchvision import transforms as tvtsf
import torch.nn.functional as F


model = SSD().cuda()
model.load_state_dict(torch.load(cfg.load_path))
model.eval()
dection_imgs = ImageFolder(cfg.test_dir)

# 为每个类名配置不同的颜色
hsv_tuples = [(x / len(cfg.class_name), 1., 1.)for x in range(len(cfg.class_name))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
with torch.no_grad():
    # 开始读取视频源或摄像头
    vid = cv2.VideoCapture(r'D:\BaiduNetdiskDownload\wenyi.avi')
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    while True:
        return_value, frame = vid.read()
        if return_value:
            # out = cv2.VideoWriter(import_param['video_out'], video_FourCC, video_fps, video_size)
            h, w, c = frame.shape
            PIL_img = Image.fromarray(frame[:, :, ::-1])
            img = tvtsf.ToTensor()(PIL_img)
            img = F.interpolate(img.unsqueeze(0), size=(300, 300), mode="nearest").squeeze(0)
            img = tvtsf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
            img_tensor = img[None].cuda()
            start_time = time.time()
            pred_cls, pred_locs = model(img_tensor)
            pred_boxes, pred_labels, pred_scores = filter_box(pred_cls, pred_locs)[0]
            pred_boxes[:, 0::2] *= w
            pred_boxes[:, 1::2] *= h
            end_time = time.time()
            # FPS计算方式比较简单
            fps = 'FPS:%.2f' % (1 / (end_time - start_time))
            # 加载字体文件
            font = ImageFont.truetype(font='FiraMono-Medium.otf',size=np.floor(3e-2 * h + 0.5).astype('int32'))
            # 目标框的厚度
            draw = ImageDraw.Draw(PIL_img)
            # 如果没有检测到目标则跳过
            if pred_boxes.shape[0] !=0:
                for (x1, y1, x2, y2), l, s in zip(pred_boxes, pred_labels, pred_scores):
                    content = '{} {:.2f}'.format(cfg.class_name[l], s)
                    label_w, label_h = draw.textsize(content, font=font)
                    draw.rectangle([x1, y1, x2, y2], outline=colors[l], width=3)
                    draw.rectangle([x1, y1 - label_h, x1 + label_w, y1], fill=colors[l])
                    draw.text((x1, y1 - label_h), content, fill=(0, 0, 0), font=font)
            draw.text((1, 1), fps, fill=colors[0], font=font)
            cv_img = np.array(PIL_img)[..., ::-1]
            cv2.imshow('result', cv_img)
            # cv2.waitKey(300)
            # out.write(cv_img)
            cv2.waitKey(1)
