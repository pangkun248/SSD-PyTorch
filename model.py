import torch
from torch import nn
import torch.nn.functional as F
import os
import wget
from config import cfg as cfg_


class SSD(nn.Module):
    def __init__(self):
        super().__init__()
        # vgg16中conv1_1到conv4_3再加上一个pool两个conv
        self.vgg = vgg(vgg_base['300'],vgg_pretrain=True)
        # vgg300的特征缩放配置文件
        self.extras = add_extras(extras_base['300'])
        self.l2_norm = L2Norm(512, scale=20)
        self.cls_blocks,self.reg_blocks = cls_reg_blocks()

    def forward(self, x):
        # :x            [10, 3, 300, 300]                输入图片
        # :return:      [10, 8732, 18]   [10, 8732, 4]   SSD网络预测的修正系数与分类概率
        # target_labels (batch_size, num_anchors): 所有框的真实类别
        # target_locs (batch_size, num_anchors, 4): 所有框真实的位置
        features = []
        # vgg16的前23层
        for i in range(23):
            x = self.vgg[i](x)
        s = self.l2_norm(x) # x为现vgg网络最后一个maxpool之前的特征图 torch.Size([10, 512, 38, 38]) 也是后续第一个特征图
        features.append(s)
        # vgg16尾部魔改的部分
        for i in range(23, len(self.vgg)):
            x = self.vgg[i](x)
        features.append(x)  # s.shape [10, 1024, 19, 19]
        # 特征缩放的部分
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                features.append(x)
        # features 最后是整个SSD中出现的6个特征图
        # [10, 512, 38, 38] [10, 1024, 19, 19] [10, 512, 10, 10] [10, 256, 5, 5] [10, 256, 3, 3] [10, 256, 1, 1]
        # 对输入的特征图中每个特征点进行分类及回归(不同特征图特征点对应的输出数是不一样的,以检测框数量为准)
        pred_cls = []
        pred_locs = []
        batch_size = features[0].shape[0]
        # 六个特征图与其对应的分类与定位卷积
        for feature, cls_block, reg_block in zip(features, self.cls_blocks, self.reg_blocks):
            pred_cls.append(cls_block(feature).permute(0, 2, 3, 1))
            pred_locs.append(reg_block(feature).permute(0, 2, 3, 1))
        # 将六个特征图每个特征点上的不同anchor预测得出的各类置信度合并到一起
        # [batch_size, num_anchors*num_classes]) ->  [batch_size, num_anchors, num_classes]
        pred_cls = torch.cat([c.reshape(batch_size, -1) for c in pred_cls], dim=1).view(batch_size, -1, cfg_.num_classes)
        # 将六个特征图每个特征点上的不同anchor预测得出的各个修正系数合并到一起
        # [batch_size, num_anchors*4]  ->  [batch_size, num_anchors, 4]
        pred_locs = torch.cat([l.reshape(batch_size, -1) for l in pred_locs], dim=1).view(batch_size, -1, 4)
        return pred_locs, pred_cls


vgg_base = {                                                 # vgg中第23层↓
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
    # '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
}


def vgg(cfg, vgg_pretrain=True):
    # 创建经过魔改的vgg特征提取层 :原生vgg16去fc层+一个pool两个conv
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            # layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            # 采用没有bn的vgg
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    vgg_layers = nn.ModuleList(layers)
    # 是否加载已经训练好的模型
    if vgg_pretrain:
        # 加载已经训练好的vgg模型,不包括extras_base层,除非你从头开始训练.否则,这个模型可以不用下载
        url = 'https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth'
        # 下载路径
        weight_path = cfg_.vgg16_reducedfc
        if not os.path.exists(weight_path):
            print('模型不存在,下载中')
            wget.download(url=url, out=weight_path)
            print('下载完成')
            print(' --- load weight finish ---')
        vgg_layers.load_state_dict(torch.load(weight_path))
    return vgg_layers


extras_base = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    # '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
}


def add_extras(cfg):
    layers = []     # 额外添加的特征缩放层
    in_channels = 1024  # 是每次conv的输入通道数
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            # 这里进行是否等于'S'的判断作用在于想在特征图19*19 -> 10*10以及10*10 -> 5*5时添加padding以使特征图尺寸能顺利减半
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    # if size == 512: 如果是SSD512的话后面还需要添加两个conv
    #     layers.append(nn.Conv2d(in_channels, 128, kernel_size=1, stride=1))
    #     layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1))
    return nn.ModuleList(layers)


def cls_reg_blocks():
    # 针对不同的特征图创建不同的定位和分类卷积,然后初始化其中的权重
    cls_blocks = nn.ModuleList()
    reg_blocks = nn.ModuleList()
    # 创建针对不同特征图的定位和分类卷积
    for anchors_per_feature, c_out in zip([4, 6, 6, 6, 4, 4], [512, 1024, 512, 256, 256, 256]):
        cls_blocks.append(nn.Conv2d(c_out, anchors_per_feature * cfg_.num_classes, kernel_size=3, stride=1, padding=1))
        reg_blocks.append(nn.Conv2d(c_out, anchors_per_feature * 4, kernel_size=3, stride=1, padding=1))
    # 参数初始化
    for ms in (cls_blocks,reg_blocks):
        for m in ms:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return cls_blocks,reg_blocks


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        # 对于conv4_3后的特征图进行L2归一化,和普通的bn不同,它只针对与channels上的归一化,可以加快网络收敛
        # 详情参考 https://zhuanlan.zhihu.com/p/39399799
        super(L2Norm, self).__init__()
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(n_channels))
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out
