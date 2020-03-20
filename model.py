import torch
from torch import nn
from utils.box_tools import filter_box
import torch.nn.functional as F
import os
import wget
import math


class SSD(nn.Module):
    def __init__(self):
        super().__init__()
        # vgg16中conv1_1到conv4_3再加上一个pool两个conv
        self.vgg = vgg(vgg_base['300'],vgg_pretrain=True)
        # vgg300的特征缩放配置文件
        self.extras = add_extras(extras_base['300'])
        self.l2_norm = L2Norm(512, scale=20)
        self.cls_blocks,self.reg_blocks = cls_reg_blocks()

    def forward(self, x, target_locs=None, target_labels=None):
        # :x            [10, 3, 300, 300]                输入图片
        # :return:      [10, 8732, 18]   [10, 8732, 4]   SSD网络预测的修正系数与分类概率
        # target_labels (batch_size, num_anchors): 所有框的真实类别
        # target_locs (batch_size, num_anchors, 4): 所有框真实的位置
        features = []
        # vgg16的前23层
        for i in range(23):
            x = self.vgg[i](x)
        # x为现vgg网络最后一个maxpool之前的特征图 torch.Size([10, 512, 38, 38]) 也是后续第一个特征图
        s = self.l2_norm(x)  # 对该特征图进行L2归一化
        features.append(s)
        # vgg16尾部魔改的部分
        for i in range(23, len(self.vgg)):
            x = self.vgg[i](x)
        # torch.Size([10, 1024, 19, 19])
        features.append(x)
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
        pred_cls = torch.cat([c.reshape(batch_size, -1) for c in pred_cls], dim=1).view(batch_size, -1, 18)
        # 将六个特征图每个特征点上的不同anchor预测得出的各个修正系数合并到一起
        # [batch_size, num_anchors*4]  ->  [batch_size, num_anchors, 4]
        pred_locs = torch.cat([l.reshape(batch_size, -1) for l in pred_locs], dim=1).view(batch_size, -1, 4)
        if target_locs is None:
            return pred_cls, pred_locs

        # 将特征图输入 conf以及loc网络 获取类别评分以及回归评分
        # 开始计算网络的Loss 类别损失和回归损失
        # 计算分类损失 这里不止正样本(N)的损失,还有前3N个loss最大的背景类损失,最后计算平均损失时是除以N
        # 这里的no_grad可以在反向传播的时候更快一些 因为这里的loss是不需要计算梯度的,只是一个工具人
        with torch.no_grad():
            # 这里如果pred_conf越接近于1则 loss越接近越0,反之如果loss越大说明pred_conf越接近于0
            loss = -F.log_softmax(pred_cls, dim=2)[:, :, 0]
            mask = hard_negative_mining(loss, target_labels, 3)
        # 正样本以及三倍正样本数量的背景anchor上的预测的各个类别置信度
        classification_loss = F.cross_entropy(pred_cls[mask], target_labels[mask], reduction='sum')

        # 计算回归损失,这里只包含正样本的回归损失
        pos_mask = target_labels > 0
        pred_locs = pred_locs[pos_mask]
        target_locs = target_locs[pos_mask]
        smooth_l1_loss = F.smooth_l1_loss(pred_locs, target_locs, reduction='sum')
        num_pos = target_locs.size(0)
        return smooth_l1_loss / num_pos, classification_loss / num_pos

    def forward_with_postprocess(self, images):
        pred_cls, pred_locs = self.forward(images)
        detections = filter_box(pred_cls, pred_locs)
        return detections


vgg_base = {                                                 # vgg中第23层↓
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
    # '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
    #         512, 512, 512],
}
extras_base = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    # '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
}


def vgg(cfg, vgg_pretrain=True):
    # 创建经过魔改的vgg特征提取层 :原生vgg16中conv4_3及其前面部分+一个pool两个conv
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
        vgg_layers.load_state_dict(torch.load(r'D:\py_pro\11\weights\vgg16_reducedfc.pth'))
    return vgg_layers


def add_extras(cfg):
    # 额外添加的特征缩放层
    layers = []
    in_channels = 1024
    flag = False
    for k, v in enumerate(cfg):
        # 这里进行是否等于'S'的判断作用在于想在特征图19*19 -> 10*10以及10*10 -> 5*5时添加padding以使特征图能顺利下降
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    # if size == 512:
    #     layers.append(nn.Conv2d(in_channels, 128, kernel_size=1, stride=1))
    #     layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1))
    return nn.ModuleList(layers)


def cls_reg_blocks():
    # 针对不同的特征图创建不同的定位和分类卷积,然后初始化其中的权重
    cls_blocks = nn.ModuleList()
    reg_blocks = nn.ModuleList()
    for anchors_per_feature, c_out in zip([4, 6, 6, 6, 4, 4], [512, 1024, 512, 256, 256, 256]):
        cls_blocks.append(nn.Conv2d(c_out, anchors_per_feature * 18, kernel_size=3, stride=1, padding=1))
        reg_blocks.append(nn.Conv2d(c_out, anchors_per_feature * 4, kernel_size=3, stride=1, padding=1))
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

def load_vgg_weights():
    # 加载已经训练好的vgg模型,不包括extras_base层,除非你从头开始训练.否则,这个模型可以不用下载
    url = 'https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth'
    weight_path = r'/\Weights\pretrained\vgg16_reducedfc.pth'
    if not os.path.exists(weight_path):
        print('模型不存在,下载中')
        wget.download(url=url,out=weight_path)
        print('下载完成')
        print(' --- load weight finish ---')


def hard_negative_mining(loss, labels, neg_pos_ratio=3):
    """
    loss其实是为所有batch_size中所有anchors的背景类损失
    先计算出所有正样本所在位置及其数量,并将正样本所在位置的背景损失设为无穷小值,以及计算出负样本数量
    然后获取所有anchor的前num_neg个最大背景loss所在的anchor位置并与正样本所在anchor位置合并起来返回
    Args:
        loss (batch_size, num_anchors):   一个batch中所有anchor的背景类损失
        labels (batch_size, num_anchors): 已经赋予值的anchor的label
        neg_pos_ratio: 负例数量/正例数量
    """
    # 统计所有张图片中正样本所在位置和每张图片正样本有多少,以及计算出负样本的数量
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio
    # 这一步目的在于除了将正样本所在的anchor的loss设置为无穷小,还有一个隐藏信息表示
    # 如果其他任意anchor(非正样本)的loss越大说明,这个anchor被预测为背景的概率越小同时也表明被预测为某一其他类的概率越大,即FP
    # 而hard_negative_mining方法就是为了解决此问题而存在的,即尽可能多的降低FP的存在
    loss[pos_mask] = -math.inf
    # 这里连续应用两次sort找出元素在降序之后的位置,可能比较难理解
    # 建议参考 https://blog.csdn.net/LXX516/article/details/78804884 对着图像化的数据来理解
    _, indexes = loss.sort(dim=1, descending=True)  # descending 降序 ,返回 value,index
    _, orders = indexes.sort(dim=1)
    # 获取那些背景类损失最大的前 num_neg个的位置mask,正样本除外
    neg_mask = orders < num_neg
    # 这里返回的mask中只有了正样本以及前num_neg最大背景loss所在anchor的位置才为True 即目标or背景
    return pos_mask | neg_mask