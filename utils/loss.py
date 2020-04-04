import torch
import torch.nn.functional as F


def loss_func(pred_locs, pred_scores, target_locs, target_labels):
    # 开始计算网络的Loss 类别损失和回归损失
    # 计算分类损失 这里不止正样本(N)的损失,还有前3N个loss最大的背景类损失,最后计算平均损失时是除以4N
    # 这里的no_grad可以在反向传播的时候更快一些 因为这里的loss是不需要计算梯度的,只是一个工具人
    with torch.no_grad():
        # 这里如果pred_conf越接近于1则 loss越接近越0,反之如果loss越大说明pred_conf越接近于0
        loss = -F.log_softmax(pred_scores, dim=2)[:, :, 0]
        mask = hard_negative_mining(loss, target_labels, 3)
    # 正样本以及三倍正样本数量的背景anchor上的预测的各个类别置信度
    classification_loss = F.cross_entropy(pred_scores[mask], target_labels[mask], reduction='mean')
    # 计算回归损失,这里只包含正样本的回归损失
    pos_mask = target_labels > 0
    pred_locs = pred_locs[pos_mask]
    target_locs = target_locs[pos_mask]
    # 这里*4是因为torch把loc的四个参数的loss都计算平均了,所以要乘回去
    smooth_l1_loss = F.smooth_l1_loss(pred_locs, target_locs, reduction='mean')*4
    return smooth_l1_loss, classification_loss


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
    loss[pos_mask] = -1
    # 这里连续应用两次sort找出元素在降序之后的位置,可能比较难理解
    # 注意:想要获取降序或升序后的索引位置,那么第一次sort就需要按照降序或升序的顺序来, 而第二个sort必须要升序才可以
    # 建议参考 https://blog.csdn.net/LXX516/article/details/78804884 对着图像化的数据来理解
    _, indexes = loss.sort(dim=1, descending=True)  # descending 降序 ,返回 value,index
    _, orders = indexes.sort(dim=1)
    # 获取那些背景类损失最大的前 num_neg个的位置mask,正样本除外
    neg_mask = orders < num_neg
    # 这里返回的mask中只有了正样本以及前num_neg最大背景loss所在anchor的位置才为True 即目标or背景
    return pos_mask | neg_mask