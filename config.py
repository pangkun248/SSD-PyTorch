class Config:
    env = 'SSD'  # Visdom可是坏环境名称
    # 训练集路径,验证集路径(mAP相关)
    train_dir = r'D:\py_pro\11\data\wenyi\train.txt'
    val_dir = r'D:\py_pro\11\data\wenyi\val.txt'
    test_dir = r'D:\py_pro\11\test'
    class_name = ("__background__", "WhitehairedBanshee", "UndeadSkeleton", "WhitehairedMonster", "SlurryMonster",
                  "MiniZalu", "Dopelliwin", "ShieldAxe", "SkeletonKnight", "Zalu", "Cyclone", "SlurryBeggar",
                  "Gerozaru", "Catalog", "InfectedMonst", "Gold", "StormRider", "Close", "Door",)
    load_path = r'D:\py_pro\11\weights\map_0.8266.pt'  # 基于此模型权重训练
    cgg16_path = r'D:\py_pro\SSD-PyTorch\weights\vgg16_reducedfc.pth'
    # 网络输入尺寸
    height = 300
    num_workers = 2  # 取决于你的cpu核数,比如9400F是六核的,建议2~4之间会比较好
    test_num_workers = 2  # 同上

    use_adam = True  # 是否使用Adam优化方式
    weight_decay = 0.0005  # 权重衰减系数
    lr_decay = 0.1  # 每隔指定epoch学习率下降的倍数
    lr = 1e-3  # 初始学习率
    epoch = 14  # 训练的轮数
    batch_size = 10

    # center_variance(xy)和size_variance(wh)是可以调整loc损失在整体loss中的比例
    # 参考https://github.com/weiliu89/caffe/issues/629
    center_variance = 0.1
    size_variance = 0.2
    iou_threshold = 0.5  # AnchorTargetCreator方法中的判断正负样本的IOU参数
    # 类别置信度阈值,网络输出最终结果时会过滤掉小于此值的pred_box
    # 也可以作为权衡recall和precision的指标,该值越大,recall越大,precision越小.反之同样
    score_threshold = 0.5
    iou_nms = 0.45  # 此值为最后进行NMS操作时其中的IOU参数


cfg = Config()
