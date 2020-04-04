from dataset import ListDataset
from config import cfg
import torch
from model import SSD
from utils.eval_tools import Eval
from utils.loss import loss_func
import visdom
import numpy as np
from tqdm import tqdm
from terminaltables import AsciiTable
from torch.utils.data import DataLoader

if __name__ == '__main__':
    train_dataset = ListDataset(path=cfg.train_dir, is_train=True)
    test_dataset = ListDataset(path=cfg.val_dir, is_train=False)
    model = SSD().cuda()
    data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    if cfg.use_adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    if cfg.pretrained_path:
        model.load_state_dict(torch.load(cfg.pretrained_path))
        print('----成功加载SSD权重', cfg.pretrained_path)
    vis = visdom.Visdom(env=cfg.env)
    mAP = 0
    for epoch in range(1, cfg.epoch):
        for images, target_locs, target_labels, image_names in tqdm(data_loader):
            images, target_locs, target_labels = images.cuda(), target_locs.cuda(), target_labels.cuda()
            pred_locs, pred_scores = model(images)
            reg_loss, cls_loss = loss_func(pred_locs,pred_scores,target_locs,target_labels)
            loss = reg_loss + cls_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        vis.line(X=torch.tensor([epoch]), Y=torch.tensor([reg_loss]), win='reg_loss',
                 update=None if epoch == 1 else 'append', opts={'title': 'reg_loss'})
        vis.line(X=torch.tensor([epoch]), Y=torch.tensor([cls_loss]), win='cls_loss',
                 update=None if epoch == 1 else 'append', opts={'title': 'cls_loss'})
        vis.line(X=torch.tensor([epoch]), Y=torch.tensor([loss]), win='loss', update=None if epoch == 1 else 'append',
                 opts={'title': 'loss'})
        eval_result = Eval(model=model, test_dataset=test_dataset)
        ap_table = [["Index", "Class name", "Precision", "Recall", "AP", "F1-score"]]
        for p, r, ap, f1, cls_id in zip(*eval_result):
            ap_table += [[cls_id, cfg.class_name[cls_id], "%.3f" % p, "%.3f" % r, "%.3f" % ap, "%.3f" % f1]]
        print('\n' + AsciiTable(ap_table).table)
        eval_map = round(eval_result[2].mean(), 4)
        print("Epoch %d/%d ---- mAP:%.4f Loss:%.4f" % (epoch, cfg.epoch, eval_map, loss))
        vis.line(X=np.array([epoch]), Y=np.array([eval_map]), win='map', update=None if epoch == 1 else 'append',
                 opts={'title': 'map'})
        if eval_map > mAP:
            mAP = eval_map
            torch.save(model.state_dict(), 'weights/map_%s.pt' % mAP)
