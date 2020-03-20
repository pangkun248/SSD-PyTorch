from dataset import ListDataset
from config import cfg
import torch
from Dataloader import Our_Dataloader
from model import SSD
from utils.eval_tools import Eval
import visdom
import numpy as np
from tqdm import tqdm
from terminaltables import AsciiTable


if __name__ == '__main__':
    train_dataset = ListDataset(path=cfg.train_dir, is_train=True)
    test_dataset = ListDataset(path=cfg.train_dir, is_train=False)
    model = SSD().cuda()
    data_loader = Our_Dataloader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    if cfg.use_adam:
        optimizer = torch.optim.Adam(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,momentum=0.9)
    vis = visdom.Visdom(env='SSD')
    mAP = 0
    for epoch in range(1,10):
        for images, target_loc, target_label, image_names in tqdm(data_loader):
            images, target_loc, target_label = images.cuda(), target_loc.cuda(), target_label.cuda()
            reg_loss, cls_loss = model(images, target_loc, target_label)
            reg_loss = reg_loss.mean()
            cls_loss = cls_loss.mean()
            loss = reg_loss + cls_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        vis.line(X=torch.tensor([epoch]),Y=torch.tensor([reg_loss]), win='reg_loss', update=None if epoch==1 else 'append', opts={'title': 'reg_loss'})
        vis.line( X=torch.tensor([epoch]),Y=torch.tensor([cls_loss]), win='cls_loss', update=None if epoch==1 else 'append', opts={'title': 'cls_loss'})
        vis.line(X=torch.tensor([epoch]), Y=torch.tensor([loss]), win='loss', update=None if epoch==1 else 'append', opts={'title': 'loss'})
        eval_result = Eval(model=model, test_dataset=test_dataset)
        ap_table = [["Index", "Class name", "Precision", "Recall", "AP", "F1-score"]]
        for p, r, ap, f1, cls_id in zip(*eval_result):
            ap_table += [[cls_id, cfg.class_name[cls_id], "%.3f" % p, "%.3f" % r, "%.3f" % ap, "%.3f" % f1]]
        print('\n' + AsciiTable(ap_table).table)
        eval_map = round(eval_result[2].mean(),4)
        print("Epoch %d/%d ---- mAP:%.4f Loss:%.4f" % (epoch, cfg.epoch, eval_map, loss))
        vis.line( X=np.array([epoch]), Y=np.array([eval_map]),win='map', update=None if epoch == 1 else 'append',
                 opts={'title': 'map'})
        if eval_map > mAP:
            mAP = eval_map
            torch.save(model.state_dict(),'weights/map_%s.pt' % mAP)