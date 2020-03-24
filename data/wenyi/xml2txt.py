import os
import xml.etree.ElementTree as ET
import random
from os import getcwd

sets = ['train', 'val']

classes = ["__background__", "WhitehairedBanshee", "UndeadSkeleton", "WhitehairedMonster", "SlurryMonster", "MiniZalu",
           "Dopelliwin", "ShieldAxe", "SkeletonKnight", "Zalu", "Cyclone", "SlurryBeggar", "Gerozaru", "Catalog",
           "InfectedMonst", "Gold", "StormRider", "Close", "Door", ]
# 当前路径
data_path = getcwd()


def convert_annotation(image_id):
    in_file = open(image_id.replace('JPGImages', 'Annotations').replace('jpg', 'xml'), 'r')
    out_file = open(image_id.replace('JPGImages', 'labels').replace('jpg', 'txt'), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (xmlbox.find('xmin').text, xmlbox.find('ymin').text, xmlbox.find('xmax').text,
             xmlbox.find('ymax').text)
        out_file.write(str(cls_id) + " " + " ".join(b) + '\n')


trainval_percent = 1
train_percent = 0.9
xmlfilepath = 'Annotations'
total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrain = open('train.txt', 'w')
fval = open('val.txt', 'w')
for i in list:
    name = os.path.join(getcwd(), 'JPGImages', total_xml[i][:-4] + '.jpg')
    # name = total_xml[i][:-4]
    if i in train:
        ftrain.write(name + '\n')
    else:
        fval.write(name + '\n')
ftrain.close()
fval.close()

for image_set in sets:
    # 如果labels文件夹不存在则创建
    if not os.path.exists(data_path + '\labels\\'):
        os.makedirs(data_path + '\labels\\')

    image_ids = open(data_path + '\%s.txt' % (image_set)).read().strip().split()
    for image_id in image_ids:
        convert_annotation(image_id)
