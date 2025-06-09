import json
import shutil
import os
import pandas as pd
import random
import numpy as np
#coco
# 10類的josn類的josn
with open("D:/123/coco2017/annotations/instances_train2017.json", "r") as f:
    content = json.load(f)
f.close()

new_annotations = []
for i in range(len(content.get("annotations"))):
    if(content.get("annotations")[i].get("category_id")<=10):
        new_annotations.append(content.get("annotations")[i])
    
new_annotations_id = []
for i in range(len(new_annotations)):
    new_annotations_id.append(new_annotations[i].get("image_id"))

new_annotations_id=list(set(new_annotations_id))

new_images = []
for i in range(len(content.get("images"))):
    if(content.get("images")[i].get("id") in new_annotations_id):
        new_images.append(content.get("images")[i])

new_categories = content.get("categories")[0:10]

content1 =content

content["images"] = new_images
content["annotations"] = new_annotations
content["categories"] = new_categories

b = json.dumps(content)
f2 = open('D:\\123\\mini-coco\\annotations\\instances_train2017.json', 'w')
f2.write(b)
f2.close()

#csv
data = pd.DataFrame()
new_image_id = []
new_image_name = []
for i in range(len(content.get("images"))):
    new_image_id.append(content.get("images")[i].get("id"))
    new_image_name.append(content.get("images")[i].get("file_name"))

data['ID'] = new_image_id
data['name'] = new_image_name
class1 = []
class2 = []
class3 = []
class4 = []
class5 = []
class6 = []
class7 = []
class8 = []
class9 = []
class10 = []

for i in range(len(content1.get("annotations"))):
    if(content1.get("annotations")[i].get("category_id")==1):
        class1.append(content1.get("annotations")[i].get("image_id"))
    if(content1.get("annotations")[i].get("category_id")==2):
        class2.append(content1.get("annotations")[i].get("image_id"))
    if(content1.get("annotations")[i].get("category_id")==3):
        class3.append(content1.get("annotations")[i].get("image_id"))
    if(content1.get("annotations")[i].get("category_id")==4):
        class4.append(content1.get("annotations")[i].get("image_id"))
    if(content1.get("annotations")[i].get("category_id")==5):
        class5.append(content1.get("annotations")[i].get("image_id"))
    if(content1.get("annotations")[i].get("category_id")==6):
        class6.append(content1.get("annotations")[i].get("image_id"))
    if(content1.get("annotations")[i].get("category_id")==7):
        class7.append(content1.get("annotations")[i].get("image_id"))
    if(content1.get("annotations")[i].get("category_id")==8):
        class8.append(content1.get("annotations")[i].get("image_id"))
    if(content1.get("annotations")[i].get("category_id")==9):
        class9.append(content1.get("annotations")[i].get("image_id"))
    if(content1.get("annotations")[i].get("category_id")==10):
        class10.append(content1.get("annotations")[i].get("image_id"))

class1_id = [0]*len(new_image_id)
class2_id = [0]*len(new_image_id)
class3_id = [0]*len(new_image_id)
class4_id = [0]*len(new_image_id)
class5_id = [0]*len(new_image_id)
class6_id = [0]*len(new_image_id)
class7_id = [0]*len(new_image_id)
class8_id = [0]*len(new_image_id)
class9_id = [0]*len(new_image_id)
class10_id = [0]*len(new_image_id)

for i in range(len(class1)):
    class1_id[new_image_id.index(class1[i])]=1
for i in range(len(class2)):
    class2_id[new_image_id.index(class2[i])]=1
for i in range(len(class3)):
    class3_id[new_image_id.index(class3[i])]=1
for i in range(len(class4)):
    class4_id[new_image_id.index(class4[i])]=1
for i in range(len(class5)):
    class5_id[new_image_id.index(class5[i])]=1
for i in range(len(class6)):
    class6_id[new_image_id.index(class6[i])]=1
for i in range(len(class7)):
    class7_id[new_image_id.index(class7[i])]=1
for i in range(len(class8)):
    class8_id[new_image_id.index(class8[i])]=1
for i in range(len(class9)):
    class9_id[new_image_id.index(class9[i])]=1
for i in range(len(class10)):
    class10_id[new_image_id.index(class10[i])]=1

data['class1'] = class1_id
data['class2'] = class2_id
data['class3'] = class3_id
data['class4'] = class4_id
data['class5'] = class5_id
data['class6'] = class6_id
data['class7'] = class7_id
data['class8'] = class8_id
data['class9'] = class9_id
data['class10'] = class10_id

csvPath='D:\\123\\mini-coco\\train.csv'
data.to_csv(csvPath,encoding="utf_8_sig")

n = 24
train240 = []
full_data = data

for col_index in range(3, 13):
    full_data_filtered = full_data[full_data.iloc[:, col_index] == 1].iloc[:, 0:3]
    X_sample = np.random.choice(full_data_filtered.iloc[:, 0], n, replace=False)
    X_indices = full_data[full_data.iloc[:, 0].isin(X_sample)].index.tolist()
    full_data = full_data.drop(X_indices)
    train240.extend(X_indices)

df = pd.DataFrame(train240, columns=["Index"])
df.to_csv("D:/123/mini-coco/train240.csv", index=False)


# 拆分
# df = pd.read_csv('D:/123/mini-coco/val60.csv')
# data = pd.read_csv('D:/123/mini-coco/val.csv')

for i in range(len(df['x'])):
    shutil.copy(os.path.join('D:\\123\\coco2017\\train2017',str(data['name'][df['x'][i]])),'D:\\123\\mini-coco\\mini-coco\\train2017')

#VOC
_list=os.listdir(r"D:\123\VOC2012_train_val\VOC2012_train_val\SegmentationClass")
x = random.sample(_list, 360)
for i in range(240):
    shutil.copy(os.path.join('D:\\123\\VOC2012_train_val\\VOC2012_train_val\\JPEGImages',x[i].split('.png')[0]+'.jpg'),'D:\\123\\mini_voc_seg\\train\\images')
    shutil.copy(os.path.join('D:\\123\\VOC2012_train_val\\VOC2012_train_val\\SegmentationClass',x[i]),'D:\\123\\mini_voc_seg\\train\\masks')

for i in range(60):
    shutil.copy(os.path.join('D:\\123\\VOC2012_train_val\\VOC2012_train_val\\JPEGImages',x[i+239].split('.png')[0]+'.jpg'),'D:\\123\\mini_voc_seg\\val\\images')
    shutil.copy(os.path.join('D:\\123\\VOC2012_train_val\\VOC2012_train_val\\SegmentationClass',x[i+239]),'D:\\123\\mini_voc_seg\\val\\masks')

for i in range(60):
    shutil.copy(os.path.join('D:\\123\\VOC2012_train_val\\VOC2012_train_val\\JPEGImages',x[i+299].split('.png')[0]+'.jpg'),'D:\\123\\mini_voc_seg\\test\\images')
    shutil.copy(os.path.join('D:\\123\\VOC2012_train_val\\VOC2012_train_val\\SegmentationClass',x[i+299]),'D:\\123\\mini_voc_seg\\test\\masks')

#imagenette
_list=os.listdir(r"D:\123\imagenette-160-full\train")
for j in _list:
    x_list=os.listdir(os.path.join(r"D:\123\imagenette-160-full\train",j))
    if not os.path.exists(os.path.join(r"D:\123\imagenette-160\train",j)):
        os.makedirs(os.path.join(r"D:\123\imagenette-160\train",j))
        os.makedirs(os.path.join(r"D:\123\imagenette-160\val",j))
        os.makedirs(os.path.join(r"D:\123\imagenette-160\test",j))
    
    x = random.sample(x_list, 36)
    for i in range(24):
        shutil.copy(os.path.join('D:\\123\\imagenette-160-full\\train',j,x[i]),os.path.join('D:\\123\\imagenette-160\\train',j))

    for i in range(6):
        shutil.copy(os.path.join('D:\\123\\imagenette-160-full\\train',j,x[i+23]),os.path.join('D:\\123\\imagenette-160\\val',j))

    for i in range(6):
        shutil.copy(os.path.join('D:\\123\\imagenette-160-full\\train',j,x[i+29]),os.path.join('D:\\123\\imagenette-160\\test',j))



