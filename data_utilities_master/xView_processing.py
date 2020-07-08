#export
from fastai import *
from fastai.vision import *
import numpy as np
# import geopandas as gpd
from matplotlib import pyplot as plt
import wv_util as wv
import aug_util as aug
import csv
import json
from tqdm import tqdm
import imageio
import os
import pandas as pd

def classes2index():
    class2index = dict()
    c = 0
    for uni_cls in np.unique(df['Class']):
        class2index[uni_cls] = c
        c += 1
    return class2index

original_dataset_path = 'C:/Users/facua/Desktop/Boeing/train_images/'
original_geojson_path = 'C:/Users/facua/Desktop/Boeing/xView_train.geojson'
class_labels_path = 'C:/Users/facua/Desktop/Boeing/xview_class_labels.txt'

# Look at an image from original dataset
image_name = '10.tif'
image_str = original_dataset_path + image_name
arr = wv.get_image(image_str)

plt.figure(figsize=(20,20))
plt.axis('off')
plt.imshow(arr)

# loading all coords, images, classes
coords, images, classes = wv.get_labels(original_geojson_path)
# only for specific image
icoords = coords[images == image_name]
iclasses = classes[images == image_name].astype(np.int64)


labels = {}
with open(class_labels_path) as f:
    for row in csv.reader(f):
        labels[int(row[0].split(":")[0])] = row[0].split(":")[1]

# visualize the same image with the bboxes

typed_id = aug.draw_bboxes(arr, icoords)
plt.figure(figsize=(20,20))
plt.axis('off')
plt.imshow(typed_id)


# Remove invalid Type IDs (aka type IDs without cooresponding labels)
i, = np.where((classes == 75) | (classes == 82))
ximages = np.delete(images, i) # list of object images
xcoords = np.delete(coords, i, axis=0).astype('int') # list of bbox coords and cast elements to int type
xclasses = np.delete(classes, i) # list of object type IDs
len(i) # number of indicies removed


# remove references to non-existent image 1395.tif
i, = np.where(ximages == '1395.tif')
ximages = np.delete(ximages, i) # list of object images
xcoords = np.delete(xcoords, i, axis=0) # list of bbox coords
xclasses = np.delete(xclasses, i) # list of object type ids
len(i) # number of indicies removed


# identify coords for same image with different classes

# first identify indices where there are the same coords
vals, inverse, count = np.unique(xcoords, return_inverse=True, return_counts=True, axis=0)

idx_vals_repeated = np.where(count > 1)[0]
vals_repeated = vals[idx_vals_repeated]

rows, cols = np.where(inverse == idx_vals_repeated[:, np.newaxis])
_, inverse_rows = np.unique(rows, return_index=True)
res = np.split(cols, inverse_rows[1:])
print(f'Number of indicies where coordinates match: {len(res)}')

# next identify where same coords also are in same images
invalid_match = [match for match in res if ximages[match[0]] == ximages[match[1]]]
print(f'Number of indicies where coordinates match in same image: {len(invalid_match)}')
num_conflict = 1
for invalid in invalid_match:
    print(f'Invalid Conflict {num_conflict}')
    num_match = 1
    for match in invalid:
        print(f'  Match {num_match} -> image: {ximages[match]}, Coords: {xcoords[match]}, Class: {xclasses[match]}')
        num_match += 1
    num_conflict += 1


# after taking a look at competing labels -> 2571.tif conflict should be bus (type_id=19)
# 1141.tif conflict should be removed. Bbox doesn't coorespond to any classification
i = np.array([invalid_match[0][0], invalid_match[1][0], invalid_match[1][1]])
ximages = np.delete(ximages, i) # list of object images
xcoords = np.delete(xcoords, i, axis=0) # list of bbox coords
xclasses = np.delete(xclasses, i) # list of object type ids
len(i) # number of indicies removed


train_str = original_dataset_path

np.unique(ximages)

# chip_shape = (32*22,32*22)
chip_shape = (700,700)
chipped_dataset_path = 'C:/Users/facua/Desktop/Boeing/chipped2_'+str(chip_shape[0])+'x'+str(chip_shape[1])+'/'
# create directory if it doesn't already exist
if not os.path.exists(chipped_dataset_path):
    os.makedirs(chipped_dataset_path)

# Create the chipped dataset

# wv.chip_image ensure there are no coordinates outside of image/chip pixel values (become 0 or max img pixel size)
# Remove any objects which have invalid bbox coordinates
# Invalid bbox coordinates include:
# - ymax minus ymin is negative
# - xmax minus ymin is negative
# - bbox with less than 5 pixel width or height

chippedimages = []
chippedannotations = []
num_chips = []
image_id = 1
for image_name in tqdm(np.unique(ximages)):
    # get image to chip
    image_str = train_str + image_name
    arr = wv.get_image(image_str)
    # get coords and classes only for specific image
    icoords = xcoords[ximages == image_name]
    iclasses = xclasses[ximages == image_name].astype('int')
    # chip the image
    c_img, c_box, c_cls = wv.chip_image(img = arr, coords=icoords, classes=iclasses, shape=chip_shape)
    chip_count = 0
    for chip_idx, chip in enumerate(c_img, 1):
        # ensure the chip is valid (ie, not blank) before saving and recording details in dict structure
        blank = np.zeros(chip.shape, dtype='int')
        comparison = blank==chip
        if comparison.all():
            continue
        else:
            chip_name = image_name[:-4]+'_'+str(chip_idx)+'.png'
            chippedimages.append({'file_name': chip_name, 'id': image_id})
            imageio.imwrite(chipped_dataset_path+chip_name, chip)
            chip_count += 1
        for o_box, o_cls in zip(c_box[chip_idx-1], c_cls[chip_idx-1]):
            # ensure the objects coords and classes are valid before recording in dict structure
            xmin, ymin, xmax, ymax = o_box
            if o_cls == 0:
                continue
            elif ((xmax - xmin) < 5) | ((ymax - ymin) < 5):
                continue
            else:
                obox = [int(xmin), int(ymin), int(xmax-xmin), int(ymax-ymin)]
                chippedannotations.append({'image_id': image_id, 'bbox': obox, 'category_id': int(o_cls)})
        image_id += 1
    num_chips.append(chip_count)
assert image_id-1 == sum(num_chips), 'Number of image IDs does not match number of chips created'


# Create COCO-style JSON for chipped dataset
chippedcategories = [dict(id=i, name=l) for i, l in labels.items()]
text = json.dumps({'images': chippedimages, 'type': 'instances', 'annotations': chippedannotations, 'categories': chippedcategories}, indent=4)
with open(chipped_dataset_path+'chippedxView.json', 'w') as outfile:
    outfile.write(text)


# read it back in to check it worked
annots = json.load(open(chipped_dataset_path+'chippedxView.json'))

with open(chipped_dataset_path+'chippedxView.json') as f:
    data = json.load(f)
    
coords = np.zeros((len(data['annotations']),4))
chips = np.zeros((len(data['annotations']),1),dtype="object")
classes = np.zeros((len(data['annotations']),1))
width = np.zeros((len(data['annotations']),1))
height = np.zeros((len(data['annotations']),1))
x = np.zeros((len(data['annotations']),1))
y = np.zeros((len(data['annotations']),1))
labels = np.zeros((len(data['annotations']),6))

#Image width and Height
wh = chip_shape[0]

for i in tqdm(range(len(data['annotations']))):
    width[i] = data['annotations'][i]['bbox'][2]/wh
    height[i] = data['annotations'][i]['bbox'][3]/wh
    x[i] = (2*data['annotations'][i]['bbox'][0]+data['annotations'][i]['bbox'][2])/(wh*2)
    y[i] = (2*data['annotations'][i]['bbox'][1]+data['annotations'][i]['bbox'][3])/(wh*2)
    classes[i] = data['annotations'][i]['category_id']
    labels[i] = [data['annotations'][i]['image_id'], classes[i], x[i], y[i], width[i], height[i]]


df = pd.DataFrame(labels, columns = ['Image','Class','X','Y','Width','Height'])
df[['X','Y','Width','Height']] = df[['X','Y','Width','Height']].astype('float').round(3)
df['Class'] = df['Class'].map(classes2index())
df['Class'] = df['Class'].astype(int)


id2img = dict()
for i in range(len(data['images'])):
    id2img[data['images'][i]['id']] = data['images'][i]['file_name']
    
df['Image'] = df['Image'].map(id2img)


unique_images = np.unique(df['Image'])

fname = chipped_dataset_path
for imageName in tqdm(unique_images):
    df.loc[df['Image'] == imageName].loc[:, ['Class','X','Y','Width','Height']].to_csv(fname + imageName.split('.')[0]+'.txt', header=None, index=None, sep=' ', mode='a')

fname = open('validImages.txt')
pictures = []
with open(chipped_dataset_path+'val_xview.txt','w') as outfile:
    for picture in fname:
        pictures.append(picture.split('\n')[0])
        outfile.write(chipped_dataset_path+str(picture))
    outfile.close()
    
files = []
for file in os.listdir(chipped_dataset_path):
    if file.endswith('.png'):
        files.append(file)

train_images = np.array(files)
val_images = np.array(pictures)

train_imagesDF = pd.DataFrame(train_images)
val_imagesDF = pd.DataFrame(val_images)

with open(chipped_dataset_path+'train_xview.txt','w') as outfile:
    for element in train_imagesDF[~train_imagesDF[0].isin(val_imagesDF[0])][0].values:
          outfile.write(chipped_dataset_path+str(element))
          outfile.write('\n')
    outfile.close()


train_images, train_lbl_bbox = get_annotations(chipped_dataset_path+'chippedxView.json')
# Note: get_annotations is only reading in the chips with objects in them

print(len(train_images))
print(train_images[0:8])


#list of num of labels for specific image
x = list(range(len(train_lbl_bbox[1][1])))


img = open_image(chipped_dataset_path+train_images[1])
bbox = ImageBBox.create(*img.size, train_lbl_bbox[1][0], x, classes=train_lbl_bbox[1][1])
img.show(figsize=(20,20), y=bbox)
