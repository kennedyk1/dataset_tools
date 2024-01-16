"""
Script: YOLO2COCO.py
Author: Kennedy Mota
Date: 15/01/2024
Description: Convert dataset from YOLO format to COCO format
"""

import os
import cv2
import json
import shutil

class Dataset:
    def __init__(self, dataset_name:str, image_path:str, label_path:str):
        self.name = dataset_name
        self.images_path = image_path
        self.labels_path = label_path
        self.info = self.Info()
        self.licenses = self.Licenses()
        self.categories = []
        data = Load_YOLO_dataset(image_path, label_path)
        self.yolo_images = data["images"]
        self.yolo_annotations = data["annotations"]
        self.images = []
        self.annotations = []

    def setCategories(self,cat:list):
        for i in cat:
            self.categories.append(
                {
                    "id": int(i['id']),
                    "name": i['name']
                }
            )

    class Info:
        def __init__(self):
            self.description = ""
            self.version = ""
            self.year = 2023
            self.contributor = ""
            self.date_created = ""

    class Licenses:
        def __init__(self):
            self.id = 0
            self.name = "License 1.0"
            self.url = "http://www"


def extract_info(id:int,image:str,label:str):
    img = cv2.imread(image,cv2.IMREAD_UNCHANGED)
    height, width = img.shape[:2]
    bboxes = []
    annotations_info = []

    with open(label,'r') as f:
        lines = f.readlines()
    f.close()

    for i in lines:
        i = i.replace('\n','')
        if len(i.split(' ')) > 3:
            obj_class = int(i.split(' ')[0])
            x = float(i.split(' ')[1])
            y = float(i.split(' ')[2])
            w = float(i.split(' ')[3])
            h = float(i.split(' ')[4])

            bboxes.append(
                {
                    "class": obj_class,
                    "x": int(x * width - 0.5 * w * width),
                    "y": int(y * height - 0.5 * h * height),
                    "w": int(w * width),
                    "h": int(h * height)
                }
            )
    
    for i in bboxes:
        annotations_info.append({
            "id": -1,
            "image_id": id,
            "category_id": i['class'],
            "segmentation": [],
            "bbox": [i['x'],i['y'],i['w'],i['h']],
            "area": i['w']*i['h'],
            "iscrowd": 0
        })

    img_info = {
        "id": id,
        "width": height,
        "height": width,
        "file_name": os.path.basename(image),
        "license": 0,
        "date_captured": "2023"
    }

    return img_info, annotations_info


def Load_YOLO_dataset(images_path:str,labels_path:str):
    images_files = os.listdir(images_path)
    labels_files = os.listdir(labels_path)
    images = []
    annotations = []
    
    id=0
    for i in images_files:
        if os.path.splitext(i)[0]+'.txt' in labels_files:
            img, ann = extract_info(id,os.path.join(images_path,i),os.path.join(labels_path,os.path.splitext(i)[0]+'.txt'))
            images.append(img)
            for j in ann:
                annotations.append(j)
            id = id + 1

    id=0
    for i in annotations:
        i['id'] = id
        id = id + 1
    
    data = {
        "images" : images,
        "annotations" : annotations
    }

    return data

def ConvertDataset(dataset_name:str, image_path:str, label_path:str, dst_folder:str):
    """
    dataset_name:str -> new name for the dataset
    image_path:str -> path of images
    label_path:str -> path of labels
    dst_folder:str -> destination folder
    """

    print("Converting dataset",dataset_name,"to folder",dst_folder)

    dataset = Dataset(dataset_name, image_path, label_path)
    dataset.categories = [{"id": 1,"name": "person"}]
    try:
        os.makedirs(os.path.join(dst_folder,dataset.name,'images'))
        os.makedirs(os.path.join(dst_folder,dataset.name,'annotations'))
    except:
        pass
    #LOAD IMAGES AND ANNOTATIONS
    data = Load_YOLO_dataset(image_path,label_path)
    dataset.images = data['images']
    dataset.annotations = data['annotations']

    info = {
            "description": dataset.info.description,
            "version": dataset.info.version,
            "year": dataset.info.year,
            "contributor": dataset.info.contributor,
            "date_created": dataset.info.date_created
        }
    
    categories = dataset.categories
    licenses = [{"id": 0,"name": "License 1.0","url": "http://www"}]
    images = dataset.images
    annotations = dataset.annotations

    JSON = {
        "info": info,
        "categories": categories,
        "licenses": licenses,
        "images": images,
        "annotations": annotations
    }

    JSON = json.dumps(JSON)

    with open(os.path.join(dst_folder,dataset.name,'annotations','annotations.json'), 'w') as f:
        f.write(str(JSON))

    for i in images:
        shutil.copy(os.path.join(dataset.images_path,i['file_name']),os.path.join(dst_folder,dataset.name,'images'))
    

if __name__ == "__main__":
    #USAGE
    #ConvertDataset( dataset_name , images_path , labels_path , destination_folder )

    # DEI DATASET AUGMENTATION
    ConvertDataset('depth',r'DEI\depth\images',r'DEI\depth\labels','DEI-COCO')
    ConvertDataset('intensity',r'DEI\intensity\images',r'DEI\intensity\labels','DEI-COCO')
    ConvertDataset('rgb',r'DEI\rgb\images',r'DEI\rgb\labels','DEI-COCO')
    ConvertDataset('thermal',r'DEI\thermal\images',r'DEI\thermal\labels','DEI-COCO')

    # DEEC DATASET AUGMENTATION
    ConvertDataset('depth',r'DEEC\depth\images',r'DEEC\depth\labels','DEEC-COCO')
    ConvertDataset('intensity',r'DEEC\intensity\images',r'DEEC\intensity\labels','DEEC-COCO')
    ConvertDataset('rgb',r'DEEC\rgb\images',r'DEEC\rgb\labels','DEEC-COCO')
    ConvertDataset('thermal',r'DEEC\thermal\images',r'DEEC\thermal\labels','DEEC-COCO')