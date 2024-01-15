import os
import cv2
import json
from time import sleep

class Dataset:
    def __init__(self):
        self.info = self.Info()
        self.licenses = self.Licenses()
        self.categories = []
        self.images = []
        self.annotations = []

    def setCategories(self,cat):
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

def ConvertDataset(filename, dataset):
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

    with open(filename, 'w') as f:
        f.write(str(JSON))

    #print(JSON)

def Load_YOLO_dataset(images_path,labels_path):
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


def extract_info(id,image,label):
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


if __name__ == '__main__':
    inhouse = Dataset()
    inhouse.setCategories([{"id":0,"name":"person"},{"id":1,"name":"car"}])

    inhouse.info.date_created = "12-2023"
    inhouse.info.description = "INHOUSE DATASET"
    inhouse.info.version = "1.0"
    inhouse.info.year = "2023"
    images_path = r'Dataset_Augmented\rgb\images'
    labels_path = r'Dataset_Augmented\rgb\labels'
    data = Load_YOLO_dataset(images_path,labels_path)

    inhouse.images = data['images']
    inhouse.annotations = data['annotations']

    ConvertDataset('teste.json', inhouse)
    print(inhouse.categories)