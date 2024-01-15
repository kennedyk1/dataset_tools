import os
import cv2
import json

class Dataset:
    def __init__(self, image_path:str, label_path:str):
        self.info = self.Info()
        self.licenses = self.Licenses()
        self.categories = []
        data = Load_YOLO_dataset(image_path, label_path)
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


def ConvertDataset(image_path:str, label_path:str, dataset_name:str):
    try:
        os.makedirs(dataset_name)
    except:
        pass
    #LOAD IMAGES AND ANNOTATIONS
    data = Load_YOLO_dataset(image_path,label_path)

if __name__ == "__main__":
    image_path = r'inhouse\DEEC\rgb\images'
    label_path = r'inhouse\DEEC\rgb\labels'
    dataset_name = 'inhouse_dataset'

    ConvertDataset()