import albumentations as A
import os
import matplotlib.pyplot as plt
from time import sleep
import cv2
from PIL import Image, ImageEnhance
from albumentations.pytorch.transforms import ToTensorV2



#def Load_image(path):
    #image = cv2.imread(path,cv2.IMREAD_UNCHANGED)
    #rotate = [-30,-20,-10,10,20,30]
    #transform_A = A.Compose([A.HorizontalFlip(p=1.0),A.rotate(-15,p=1.0)])
    #transform = A.Compose(
    #    [A.RandomSizedBBoxSafeCrop(width=448, height=336, erosion_rate=0.2)],
    #    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
    #)
#    return image
dataset_save = 'Dataset_Augmented'

def data_augmentation(dataset):
    rotation_degrees = [-30,-20,-10,10,20,30]
    t_degres = []
    
    #TO SAVE IMAGE ERRORS FOR DATA AUGMENTATION
    errors = []
    
    #ROTATION DEGREES
    for i in rotation_degrees:
        transform = A.Compose([
            A.Rotate(limit=i,p=1.0)
        ],
        bbox_params=A.BboxParams(format='yolo',min_visibility=0.5)
        )

        t_degres.append(transform)
    
    #HORIZONTAL FLIP AND ROTATION DEGREES
    for i in rotation_degrees:
        transform = A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Rotate(limit=i,p=1.0)
        ],
        bbox_params=A.BboxParams(format='yolo',min_visibility=0.5)
        )
        t_degres.append(transform)

    #HORIZONTAL FLIP
    transform = A.Compose([
        A.HorizontalFlip(p=1.0)
    ],
    bbox_params=A.BboxParams(format='yolo',min_visibility=0.5)
    )
    
    t_degres.append(transform)
    
    #ORIGINAL FILE
    transform = A.Compose([],
    bbox_params=A.BboxParams(format='yolo',min_visibility=0.5)
    )
    
    t_degres.append(transform)


    for i in dataset:
        #CREATE FOLDER TO SAVE NEW DATASET
        try:
            os.makedirs(os.path.join(dataset_save,i['modality'],'images'))
            os.makedirs(os.path.join(dataset_save,i['modality'],'labels'))
        except:
            pass
        
        modality = i['modality']
        root_path = i['folder']
        data = i['data']
        for j in data:
            filename, extension = os.path.splitext(os.path.basename(j['img']))
            img = cv2.imread(j['img'],cv2.IMREAD_UNCHANGED)
            #img = cv2.imread(j['img'],cv2.IMREAD_ANYDEPTH)
            bounding_boxes = j['bbox']
            new_imgs = []
            
            for k in t_degres:
                try:
                    transformed_img = k(image=img, bboxes=bounding_boxes)
                    new_imgs.append(transformed_img)
                except:
                    errors.append(modality+';'+filename+'.'+extension+'\n')
            
            for n in range(len(new_imgs)):
                new_filename_img = os.path.join(dataset_save,modality,'images',filename+'_'+str(n)+extension)
                new_filename_label = os.path.join(dataset_save,modality,'labels',filename+'_'+str(n)+'.txt')
                cv2.imwrite(new_filename_img,new_imgs[n]['image'])
                with open(new_filename_label,'w') as f:
                    f.write(bbox_to_file(new_imgs[n]['bboxes']))
                f.close()
    with open('image_errors.csv', 'w') as f:
        f.writelines(errors)

def extract_bbox(lines):
    tmp = []
    bounding_boxes = []
    for i in lines:
        bbox = i.replace('\n','')
        bbox = bbox.split(' ')
        category = str(bbox[0])
        bbox = bbox[1:]
        tmp = bbox
        bbox = []
        for i in tmp:
            bbox.append(float(i))
        bbox.append(category)
        bounding_boxes.append(bbox)
    return bounding_boxes

def bbox_to_file(bboxes):
    line = ''
    data = ''
    lines = []
    for i in bboxes:
        if len(i) > 1:
            line = ''
            bbox = i[0:-1]
            for j in bbox:
                line = line + str(j) + ' '
            line = str(i[-1]) + ' ' + line + '\n'
            lines.append(line)
    for i in lines:
        data = data+i
    return data

def LoadDataset(root,modalities):
    dataset = []
    for i in modalities:
        folder = os.path.join(root,i)
        files_dir = os.listdir(os.path.join(folder,'images')) #LIST ALL FILES IN PATH IMAGES
        data = []
        for j in files_dir:
            tmp = j.split('.')[0]
            if os.path.exists(os.path.join(folder,'labels',tmp+'.txt')): #VERIFY IF EXISTS LABEL FOR IMAGE
                with open(os.path.join(folder,'labels',tmp+'.txt'),'r') as f: #OPEN LABELS TO EXTRACT BOUNDIND BOXES
                    lines = f.readlines()
                    bb = extract_bbox(lines)
                    reg = {
                        'img' : os.path.join(folder,'images',j),
                        'label' : os.path.join(folder,'labels',tmp+'.txt'),
                        'bbox' : bb 
                    }
                    data.append(reg)
        dataset.append({
            'modality' : i,
            'folder' : os.path.join(root,i),
            'data' : data
        })
    return dataset

if __name__ == '__main__':
    root_dataset = ['inhouse/DEEC','inhouse/DEI']
    modalities = ['depth','intensity','rgb','thermal']
    
    dataset_DEEC = LoadDataset('inhouse\DEEC',modalities)
    dataset_DEI = LoadDataset('inhouse\DEI',modalities)

    data_augmentation(dataset_DEEC)

    #print(dataset_DEEC[0]['modality'])
    #print(dataset_DEEC[0]['folder'])
    #sleep(5)
    #for i in dataset_DEEC[0]['data']:
    #    print(i)
    #    sleep(2)