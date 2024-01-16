"""
Script: YOLO-DATA-AUG.py
Author: Kennedy Mota
Date: 15/01/2024
Description: Dataset Expansion with YOLO format with data augmentations
"""

import os
import cv2
import albumentations as A

def data_augmentation(dataset_name:str,images_path:str,labels_path:str,dst_folder:str):
    """
    dataset_name:str -> new name for the dataset
    images_path:str -> path of images
    labels_path:str -> path of labels
    dst_folder:str -> destination folder
    """

    print("Data Augmentation for",dataset_name,"to",dst_folder)

    destination = dst_folder
    dst_folder = os.path.join(destination,dataset_name)
    dataset = LoadDataset(images_path,labels_path)
    rotation_degrees = [-30,-20,-10,10,20,30]
    transformations = []
    
    #TO SAVE IMAGE ERRORS FOR DATA AUGMENTATION
    errors = []
    
    #ROTATION DEGREES
    for i in rotation_degrees:
        transform = A.Compose([
            A.Rotate(limit=i,p=1.0)
        ],
        bbox_params=A.BboxParams(format='yolo',min_visibility=0.5)
        )

        transformations.append(transform)
    
    #HORIZONTAL FLIP AND ROTATION DEGREES
    for i in rotation_degrees:
        transform = A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Rotate(limit=i,p=1.0)
        ],
        bbox_params=A.BboxParams(format='yolo',min_visibility=0.5)
        )
        transformations.append(transform)

    #HORIZONTAL FLIP
    transform = A.Compose([
        A.HorizontalFlip(p=1.0)
    ],
    bbox_params=A.BboxParams(format='yolo',min_visibility=0.5)
    )
    
    transformations.append(transform)
    
    #ORIGINAL FILE
    transform = A.Compose([],
    bbox_params=A.BboxParams(format='yolo',min_visibility=0.5)
    )
    
    transformations.append(transform)

    #CREATE FOLDER TO SAVE NEW DATASET
    try:
        os.makedirs(os.path.join(dst_folder,'images'))
        os.makedirs(os.path.join(dst_folder,'labels'))
    except:
        pass

    for j in dataset:
        filename, extension = os.path.splitext(os.path.basename(j['img']))
        img = cv2.imread(j['img'],cv2.IMREAD_UNCHANGED)
        #img = cv2.imread(j['img'],cv2.IMREAD_ANYDEPTH)
        bounding_boxes = j['bbox']
        new_imgs = []
        
        for k in transformations:
            try:
                transformed_img = k(image=img, bboxes=bounding_boxes)
                new_imgs.append(transformed_img)
            except:
                errors.append(filename+extension+'\n')
        
        for n in range(len(new_imgs)):
            new_filename_img = os.path.join(dst_folder,'images',filename+'_'+str(n)+extension)
            new_filename_label = os.path.join(dst_folder,'labels',filename+'_'+str(n)+'.txt')
            cv2.imwrite(new_filename_img,new_imgs[n]['image'])
            with open(new_filename_label,'w') as f:
                f.write(bbox_to_file(new_imgs[n]['bboxes']))
            f.close()
    with open(destination+'_'+dataset_name+'_images_error.csv', 'w') as f:
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

def LoadDataset(images_path:str,labels_path:str):
    files_dir = os.listdir(images_path) #LIST ALL FILES IN PATH IMAGES
    data = []
    for j in files_dir:
        tmp = j.split('.')[0]
        if os.path.exists(os.path.join(labels_path,tmp+'.txt')): #VERIFY IF EXISTS LABEL FOR IMAGE
            with open(os.path.join(labels_path,tmp+'.txt'),'r') as f: #OPEN LABELS TO EXTRACT BOUNDIND BOXES
                lines = f.readlines()
                bb = extract_bbox(lines)
                reg = {
                    'img' : os.path.join(images_path,j),
                    'bbox' : bb,
                    'file' : os.path.join(labels_path,tmp+'.txt')                    
                }
                data.append(reg)
    return data

if __name__ == '__main__':
    #USAGE
    # data_augmentation( dataset_name , images_path , labels_path , destination_folder )

    # DEI DATASET AUGMENTATION
    data_augmentation('depth',r'inhouse\DEI\depth\images',r'inhouse\DEI\depth\labels','DEI')
    data_augmentation('intensity',r'inhouse\DEI\intensity\images',r'inhouse\DEI\intensity\labels','DEI')
    data_augmentation('rgb',r'inhouse\DEI\rgb\images',r'inhouse\DEI\rgb\labels','DEI')
    data_augmentation('thermal',r'inhouse\DEI\thermal\images',r'inhouse\DEI\thermal\labels','DEI')

    # DEEC DATASET AUGMENTATION
    data_augmentation('depth',r'inhouse\DEEC\depth\images',r'inhouse\DEEC\depth\labels','DEEC')
    data_augmentation('intensity',r'inhouse\DEEC\intensity\images',r'inhouse\DEEC\intensity\labels','DEEC')
    data_augmentation('rgb',r'inhouse\DEEC\rgb\images',r'inhouse\DEEC\rgb\labels','DEEC')
    data_augmentation('thermal',r'inhouse\DEEC\thermal\images',r'inhouse\DEEC\thermal\labels','DEEC')
    
