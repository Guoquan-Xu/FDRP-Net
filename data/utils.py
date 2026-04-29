import os
import torch
import torchvision
import random
import numpy as np

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    for i in range(len(path)):
        assert os.path.isdir(path[i]), '{:s} is not a valid directory'.format(path[i])
    images = []
    for i in range(len(path)):
        for dirpath, _, fnames in sorted(os.walk(path[i])):  # dirpath：当前遍历的目录路径，_：忽略子目录列表（用_表示不使用），fnames：当前目录中的文件名列表
            for fname in fnames:
                if is_image_file(fname):
                    img_path = os.path.join(dirpath, fname)
                    # print(img_path)
                    images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)  #只在这里sorted以便就够了



def transform_augment(img_list,img_size, phase='val', min_max=(-1, 1)):
    totensor = torchvision.transforms.ToTensor()
    hflip = torchvision.transforms.RandomHorizontalFlip()
    Resize=torchvision.transforms.Resize((img_size, img_size))
    imgs = [totensor(Resize(img)) for img in img_list]
    if phase == 'train':
        imgs = torch.stack(imgs, 0)  #将张量列表堆叠成一个新的张量，创建一个新维度
        imgs = hflip(imgs)  #进行上面的堆叠操作是要使所有图像同时进行相同的翻转
        imgs = torch.unbind(imgs, dim=0)  #将堆叠的张量分解回列表
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    return ret_img

# print('trainImgS:')
# get_paths_from_images(["D:/pyhthonProject/DATA/train/LSUI","D:/pyhthonProject/DATA/train/UIEB"])
# print('referenceImgs:')
# get_paths_from_images(["D:/pyhthonProject/DATA/reference/LSUI","D:/pyhthonProject/DATA/reference/UIEB"])