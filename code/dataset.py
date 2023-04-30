import os
import time
from PIL import Image
import imageio
import pickle
from tqdm.notebook import tqdm

import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.cm import get_cmap



def load_data_from_files(store_data=False, resize_factor=1, base_dir=os.path.join(os.getcwd(), "../")):
    data_dir = os.path.join(base_dir, "data/bottles/")
    rgb_dir = os.path.join(data_dir, "rgb")
    pose_dir = os.path.join(data_dir, "pose")
    
    train_img_files = []
    train_pose_files = []
    val_img_files = []
    val_pose_files = []
    test_pose_files = []
    
    split = 0

    for filename in sorted(os.listdir(rgb_dir)):
        if (filename.endswith(".png")):
            if filename.split('_')[1] == 'train':
                train_img_files.append(os.path.join(rgb_dir, filename))
                train_pose_files.append(os.path.join(pose_dir, filename.split('.')[0] + ".txt"))
            elif filename.split('_')[1] == 'val':
                if split < 50:
                    train_img_files.append(os.path.join(rgb_dir, filename))
                    train_pose_files.append(os.path.join(pose_dir, filename.split('.')[0] + ".txt"))
                else: 
                    val_img_files.append(os.path.join(rgb_dir, filename))
                    val_pose_files.append(os.path.join(pose_dir, filename.split('.')[0] + ".txt"))
                
                split += 1

    for filename in sorted(os.listdir(pose_dir)):
        if (filename.endswith(".txt")):
            if filename.split('_')[1] == 'test':
                test_pose_files.append(os.path.join(pose_dir, filename))

    num_train_files = len(train_img_files)
    num_val_files = len(val_img_files)
    num_test_files = len(test_pose_files)
    
    train_data_dict = {'images':[], 'poses':[]}
    val_data_dict = {'images':[], 'poses':[]}
    test_data_dict = {'images':[], 'poses':[]}
    
    
    # load camera intrinsic matrix and bbox matrix
    cam_int_mat = np.loadtxt(os.path.join(data_dir, "intrinsics.txt"))
    bbox_mat = np.loadtxt(os.path.join(data_dir, "bbox.txt"))
    
    if resize_factor != 1:
        img = Image.open(train_img_files[0])
        new_size = (img.size[0] // resize_factor, img.size[1] // resize_factor)
        
        cam_int_mat[0:2] = cam_int_mat[0:2] / resize_factor

    for i in tqdm(range(num_train_files)):
        img = Image.open(train_img_files[i])
        if resize_factor != 1:
            img = img.resize(new_size)

        rgb_img = np.array(img) / 255.0
        pose_mat = np.loadtxt(train_pose_files[i])

        train_data_dict['images'].append(rgb_img)
        train_data_dict['poses'].append(pose_mat)

    for i in tqdm(range(num_val_files)):
        img = Image.open(val_img_files[i])
        if resize_factor != 1:
            img = img.resize(new_size)

        rgb_img = np.array(img) / 255.0
        pose_mat = np.loadtxt(val_pose_files[i])

        val_data_dict['images'].append(rgb_img)
        val_data_dict['poses'].append(pose_mat)

    train_data_dict['images'] = np.array(train_data_dict['images'])
    train_data_dict['poses'] = np.array(train_data_dict['poses'])
    
    val_data_dict['images'] = np.array(val_data_dict['images'])
    val_data_dict['poses'] = np.array(val_data_dict['poses'])

    for i in tqdm(range(num_test_files)):
        pose_mat = np.loadtxt(test_pose_files[i])
        test_data_dict['poses'].append(pose_mat)

    test_data_dict['poses'] = np.array(test_data_dict['poses'])

    if store_data:
        # store the train and test data into separate pickle files
        with open('../data/bottles/pkl_files/train_data_pkl.pickle', 'wb') as handle:
            pickle.dump(train_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('../data/bottles/pkl_files/test_data_pkl.pickle', 'wb') as handle:
            pickle.dump(test_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return train_data_dict, val_data_dict, test_data_dict, cam_int_mat, bbox_mat