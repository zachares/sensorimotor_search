import yaml
import numpy as np
import scipy
import scipy.misc
import time
import h5py
import sys
import os
import datacollection_util as dc_T

if __name__ == '__main__':

    with open("datacollection_params.yml", 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    dataset_path = cfg['datacollection_params']['logging_folder']
    dest_path = "/scr2/muj_senssearch_dataset_2HZ/"
    # dest_path = "/scr2/muj_senssearch_dataset_10HZ/"

    filename_list = []
    destname_list = []

    for file in os.listdir(dataset_path):
        if file.endswith(".h5"): # and len(filename_list) < 20:
            filename_list.append(dataset_path + file)
            destname_list.append(dest_path + file)

    for idx, filename in enumerate(filename_list):
        # print(idx)
        if ((idx + 1) % 100) == 0:
            print(idx + 1, " of " , len(filename_list), ' finished')

        # dataset = h5py.File(filename, 'a')

        # images = np.array(dataset['image']).astype(np.uint8)
        # depths = np.expand_dims(np.array(dataset['depth']).astype(np.uint8), axis = 1)

        # rgbd = np.concatenate([images, depths], axis = 1)

        # if 'image_s' not in dataset.keys():
        #     chunk_size = (1,) + images[0].shape
        #     dataset.create_dataset('image_s', data= images, chunks = chunk_size)

        # if 'depth_s' not in dataset.keys():
        #     chunk_size = (1,) + depths[0].shape
        #     dataset.create_dataset('depth_s', data= depths, chunks = chunk_size)

        # if 'rgbd' not in dataset.keys():
        #     chunk_size = (1,) + rgbd[0].shape
        #     dataset.create_dataset('rgbd', data= rgbd, chunks = chunk_size)    
        
        # dataset.close()

        dataset = h5py.File(filename, 'r')
        destset = h5py.File(destname_list[idx], 'w')

        for key in dataset.keys():  
            if key == 'image' or key == 'image_s' or key == 'depth' or key == 'depth_s':
                continue

            array = np.array(dataset[key])

            if len(array.shape) > 1:
                chunk_size = (1,) + array[0].shape
                destset.create_dataset(key, data= array, chunks = chunk_size)
            else:
                destset.create_dataset(key, data= array) 

        dataset.close()
        destset.close()                




