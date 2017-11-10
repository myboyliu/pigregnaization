# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:38:41 2017

@author: lilhope
"""
import os
import cv2
from glob import glob
import numpy as np

def get_img_info(img_path):
    images = os.listdir(img_path)
    means = np.zeros((3,))
    i = 0
    for img in images:
        img_path = os.path.join(img_path,img)
        im_array = cv2.imread(img_path)
        means[0] += np.mean(im_array[:,:,0])
        means[1] += np.mean(im_array[:,:,1])
        means[2] += np.mean(im_array[:,:,2])
        i += 1
    means = means/i
    return means
    
def extract_frame_from_video(src_path,dst_path,spacing=0.1,write=True):
    vidcap = cv2.VideoCapture(src_path)
    #success,image = vidcap.read()
    framgerate = vidcap.get(cv2.CAP_PROP_FPS)
    count = 0
    while True:
        success,image = vidcap.read()
        
        print('Read New frame:',success)
        if not success:
            break
        if not (count % (framgerate* spacing)):
            dst_img = dst_path +'/'+ 'frame_' + str(count).rjust(4,'0') + '.jpg'
            print('writing to:',dst_img)
            cv2.imwrite(dst_img,image)
        count += 1
def preprocess(video_path,data_path):
    video_files = os.listdir(video_path)
    for video_file in video_files:
        index = int(video_file.replace('.mp4',''))
        dst_path = os.path.join(data_path,str(index))
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        src_path = os.path.join(video_path,video_file)
        extract_frame_from_video(src_path,dst_path)
    
if __name__=='__main__':
    #img_path = 'D:/visual_genome/VG_100K/'
    video_path = 'D:/jd/data/wangyi/Pig_Identification_Qualification_Train/train/'
    dst_path = 'D:/jd/data/extracted_images/'
    preprocess(video_path,dst_path)
    #means = get_img_info(img_path)