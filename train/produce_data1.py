# 导入所需要的库
import cv2
import numpy as np
import os 
from find_face import find_face

def video2pic(path,savepath,ID,begin):
    # 读取视频文件，分帧输出，一段视频大概分20帧
    videoCapture = cv2.VideoCapture(path)
    #读帧
    success, frame = videoCapture.read()
    i = 0
    timeF = 6
    j = begin
    while success :
        i = i + 1
        if (i % timeF == 0):
            j = j + 1
            address = savepath + str(j)+ '.jpg'
            cv2.imwrite(address,frame)
            print("ID = ",ID,"path = ",path,'save image:',j)
        success, frame = videoCapture.read()
    return j

def file_name(file_dir):   
    dirs = ()
    for dir in os.walk(file_dir):  
        dirs += dir
    dir_list = dirs[2]
    for path in dir_list:
        if '._' in path:
            dir_list.remove(path)
    return dir_list

def data_producer():
    if not os.path.exists('output'):
        os.makedirs('output')
    for i in range(1,21):
        os.makedirs('./output/output_ID%s'%i)
        name = 'train/ID%s'%i
        savepath = './output/output_ID%s/'%i
        begin = 0
        dir_list = file_name(name)
        for path in dir_list:
            path = name + '/' + path
            begin = video2pic(path,savepath,i,begin)
    for i in range(1,21):
        f = './faceoutput/output_ID%s'%i
        if not os.path.exists(f):
            os.makedirs('./faceoutput/output_ID%s'%i)
        savepath = './faceoutput/output_ID%s/'%i
        img_name = file_name('output/output_ID%s'%i)
        imgpath = []
        for j in range(len(img_name)):
            imgpath.append('output/output_ID%s/'%i + img_name[j])
        modelpath = "pretrained_frontalface.xml"
        find_face(modelpath,imgpath,savepath,i)