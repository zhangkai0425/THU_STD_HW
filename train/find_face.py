import cv2
import os
import glob
#最后剪裁的图片大小
size_m = 48
size_n = 48
def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects
def find_face(modelpath,imgpath,savepath,ID):
    cascade = cv2.CascadeClassifier(modelpath)
    begin = 0
    print("ID = ",ID)
    for i in range(len(imgpath)):
        # cv2读取图像
        path = imgpath[i]
        img = cv2.imread(path)
        rects=detect(img,cascade)
        # 对于人脸数大于1的结果，应当予以舍弃
        if len(rects)!=1:
            continue
        else:
            begin = begin + 1
        for x1,y1,x2,y2 in rects:
                # 调整人脸截取的大小。横向为x,纵向为y
            roi = img[y1+10 :y2+20, x1+10 :x2]
            re = cv2.resize(roi, (size_m, size_n))   
            picpath = savepath + "%s.jpg"%begin
            cv2.imwrite(picpath, re)
    print("成功截取",begin,"张人脸图片,已保存！")