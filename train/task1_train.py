import face_recognition
import numpy as np
import os 
import random
from produce_data1 import data_producer
def feature(path):
    pic = face_recognition.load_image_file(path)
    if len(face_recognition.face_encodings(pic))>=1:      
        encoding = face_recognition.face_encodings(pic)[0]
    else:
        encoding = []
    return encoding
    
if __name__ == '__main__':
    # 生成训练数据
    data_producer()
    facefeature=np.zeros([20,128])
    print("begin training")
    for i in range(1,21):
        path = "faceoutput/output_ID%s/"%i
        paths = os.listdir(path)
        N = 0
        for filename in paths:
            picpath = path + filename
            eigen = feature(picpath)
            if len(eigen)!=0:
                N = N + 1
                facefeature[i-1,:] += eigen
        print(N)
        facefeature[i-1,:] = facefeature[i-1,:]/N
    np.save('facefeature.npy', facefeature)
    print("training succeed!")
