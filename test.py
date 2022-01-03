import numpy as np
import soundfile as sf
import os,json
import utils
import nussl

def test_task1(video_path):
    # 测试1
    result_dict = {}
    for file_name in os.listdir(video_path):
        ## 读取MP4文件中的视频,可以用任意其他的读写库
        print(file_name)
        ## 做一些处理
        path = os.path.join(video_path,file_name)
        ID = utils.predict1(path)
        ## 返回一个ID
        result_dict[file_name]=utils.ID_dict[ID]
    return result_dict

def test_task2(wav_path):
    # 测试2
    result_dict = {}
    for file_name in os.listdir(wav_path):
        id=utils.predict2(wav_path+'/'+file_name)
        print('result is:',id)
        ## 返回一个ID
        result_dict[file_name]=utils.ID_dict[id]

    return result_dict

def test_task3(video_path,result_path):
    # 测试3
    if os.path.isdir(result_path):
        print('warning: using existed path as result_path')
    else:
        os.mkdir(result_path)
    for file_name in os.listdir(video_path):
        ## 读MP4中的图像和音频数据，例如：
        idx = file_name[-7:-4]  # 提取出序号：001, 002, 003.....
        ## 做一些处理
        testpath = video_path + '/' + file_name
        audio_left,audio_middle,audio_right = utils.predict3_new(testpath)
        print(testpath)
        print(audio_left.shape)
        ## 输出结果到result_path
        sf.write(os.path.join(result_path,idx+'_left.wav'),   audio_left, 8000)
        sf.write(os.path.join(result_path,idx+'_middle.wav'), audio_middle, 8000)
        sf.write(os.path.join(result_path,idx+'_right.wav'),  audio_right, 8000)

if __name__=='__main__':

    ## testing task1
    with open('./test_offline/task1_gt.json','r') as f:
        task1_gt = json.load(f)
    task1_pred = test_task1('./test_offline/task1')
    task1_acc = utils.calc_accuracy(task1_gt,task1_pred)
    print('accuracy for task1 is:',task1_acc)   

    ## testing task2
    with open('./test_offline/task2_gt.json','r') as f:
        task2_gt = json.load(f)
    task2_pred = test_task2('./test_offline/task2')
    task2_acc = utils.calc_accuracy(task2_gt,task2_pred)
    print('accuracy for task2 is:',task2_acc)   

    # testing task3
    test_task3('./test_offline/task3','./test_offline/task3_estimate')
    task3_SISDR_blind = utils.calc_SISDR('./test_offline/task3_gt','./test_offline/task3_estimate',permutaion=True)  # 盲分离
    print('strength-averaged SISDR_blind for task3 is:',task3_SISDR_blind)
    task3_SISDR_match = utils.calc_SISDR('./test_offline/task3_gt','./test_offline/task3_estimate',permutaion=False) # 定位分离
    print('strength-averaged SISDR_match for task3 is: ',task3_SISDR_match)

