import os,ffmpeg,sys

from numpy.lib.utils import deprecate
import numpy as np
import metrics
import soundfile as sf
import json
import face_recognition
import nussl
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import math
import cv2
from PIL import Image

from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio
from moviepy.editor import *

ID_dict ={1:'ID1',2:'ID2',3:'ID3',4:'ID4',5:'ID5',6:'ID6',7:'ID7',8:'ID8',9:'ID9',10:'ID10'
         ,11:'ID11',12:'ID12',13:'ID13',14:'ID14',15:'ID15',16:'ID16',17:'ID17',18:'ID18',19:'ID19',20:'ID20'}

def calc_accuracy(gt_dict,pred_dict):
    ## 计算准确率
    ## 输入:    gt_dict: ground-true字典，type=dict
    ##          pred_dict: 预测结果字典, type=dict
    ## 输出:    准确率，type=float, value_range=[0,1]
    correct = 0 
    for key,value in gt_dict.items():
        if gt_dict[key]==pred_dict[key]:
            correct+=1
    return correct/len(gt_dict)

def calc_SISDR(gt_dir,estimate_dir,permutaion=False):
    ## 计算SISDR指标
    ## 输入:    gt_dir: ground-true文件路径，type=str 如： './test_offline/task3_gt'
    ##          estimate_dir: 估计结果文件路径，type=str 如： './test_offline/task3_estimate'
    ##          permutaion: 是否允许排序，如果是则计算盲分离指标, type=bool
    ## 输出:    si-sdr指标加权均值, type=float
    si_sdr_list = []
    idx_set = set([x.split('_')[0] for x in os.listdir(gt_dir)])
    idx_set = sorted(idx_set,key=lambda x: int(x))
    for file in idx_set:
        sources_list = []
        estimates_list=[]
        strength_list=[]
        person_list=['_left.wav','_middle.wav','_right.wav']
        for idx, appendix in enumerate(person_list):
            est_audio_temp = read_audio(os.path.join(estimate_dir,file+appendix))
            gt_audio_temp = read_audio(os.path.join(gt_dir,file+appendix))[:len(est_audio_temp)]
            strength_list.append((gt_audio_temp**2).sum()**(1/2))
            sources_list.append(nussl.AudioSignal(audio_data_array=gt_audio_temp, sample_rate=44100))
            estimates_list.append(nussl.AudioSignal(audio_data_array=est_audio_temp, sample_rate=44100))

        new_bss = nussl.evaluation.BSSEvalScale(sources_list, estimates_list,compute_permutation=permutaion)
        weight = np.stack(strength_list)
        weight = (weight/weight.sum())*3   # weight should sum up to 3 because 3 audios are considered
        scores = new_bss.evaluate()
        for idx in range(len(sources_list)):
            si_sdr_list.append(scores['source_%d'%idx]['SI-SDR'][0]*weight[idx])
    si_sdr = np.stack(si_sdr_list)
    return si_sdr.mean()
       
def read_video(file):
    ## 读取文件中的视频
    ## 输入:    file: 文件名，type=str 如： './test_offline/task1/001.mp4'
    ## 输出:    video: 视频数据，type=numpy.ndarray, shape=(F,H,W,3) F为总帧数,H为图像高，W为图像宽，3通道RGB
    ##          video_fps: 帧率，type=int, 每秒钟帧数
    probe = ffmpeg.probe(file)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    video_fps = int(video_stream['avg_frame_rate'][:-2])
    input_mp4 = ffmpeg.input(file)
    video_buff, _ = (input_mp4.video
        .output('pipe:', format='rawvideo', pix_fmt='rgb24') 
        .global_args('-loglevel', 'quiet')
        .run(capture_stdout=True)
    )
    video = np.frombuffer(video_buff, np.uint8).reshape([-1, height, width, 3])
    return video,video_fps

def read_audio(file,sr=44100):
    ## 读取文件中的视频
    ## 输入:    file: 文件名，type=str 如： './test_offline/task1/001.mp4'
    ##          sr: 采样率，type=int, 与帧率相同，表示每秒钟采样点个数
    ## 输出:    audio: 音频数据，type=numpy.ndarray, shape=(N,C) N为采样点数量，C为通道数，双通道为2，单通道为1
    probe = ffmpeg.probe(file)
    audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
    audio_channel = audio_stream['channels']
    audio_buffer, _ = (ffmpeg.input(file).audio
        .output('pipe:', format='f32le', acodec='pcm_f32le',ac=audio_channel,ar=str(sr))
        .global_args('-loglevel', 'quiet')
        .run(capture_stdout=True, capture_stderr=True)
    )
    audio = np.frombuffer(audio_buffer,np.float32).reshape([-1,audio_channel])
    if audio_channel==2 and np.array_equal(audio[:,0],audio[:,1]): # 如果两个音轨一致，则只取其中一个
        audio = audio[:0]
    return audio

def generate_combine(source_dir,dst_dir,N=10,T=3.0):
    ## 生成一段多人说话的音频，可以在训练过程中生成一些数据
    ## 输入： source_dir: 原路径，要求内含子路径ID1,ID2...ID20,type=str, 如 './train'
    ##       dst_dir: 目标路径，type=str 如 './train_gen'
    ##       N: 生成视频总数,type=int
    ##       T: 生成视频时长,type=float
    ## 输出：None
    ## 目标文件夹中存有：'audio%03d.wav' 组合的纯音频
    ##                  'video%03d.mp4' 组合的纯视频
    ##                  'combine%03d.mp4' 组合的带音频视频

    if os.path.isdir(dst_dir):
        print('warning: using existed path as result_path')
    else:
        os.mkdir(dst_dir)
    nc = 3 # number of channels, 3个人同时说话
    id_map={} # 记录每段视频所用到的人物ID
    for i in range(N):
        idx = np.random.permutation(20)
        combinec = np.random.rand(nc, 2).astype(np.float32)
        combinec = (combinec / 2 + 0.75)
        audio_list = []
        video_list = []
        id_list = []
        for j in range(nc):
            person_idx = idx[j] + 1
            id_list.append(ID_dict[person_idx])
            mp4_list = os.listdir(os.path.join(source_dir,ID_dict[person_idx]))
            video_idx = np.random.permutation(len(mp4_list))[0]
            video_file = os.path.join(source_dir,ID_dict[person_idx],mp4_list[video_idx])
            video,fps = read_video(video_file)
            audio = read_audio(video_file)

            video_list.append(video[:int(fps*T)])
            audio_list.append(audio[:int(44100*T)])
        # id recording
        id_map['combine%03d.mp4'%(i+1)]=id_list

        # audio only
        audio = np.concatenate(audio_list,axis=-1) @ combinec
        sf.write(os.path.join(dst_dir,'audio%03d.wav'%(i+1)),audio,44100)

        # video only
        video = np.concatenate(video_list,axis=2)
        F,H,W,C = video.shape
        video_only = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(W, H))
            .output(os.path.join(dst_dir,'video%03d.mp4'%(i+1)), pix_fmt='yuv420p')
            .global_args('-loglevel', 'quiet')
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
        video_only.stdin.write(video.astype(np.uint8).tobytes())
        video_only.stdin.close()
        video_only.wait()

        ## combine
        video_in = ffmpeg.input(os.path.join(dst_dir,'video%03d.mp4'%(i+1))).video
        audio_in = ffmpeg.input(os.path.join(dst_dir,'audio%03d.wav'%(i+1))).audio
        ffmpeg.output(video_in, audio_in, os.path.join(dst_dir,'combine%03d.mp4'%(i+1))) \
              .global_args('-loglevel', 'quiet').overwrite_output().run()
    
    with open(os.path.join(dst_dir,'id_map.json'),'w') as f:
        json.dump(id_map,f)



##TODO:

def video2pic(path):
    # 读取视频文件，分帧输出，一段视频大概分20帧
    videoCapture = cv2.VideoCapture(path)
    #读帧
    success, frame = videoCapture.read()
    img_list = []
    i = 0
    timeF = 4
    while success :
        i = i + 1
        if (i % timeF == 0):
            img_list.append(frame)
        success, frame = videoCapture.read()
    return img_list

def feature1(path):
    pic = face_recognition.load_image_file(path)
    if len(face_recognition.face_encodings(pic))>=1:      
        encoding = face_recognition.face_encodings(pic)[0]
    else:
        encoding = []
    return encoding

### task1 函数

def predict1(testpath):

    facefeature = np.load("facefeature.npy")
    img_list = video2pic(testpath)
    vector = np.zeros(20)
    for img in img_list:
        img_numpy = np.array(img, 'uint8')
        # TODO:
        unknown_face_encoding = face_recognition.face_encodings(img_numpy)
        if len(unknown_face_encoding) >= 1:
            unknown_face = unknown_face_encoding[0]
            for i in range(20):
                results = np.linalg.norm([facefeature[i, :]] - unknown_face, axis=1)
                vector[i] += results
    ID = np.argmin(vector) + 1
    # print("测试结果:",ID_dict[ID])
    return ID

### task2 函数

def feature2(path):#之后都是新写的

    fpath = Path(path)
    wav = preprocess_wav(fpath)
    # wav = (wav[:,0]+wav[:,1])/2
    encoder = VoiceEncoder()
    embed = encoder.embed_utterance(wav)
    
    embed = embed.reshape(-1)
    return embed

def predict2(path):
    eigen=np.load('voicefeature.npy')
    test=feature2(path)
    result=1
    d=0
    for i in range(20):
        dot=np.dot(test,eigen[i,:])
        if dot>d:
            d=dot
            result=i+1
    return result

def predict2_3(path1,path2,path3,ID1,ID2,ID3):
    eigen=np.load('voicefeature.npy')
    test1=feature2(path1)
    test2=feature2(path2)
    test3=feature2(path3)
    test = [test1,test2,test3]
    flag = [1,1,1]
    # find ID1
    id1 = eigen[ID1 - 1,:]
    d=0
    idx1 = 0
    for i in range(3):
        dot=np.dot(test[i],id1)
        if dot>d:
            d=dot
            idx1=i
    flag[idx1] = 0 # delete ID1.wav

  # find ID2
    id2 = eigen[ID2 - 1,:]
    d=0
    idx2 = 0
    for i in range(3):
        if flag[i]==0:
            continue
        dot=np.dot(test[i],id2)
        if dot>d:
            d=dot
            idx2=i
    flag[idx2] = 0 # delete ID2.wav

  # find ID2
    id3 = eigen[ID3 - 1,:]
    idx3 = 0
    for i in range(3):
        if flag[i]==1:
            idx3 = i
    flag[idx3] = 0 # delete ID3.wav
    return idx1,idx2,idx3

### task3 函数

def get3face(testpath):
    facefeature = np.load("facefeature.npy")
    img_list = video2pic(testpath)
    vector1 = np.zeros(20)
    vector2 = np.zeros(20)
    vector3 = np.zeros(20)
    img = img_list[0]
    for img in img_list:
        img_numpy = np.array(img, 'uint8')
        shape = img_numpy.shape
        # img_numpy:H*W*3
        step = math.floor(shape[1]/3)
        img1 = img_numpy[:,0:step,:]
        img2 = img_numpy[:,step:2*step,:]
        img3 = img_numpy[:,2*step:-1,:]
        face = []
        face1 = face_recognition.face_encodings(img1)
        face2 = face_recognition.face_encodings(img2)
        face3 = face_recognition.face_encodings(img3)
        if len(face1) >= 1:
            unknown_face = face1[0]
            for j in range(20):
                results = np.linalg.norm([facefeature[j, :]] - unknown_face, axis=1)
                vector1[j] += results

        if len(face2) >= 1:
            unknown_face = face2[0]
            for j in range(20):
                results = np.linalg.norm([facefeature[j, :]] - unknown_face, axis=1)
                vector2[j] += results

        if len(face3) >= 1:
            unknown_face = face3[0]
            for j in range(20):
                results = np.linalg.norm([facefeature[j, :]] - unknown_face, axis=1)
                vector3[j] += results
    ID1 = np.argmin(vector1) + 1
    ID2 = np.argmin(vector2) + 1
    ID3 = np.argmin(vector3) + 1
    return ID1,ID2,ID3

def MP42WAV(mp4_path, wav_path):
    """
    这是MP4文件转化成WAV文件的函数
    :param mp4_path: MP4文件的地址
    :param wav_path: WAV文件的地址
    """
    video = VideoFileClip(mp4_path)
    audio = video.audio
    audio.write_audiofile(wav_path)

def generate(testpath):
    model = separator.from_hparams(source="speechbrain/sepformer-wsj03mix", savedir='pretrained_models/sepformer-wsj03mix')
    est_sources = model.separate_file(path=testpath) 
    torchaudio.save("1.wav", est_sources[:, :, 0].detach().cpu(), 8000)
    torchaudio.save("2.wav", est_sources[:, :, 1].detach().cpu(), 8000)
    torchaudio.save("3.wav", est_sources[:, :, 2].detach().cpu(), 8000)

def readwav(path):
    fpath = Path(path)
    wav = preprocess_wav(fpath)
    return wav

def predict2_new(path1,path2,path3,ID1,ID2,ID3):
    #ID=[ID1,ID2,ID3]
    eigen=np.load('voicefeature.npy')
    test1=feature2(path1)
    test2=feature2(path2)
    test3=feature2(path3)
    dict=[[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
    dot_result=np.zeros(6)
    e1 = eigen[ID1-1,:]
    e2 = eigen[ID2-1,:]
    e3 = eigen[ID3-1,:]
    dot_result[0]=np.dot(test1,e1)**2+np.dot(test2,e2)**2+np.dot(test3,e3)**2
    dot_result[1]=np.dot(test1,e1)**2+np.dot(test3,e2)**2+np.dot(test2,e3)**2
    dot_result[2]=np.dot(test2,e1)**2+np.dot(test1,e2)**2+np.dot(test3,e3)**2
    dot_result[3]=np.dot(test2,e1)**2+np.dot(test3,e2)**2+np.dot(test1,e3)**2
    dot_result[4]=np.dot(test3,e1)**2+np.dot(test1,e2)**2+np.dot(test2,e3)**2
    dot_result[5]=np.dot(test3,e1)**2+np.dot(test2,e2)**2+np.dot(test1,e3)**2
    index=np.argmax(dot_result)
    result=dict[index]
    
    return result[0],result[1],result[2]

def predict3_new(testpath):
    ID1,ID2,ID3 = get3face(testpath)
    # print("ID = ",ID1,ID2,ID3)
    ID=[ID1,ID2,ID3]
    mixedpath = "mixed.wav"
    audio_trace = read_audio(testpath,sr=44100)
    audio_trace = (audio_trace[:,0] + audio_trace[:,1])*0.5
    sf.write("mixed.wav",audio_trace,44100)
    generate(mixedpath)
    id1 ,id2, id3= predict2_new("1.wav",'2.wav','3.wav',ID1,ID2,ID3)
    
    # print("id=",id1,id2,id3)
    wav1,fs= sf.read("1.wav")
    wav2,fs= sf.read("2.wav")
    wav3,fs= sf.read("3.wav")
    wav=[wav1,wav2,wav3]
    
    return wav[id1],wav[id2],wav[id3]

if __name__=='__main__':
    generate_combine('./train','./train_gen')