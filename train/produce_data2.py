from moviepy.editor import *
import os

def MP42WAV(mp4_path, wav_path):
    """
    这是MP4文件转化成WAV文件的函数
    :param mp4_path: MP4文件的地址
    :param wav_path: WAV文件的地址
    """
    video = VideoFileClip(mp4_path)
    audio = video.audio
    audio.write_audiofile(wav_path)
    
def data_producer():
    for i in range(1,21):
        # MP4文件和WAV文件的地址
        os.makedirs('wave/wave%s'%i)
        path1 = 'train/ID%s'%i
        path2 = "wave/wave%s"%i
        paths = os.listdir(path1)
        mp4_paths = []
        # 获取mp4文件的相对地址
        for mp4_path in paths:
            mp4_paths.append(path1 + "/" + mp4_path)
        # 得到MP4文件对应的WAV文件的相对地址
        wav_paths = []
        for mp4_path in mp4_paths:
            wav_path = path2 + "/" + mp4_path[1:].split('.')[0].split('/')[-1] + '.wav'
            wav_paths.append(wav_path)
        # 将MP4文件转化成WAV文件
        for (mp4_path, wav_path) in zip(mp4_paths, wav_paths):
            MP42WAV(mp4_path, wav_path)