from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
import os
from produce_data2 import data_producer
def feature(path):
    fpath = Path(path)
    wav = preprocess_wav(fpath)

    encoder = VoiceEncoder()
    embed = encoder.embed_utterance(wav)
    
    embed = embed.reshape(-1)
    return embed

if __name__ == '__main__':
    # 生成训练数据
    data_producer()
    eigen=np.zeros([20,256])
    for i in range(1,21):
        path1="wave/wave%s"%i
        paths = os.listdir(path1)
        wav_paths = []
        # 获取wav文件的相对地址
        for wav_path in paths:
            wav_paths.append(path1 + "/" + wav_path)
        N=len(wav_paths)
        e=np.zeros([N,256])
        for j in range(N):
            e[j,:]=feature(wav_paths[j])
            eigen[i-1,:]=eigen[i-1,:]+feature(wav_paths[j])
        eigen[i-1,:]=eigen[i-1,:]/N
    np.save('voicefeature.npy', eigen)
    print("training succeed!")