## Readme

#### 0.配置相应环境

如果使用conda虚拟环境，则运行

`conda env create -f environment.yaml`

即可创建相同的虚拟环境，但是由于此虚拟环境还包含其他大量不需要的库和依赖，因此存在较多冗余，不推荐

如果使用pip直接安装所需要的包和依赖，则运行

`pip install -r requirements.txt`



#### 1.训练

**task1:**

`./train/task1_train.py`文件同时进行了数据的处理、准备和训练

```bash
cd train
ls
python task1_train.py
```

生成结果保存为`facefeature.npy`二进制文件

**task2:**

`./train/task2_train.py`文件同时进行了数据的处理、准备和训练

```bash
cd train
ls
python task2_train.py
```

生成结果保存为`voicefeature.npy`二进制文件

**task3:**

分离部分采用`speechbrain`的预训练模型，无需再重新进行训练；对于语音位置的标定，直接调用task1、task2的模型，无需重复训练



#### 2.测试

**task1:**

`test1.py`单独用于task1的单独测试，运行：

`python test1.py`

**task2:**

`test2.py`单独用于task2的单独测试，运行：

`python test2.py`

**task3:**

`test3.py`单独用于task1的单独测试，运行：

`python test3.py`

**joint-task:**

`test.py`单独用于task的联合测试，运行：

`python test.py`