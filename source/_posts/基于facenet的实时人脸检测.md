---
title: 基于facenet的实时人脸检测
date: 2018-07-27 17:33:00
categories: 项目
tags: 人脸识别
---
### 参考自https://github.com/shanren7/real_time_face_recognition

### 本人的项目代码https://github.com/zouzhen/real_time_face_recognize

~~虽然名字相同，但里面的内容可是有很大的不同~~  
由于不能满足当前的tensorflow版本，以及未能满足设计要求，进行了优化与重新设计
##  基于facenet的实时人脸检测

### 工作环境

* python 3.6
* tensorflow==1.9.0(可运行在无gpu版)

### 代码结构

real_time_face_recognize  
* |—— model_check_point（保存人脸识别模型）
* |—— models（储存了facenet采用的神经网络模型）
* |—— detect_face.py(主要实现人脸的检测，同时返回可能的人脸框)
* |—— facenet.py（这里存储了facenet的主要函数）
* |—— real_time_face_recognize.py(实现了实时人脸检测)  

### 运行

1. 从 https://github.com/davidsandberg/facenet 中下载预训练的分类模型，放在model_check_point下  
2. 使用pip install requirements.txt安装需要的包，建议在virtualenv环境安装  
3. 在目录下新建picture文件，将需要识别的人的图片放入其中，每人放入一张清晰的图片即可  
4. 执行python real_time_face_recognize.py 

### 注意  

除可在facenet作者的github中下载模型外，我自己基于lfw训练集训练了一个模型，点击