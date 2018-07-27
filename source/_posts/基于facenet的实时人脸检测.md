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

