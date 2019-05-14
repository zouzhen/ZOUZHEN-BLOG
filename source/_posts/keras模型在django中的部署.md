---
title: keras模型在django中的部署
date: 2019-05-14 16:31:40
categories:
tags:
---
### 项目环境
搭建服务器：0.0..  
项目路径：****

Python版本：python 3.6

Django版本：django 2.2

### 项目的最终开发

#### Tensorflow的设置
TensorFlow 的特点：

- 图 (graph) 来表示计算任务.
- 称之为?会话 (Session)?的上下文 (context) 中执行图.
- tensor 表示数据.
-?变量 (Variable)?维护状态.
- feed 和 fetch 可以为任意的操作(arbitrary operation) 赋值或者从其中获取数据.  

sess = tf.Session()  
创建一个新的TensorFlow session。
如果在构建session时没有指定graph参数，则将在session中启动默认关系图。如果使	用多个图（在同一个过程中使用tf.Graph()创建，则必须为每个图使用不同的sessio，但	是每个图都可以用于多个sessio中，在这种情况下，将图形显式地传递给sessio构造函	数通常更清晰。

sess.as_default()  
返回使该对象成为默认session的上下文管理器。

graph = tf.Graph()  
graph.as_default()  

** 在Tensorflow中，所有操作对象都包装到相应的Session中的，所以想要使用不同的模型就需要将这些模型加载到不同的Session中并在使用的时候申明是哪个Session，从而避免由于Session和想使用的模型不匹配导致的错误。而使用多个graph，就需要为每个graph使用不同的Session，但是每个graph也可以在多个Session中使用，这个时候就需要在每个Session使用的时候明确申明使用的graph。

    g1 = tf.Graph() # 加载到Session 1的graph
    g2 = tf.Graph() # 加载到Session 2的graph

    sess1 = tf.Session(graph=g1) # Session1
    sess2 = tf.Session(graph=g2) # Session2
    # 加载第一个模型with sess1.as_default(): 
        with g1.as_default():
            tf.global_variables_initializer().run()
            model_saver = tf.train.Saver(tf.global_variables())
            model_ckpt = tf.train.get_checkpoint_state(“model1/save/path”)
            model_saver.restore(sess, model_ckpt.model_checkpoint_path)# 加载第二个模型with sess2.as_default():  # 1
        with g2.as_default():  
            tf.global_variables_initializer().run()
            model_saver = tf.train.Saver(tf.global_variables())
            model_ckpt = tf.train.get_checkpoint_state(“model2/save/path”)
            model_saver.restore(sess, model_ckpt.model_checkpoint_path)

    ...
    # 使用的时候with sess1.as_default():
        with sess1.graph.as_default():  # 2
            ...
    with sess2.as_default():
        with sess2.graph.as_default():
            ...
    # 关闭sess
    sess1.close()
    sess2.close()

注：

- 1、在1处使用as_default使session在离开的时候并不关闭，在后面可以继续使用直到手动关闭；
- 2、由于有多个graph，所以sess.graph与tf.get_default_value的值是不相等的，因此在进入sess的时候必须sess.graph.as_default()明确申明sess.graph为当前默认graph，否则就会报错。  

设计上，为每个应用模型，分配一个session和graph，进行统一绑定管理。当使用特定模型的时候，采用如下模式：

    with self.session.as_default():
        with self.graph.as_default():
            Use model do sth;

#### 模型管理类

    import keras
    from keras.models import load_model
    from keras.preprocessing.image import array_to_img, img_to_array, load_img
    from scipy.misc import imresize
    import numpy as np
    import cv2
    from keras.models import model_from_json
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from keras import backend as K
    import os
    import logging
    logger = logging.getLogger('django')

    class ModelUtil(object):
        def __init__(self,modelPath,data):
            self.graph = tf.Graph()
            self.session = tf.Session(graph=self.graph)
            self.modelList = []
            self.data = data
            self.modelPath = modelPath

        def __str__(self):
            return '(%s,%s,%s)' %(self.session, self.graph, self.modelList);

        def upload_h5_Models(self):
            logger.debug("upload models begin")
            logger.info(self.data.shape)
            path_name=self.modelPath
            self.modelList = []
            model = []
            with self.session.as_default():
                with self.graph.as_default():
                    for dir_item in os.listdir(path_name):
                        #从初始路径开始叠加，合并成可识别的操作路径
                        full_path = os.path.abspath(os.path.join(path_name, dir_item))
                        logger.debug(full_path)
                        if not full_path.endswith('.h5'):
                            continue
                        model = [dir_item,load_model(full_path)]
                        logger.debug("each load finish")
                        logger.debug("%s" %model[1].predict(self.data))
                        self.modelList.append(model)
                    logger.info(self.modelList)
                logger.debug("upload models end")

        def predict_proba(self,data):
            result = []
            with self.session.as_default():
                with self.graph.as_default():
                    for m in self.modelList:
                        myObject= {}
                        logger.debug("predict use model %s" %(m[0]))
                        logger.debug(m[1])
                        #with graph.as_default():
                        myObject['name']=m[0]
                        proba = m[1].predict(data)
                        myObject['value']=[proba[0][0],proba[0][1]]
                        logger.debug("predict value is %s" %(myObject['value']))
                        result.append(myObject)
            return result

        def predict(self,data):
            result = []
            with self.session.as_default():
                with self.graph.as_default():
                    for m in self.modelList:
                        myObject= {}
                        logger.debug("predict use model %s" %(m[0]))
                        logger.debug(m[1])
                        #with graph.as_default():
                        myObject['name']=m[0]
                        myObject['value']=round(m[1].predict(data)[0][0], 2)
                        logger.debug("predict value is %s" %(myObject['value']))
                        result.append(myObject)
            return result


#### 应用api开发

    from django.shortcuts import render
    from django.contrib.auth.models import User, Group
    from rest_framework import viewsets
    from yyai import serializers #import UserSerializer, GroupSerializer
    from django.http import HttpResponse, JsonResponse
    from django.views.decorators.csrf import csrf_exempt
    from rest_framework.decorators import action

    from keras.preprocessing.image import array_to_img, img_to_array, load_img
    from scipy.misc import imresize

    import numpy as np
    import cv2
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from yyai.modelUtils import ModelUtil
    #from yyai import modelUtils
    import logging
    logger = logging.getLogger('django')

    ################################################################
    #模型路径需要修改
    modelPath='/opt/web/aiServer/yyai/model/facescore/'
    #模型输入数据实例
    testData=np.zeros((1,350,350,3),dtype=np.float32)
    #实例化模型管理类
    modelU = ModelUtil(modelPath,testData)
    logger.info(modelU)
    modelU.upload_h5_Models()
    logger.info(modelU)

    #webapi接口，进行该应用的模型更新
    def loadModel(request):
        logger.info(modelU)
        modelU.upload_h5_Models()
        logger.info(modelU)
        res = {"result":"load model success!"}
        return JsonResponse(res, safe=False)

    def get_score(img):
        logger.info("enter in to get_score...")
        logger.info(modelU)
        img_height, img_width, channels = 350, 350, 3
        if modelU.modelList==[]:
            logger.info('no faceScore models ... return []')
            return {"result":"no faceScore models"}
        result = []
        try:
            resized_image = cv2.resize(img, (img_height, img_width))
        except :
            logger.info("resize is except: imresize" )
            return {"result":"resize is except"}
        test_x = resized_image
        test_x = test_x.astype("float32") / 255.
        test_x = test_x.reshape((1,) + test_x.shape)
        logger.info(test_x.shape)

        result = modelU.predict(test_x)

        return {"result":result}

    # Create your views here.
    # 对外的模型API接口
    @csrf_exempt
    def predict_face_score(request):
        res = {"name":"lida","path":"/home/lida"}
        #使用get或post模式都可以，yyImage使用POST模式
        if request.method == 'GET':
            logger.info("get data and return defalt")
            #name=request.GET.get("name")
            imgPath="uploads/admin/0_cropped_bdb19711-11b7-484e-b309-6f7c196687de.jpg"
            img = img_to_array(load_img(imgPath))
            print(img)
            res = "%s" %get_score(img)
        elif request.method == 'POST':
            #name=request.POST.get("name",'')
            imgPath = request.POST.get("imgPath",'')
            logger.info(imgPath)
            if imgPath=='':
                logger.info("no imgPath")
                res = [{'result':'no images'}]
            else:
                img = img_to_array(load_img(imgPath))
                res = "%s" %get_score(img)
        logger.info(res)
        return JsonResponse(res, safe=False)
