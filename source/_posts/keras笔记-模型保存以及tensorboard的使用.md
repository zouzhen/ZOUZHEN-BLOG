---
title: keras笔记-模型保存以及tensorboard的使用
date: 2019-04-17 17:26:22
categories: 框架
tags: [tensoflow,keras]
---

对于一个epoch的模型保存以及tensorboard的使用都是可以使用keras中的回调函数 Callbacks。

首先还是看代码吧:

    from keras import backend as K
    from keras.models import Sequential
    from keras.layers.core import Activation, Dropout
    from keras.layers.core import Dense
    from keras.optimizers import SGD
    from keras.callbacks import ModelCheckpoint, TensorBoard
    
    from keras.datasets import mnist
    from keras.utils import np_utils
    
    
    import numpy as np
    import os
    
    
    np.random.seed(1671)
    
    
    #网络和训练
    NB_EPOCH = 20
    BATCH_SIZE = 128
    VERBOSE = 1
    NB_CLASSES = 10
    OPTIMIZER = SGD()
    N_HIDDEN = 128
    VALIDATION_SPLIT = 0.2
    DROPOUT = 0.3
    MODEL_path = './model/'
    if not os.path.exists(MODEL_path):
        os.mkdir(MODEL_path)
    
    
    
    #数据
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    
    RESHAPED = 784
    
    #
    X_train = X_train.reshape(60000, RESHAPED)
    X_test = X_test.reshape(10000, RESHAPED)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    
    #归一化
    X_train /= 255
    X_test /= 255
    
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    
    Y_train = np_utils.to_categorical(Y_train, NB_CLASSES)
    Y_test = np_utils.to_categorical(Y_test, NB_CLASSES)
    
    
    model = Sequential()
    model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(N_HIDDEN))
    model.add(Activation('relu'))
    model.add(Dense(NB_CLASSES))
    model.add(Activation('softmax'))
    model.summary()
    
    
    model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
    
    checkpoit = ModelCheckpoint(filepath=os.path.join(MODEL_path, 'model-{epoch:02d}.h5'))
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    
    history = model.fit(X_train, Y_train,
                        batch_size=BATCH_SIZE, epochs=NB_EPOCH,
                        verbose=VERBOSE, validation_split=VALIDATION_SPLIT, callbacks=[checkpoit, tensorboard])
    
    #模型保存
    model.save('mnist.h5')
    
    score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
    print('test score:', score[0])
    print('test accuracy:', score[1])
 
### 回调函数使用
回调函数是一个函数的合集，会在训练的阶段中所使用。你可以使用回调函数来查看训练模型的内在状态和统计。你可以传递一个列表的回调函数（作为 callbacks 关键字参数）到 Sequential 或 Model 类型的 .fit() 方法。在训练时，相应的回调函数的方法就会被在各自的阶段被调用。

### ModelCheckpoint
    keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
在每个训练期之后保存模型。

filepath 可以包括命名格式选项，可以由 epoch 的值和 logs 的键（由 on_epoch_end 参数传递）来填充。

例如：如果 filepath 是 weights.{epoch:02d}-{val_loss:.2f}.hdf5， 那么模型被保存的的文件名就会有训练轮数和验证损失。

#### 参数

    filepath: 字符串，保存模型的路径。
    monitor: 被监测的数据。
    verbose: 详细信息模式，0 或者 1 。
    save_best_only: 如果 save_best_only=True， 被监测数据的最佳模型就不会被覆盖。
    mode: {auto, min, max} 的其中之一。 如果 save_best_only=True，那么是否覆盖保存文件的决定就取决于被监测数据的最大或者最小值。 对于 val_acc，模式就会是 max，而对于 val_loss，模式就需要是 min，等等。 在 auto 模式中，方向会自动从被监测的数据的名字中判断出来。
    save_weights_only: 如果 True，那么只有模型的权重会被保存 (model.save_weights(filepath))， 否则的话，整个模型会被保存 (model.save(filepath))。
    period: 每个检查点之间的间隔（训练轮数）。
    TensorBoard
    keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    Tensorboard 基本可视化。

TensorBoard 是由 Tensorflow 提供的一个可视化工具。

这个回调函数为 Tensorboard 编写一个日志， 这样你可以可视化测试和训练的标准评估的动态图像， 也可以可视化模型中不同层的激活值直方图。

如果你已经使用 pip 安装了 Tensorflow，你应该可以从命令行启动 Tensorflow：

    tensorboard --logdir=/full_path_to_your_logs

### 参数

    log_dir: 用来保存被 TensorBoard 分析的日志文件的文件名。
    histogram_freq: 对于模型中各个层计算激活值和模型权重直方图的频率（训练轮数中）。 如果设置成 0 ，直方图不会被计算。对于直方图可视化的验证数据（或分离数据）一定要明确的指出。
    write_graph: 是否在 TensorBoard 中可视化图像。 如果 write_graph 被设置为 True，日志文件会变得非常大。
    write_grads: 是否在 TensorBoard 中可视化梯度值直方图。 histogram_freq 必须要大于 0 。
    batch_size: 用以直方图计算的传入神经元网络输入批的大小。
    write_images: 是否在 TensorBoard 中将模型权重以图片可视化。
    embeddings_freq: 被选中的嵌入层会被保存的频率（在训练轮中）。
    embeddings_layer_names: 一个列表，会被监测层的名字。 如果是 None 或空列表，那么所有的嵌入层都会被监测。
    embeddings_metadata: 一个字典，对应层的名字到保存有这个嵌入层元数据文件的名字。 查看 详情 关于元数据的数据格式。 以防同样的元数据被用于所用的嵌入层，字符串可以被传入。
    
作者：pursuit_zhangyu 
来源：CSDN 
原文：https://blog.csdn.net/pursuit_zhangyu/article/details/85226481 
版权声明：本文为博主原创文章，转载请附上博文链接！