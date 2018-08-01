---
title: PyTorch入门
date: 2018-08-01 14:59:07
categories: 框架
tags: pytorch
---
### 基本概念

---
*  <font color="#00dddd" size="4">张量</font><br />  
---
张量类似于NumPy的ndarray，另外还有Tensors也可用于GPU以加速计算。  

    from __future__ import print_function
    import torch  

构造一个未初始化的5x3矩阵：

    x = torch.empty(5, 3)
    print(x)    
    tensor([[ 3.2401e+18,  0.0000e+00,  1.3474e-08],
        [ 4.5586e-41,  1.3476e-08,  4.5586e-41],
        [ 1.3476e-08,  4.5586e-41,  1.3474e-08],
        [ 4.5586e-41,  1.3475e-08,  4.5586e-41],
        [ 1.3476e-08,  4.5586e-41,  1.3476e-08]])  

构造一个矩阵填充的零和dtype long：

    x = torch.zeros(5, 3, dtype=torch.long)
    print(x)  
    tensor([[ 0,  0,  0],
            [ 0,  0,  0],
            [ 0,  0,  0],
            [ 0,  0,  0],
            [ 0,  0,  0]])  

直接从数据构造张量：

    x = torch.tensor([5.5, 3])
    print(x)
    tensor([ 5.5000,  3.0000])

或者根据现有的张量创建张量。除非用户提供新值，否则这些方法将重用输入张量的属性，例如dtype

    x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
    print(x)
    x = torch.randn_like(x, dtype=torch.float)    # override dtype!
    print(x)                                      # result has the same size

    tensor([[ 1.,  1.,  1.],
            [ 1.,  1.,  1.],
            [ 1.,  1.,  1.],
            [ 1.,  1.,  1.],
            [ 1.,  1.,  1.]], dtype=torch.float64)
    tensor([[ 0.2641,  0.0149,  0.7355],
            [ 0.6106, -1.2480,  1.0592],
            [ 2.6305,  0.5582,  0.3042],
            [-1.4410,  2.4951, -0.0818],
            [ 0.8605,  0.0001, -0.7220]])
得到它的大小：  

    print(x.size())
    torch.Size([5, 3])  

**注意**  

**torch.Size** 实际上是一个元组，因此它支持所有元组操作。  

---
* <font color="#00dddd" size="4">操作</font><br />   
---
操作有多种语法。在下面的示例中，我们将查看添加操作。

增加：语法1  

    y = torch.rand(5, 3)
    print(x + y)

    tensor([[ 0.7355,  0.2798,  0.9392],
            [ 1.0300, -0.6085,  1.7991],
            [ 2.8120,  1.2438,  1.2999],
            [-1.0534,  2.8053,  0.0163],
            [ 1.4088,  0.9000, -0.1172]])

增加：语法2

    print(torch.add(x, y))

    tensor([[ 0.7355,  0.2798,  0.9392],
            [ 1.0300, -0.6085,  1.7991],
            [ 2.8120,  1.2438,  1.2999],
            [-1.0534,  2.8053,  0.0163],
            [ 1.4088,  0.9000, -0.1172]])

增加：提供输出张量作为参数

    result = torch.empty(5, 3)
    torch.add(x, y, out=result)
    print(result)

    tensor([[ 0.7355,  0.2798,  0.9392],
            [ 1.0300, -0.6085,  1.7991],
            [ 2.8120,  1.2438,  1.2999],
            [-1.0534,  2.8053,  0.0163],
            [ 1.4088,  0.9000, -0.1172]])
增加：就地

    # adds x to y
    y.add_(x)
    print(y)

    tensor([[ 0.7355,  0.2798,  0.9392],
            [ 1.0300, -0.6085,  1.7991],
            [ 2.8120,  1.2438,  1.2999],
            [-1.0534,  2.8053,  0.0163],
            [ 1.4088,  0.9000, -0.1172]])
**注意**

任何使原位张量变形的操作都是用_。后固定的。例如：x.copy_(y)，x.t_()，将改变x。

你可以使用标准的NumPy索引与所有的铃声和​​口哨！

    print(x[:, 1])

    tensor([ 0.0149, -1.2480,  0.5582,  2.4951,  0.0001])

调整大小：如果要调整大小/重塑张量，可以使用torch.view：

    x = torch.randn(4, 4)
    y = x.view(16)
    z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
    print(x.size(), y.size(), z.size())

    torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])

如果你有一个元素张量，用于.item()获取值作为Python数字

    x = torch.randn(1)
    print(x)
    print(x.item())

    tensor([ 1.3159])
    1.3159412145614624

### NumPy Bridge

将Torch Tensor转换为NumPy阵列（反之亦然）是一件轻而易举的事。  
Torch Tensor和NumPy阵列将共享其底层内存位置，更改一个将改变另一个。

---
* <font color="#00dddd" size="4">将Torch Tensor转换为NumPy数组</font><br />    
---
    a = torch.ones(5)
    print(a)

    tensor([ 1.,  1.,  1.,  1.,  1.])
    b = a.numpy()
    print(b)

    [1. 1. 1. 1. 1.]

了解numpy数组的值如何变化。

    a.add_(1)
    print(a)
    print(b)

    tensor([ 2.,  2.,  2.,  2.,  2.])
    [2. 2. 2. 2. 2.]

---
* <font color="#00dddd" size="4">将NumPy数组转换为Torch Tensor</font><br />
---
了解更改np阵列如何自动更改Torch Tensor

    import numpy as np
    a = np.ones(5)
    b = torch.from_numpy(a)
    np.add(a, 1, out=a)
    print(a)
    print(b)

    [2. 2. 2. 2. 2.]
    tensor([ 2.,  2.,  2.,  2.,  2.], dtype=torch.float64)
除了CharTensor之外，CPU上的所有Tensors都支持转换为NumPy并返回。

---
* <font color="#00dddd" size="4">CUDA Tensors</font><br />
---
可以使用该.to方法将张量移动到任何设备上。

    # let us run this cell only if CUDA is available
    # We will use ``torch.device`` objects to move tensors in and out of GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")          # a CUDA device object
        y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
        x = x.to(device)                       # or just use strings ``.to("cuda")``
        z = x + y
        print(z)
        print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!

    tensor([ 2.3159], device='cuda:0')
    tensor([ 2.3159], dtype=torch.float64)