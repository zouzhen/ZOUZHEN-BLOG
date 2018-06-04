---
title: Adm-zip工具
date: 2018-05-29 09:12:01
categories: 工具
tags: [工具,Adm-zip]
---

### Adm-zip介绍
***
**Adm-zip是JavaScript中的一款与压缩文件相关的插件，其功能相当的强大（我看来），我用其实现了对Word文档的内容替换。**

<font color=#00099ff face='黑体'>Word 文档本质上是一个压缩 文件夹，其中的word文件夹下的document.xml文件是包含文档内容的文件，而我们需要操作的也正是这个文件。</font>

<font color=#00099ff face='微软雅黑'>Adm-zip这款插件则正好满足我们即对压缩文件内部条目文件的处理，同时又保证不影响压缩文件内部其余文件的要求。</font>  

<font  face='黑体'>我们需要的函数接口主要有四个，分别为：</font>  

读取压缩文件内指定目录里面的文件或者文件夹：  

    Admzip.readAsText()  

删除压缩文件内的指定文件或者文件夹：

    Admzip.deleteFile()  

将指定文件写入到压缩文件夹中：

    Admzip.addFile()  

将所做的更改重新写入文件（可以是当前文件，也可以重命名的word文档）

    Admzip.writeZip()

**关于Adm-zip的使用方法，暂时只发现了这样一种，其还有别的Api接口，有兴趣的小伙伴可以自己再研究下^_^**
