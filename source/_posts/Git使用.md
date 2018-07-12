---
title: Git使用
date: 2018-07-12 11:27:13
categories: 工具
tags: Git
---

转载自https://blog.csdn.net/zwhfyy/article/details/8625228，如有侵权，请联系删除

# 出错信息
Your local changes to the following files would be overwritten by merge
error: Your local changes to the following files would be overwritten by merge:
        123.txt
Please, commit your changes or stash them before you can merge.

如果希望保留生产服务器上所做的改动,仅仅并入新配置项, 处理方法如下:  

    git stash
    git pull
    git stash pop  

然后可以使用git diff -w +文件名 来确认代码自动合并的情况.

反过来,如果希望用代码库中的文件完全覆盖本地工作版本. 方法如下:  

    git reset --hard
    git pull
    其中git reset是针对版本,如果想针对文件回退本地修改,使用