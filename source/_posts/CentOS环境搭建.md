---
title: CentOS环境搭建
date: 2018-06-05 19:13:28
categories:
tags:
---

# 系统安装
### 踩坑记录
将CentOS 7镜像刻到U盘之后，向服务器安装时，使用U盘启动会出现两种启动选项，一种是UEFI启动选项，一种是默认的启动选项，如果不使用UEFI方式安装，那么一般是没有问题的，如果选择UEFI方式安装系统，那么引导系统时会出现如下的提示：

    [sdb] No Caching mode page found

    [sdb] Assuming drive cache:write through

    Could not boot  /dev/root does not exist

　　然后命令行就卡在这了，现在只需要耐心等待，等一会之后会不断的滚动错误警告，这个时候继续等待，那么一会就会出来命令行输入界面，这个时候输入以下命令：

    ls /dev/sd*

　　输入命令之后会列出所有的存储设备，这个时候一般情况下第一块硬盘是sda，如果有多个分区，那么依次就是sda1、sda2等等，如果有两块硬盘那么就是sdb，U盘一般是排最后的号，如果有一块硬盘，那么U盘就是sdb，如果有两块硬盘，那么U盘就是sdc，U盘一般会有sdc和sdc4两个选项，sdc属于U盘存储，sdc4就是镜像所在分区了，这样一般是没有问题的，如果出现问题，那么接下来多配置几次就好了，接下来输入命令reboot重启计算机，在安装界面，先不要选择安装，这个时候按一下e键，会进入编辑界面，移动光标进行如下修改：

　　在第二行默认是：vmlinuz initrd=initrd.img inst.stage2=hd:LABEL=CentOS\x207\x20x86_64 rd.live.check quiet

　　把这行修改为：vmlinuz initrd=initrd.img inst.stage2=hd:/dev/sdc4:/ quiet

　　就是把hd:和quiet之间的内容修改为U盘镜像所在位置这样就可以了，注意要写成/dev/sdc4:/

　　然后根据提示按Ctrl+X键就可以开始安装了，现在就正常进入安装界面了

# NVIDIA驱动安装
关于NVIDIA驱动安装，需要选择最新的安装版本。




# CUDA&&CUDNN
　　关于cuda和cudnn的安装，这一点尤其要注意。由于考虑到tensorflow的版本编译问题