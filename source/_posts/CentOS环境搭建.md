---
title: CentOS环境搭建
date: 2018-06-05 19:13:28
categories:
tags:
---

# 系统安装
## 电脑配置
1. 微星1080Ti
2. 至强E5 -2620v4
3. 技嘉的主板
### 踩坑记录
将CentOS 7.4镜像刻到U盘之后，向服务器安装时，使用U盘启动会出现两种启动选项，一种是UEFI启动选项，一种是默认的启动选项，如果不使用UEFI方式安装，那么一般是没有问题的，如果选择UEFI方式安装系统，那么引导系统时会出现如下的提示：

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
1、在官网上http://www.geforce.cn/drivers搜索到对应型号的显卡驱动并下载，下载到的驱动文件是一个后缀名为.run的文件（例如NVIDIA-Linux-x86_64-384.98.run）；

2、安装gcc编译环境以及内核相关的包：  
    yum install kernel-devel kernel-doc kernel-headers gcc\* glibc\*  glibc-\*  
注意：安装内核包时需要先检查一下当前内核版本是否与所要安装的kernel-devel/kernel-doc/kernel-headers的版本一致，请务必保持两者版本一致，否则后续的编译过程会出问题。

3、禁用系统默认安装的 nouveau 驱动，修改/etc/modprobe.d/blacklist.conf 文件：  
### 修改配置
    echo -e "blacklist nouveau\noptions nouveau modeset=0" > /etc/modprobe.d/blacklist.conf

### 备份原来的镜像文件
    mv /boot/initramfs-$(uname -r).img /boot/initramfs-$(uname -r).img.bak

### 重建新镜像文件
    dracut /boot/initramfs-$(uname -r).img $(uname -r)

### 重启
    reboot  

### 在命令行界面
    init 5 
    
### 查看nouveau是否启动，如果结果为空即为禁用成功
    lsmod | grep nouveau

4、安装DKMS模块

DKMS全称是DynamicKernel ModuleSupport，它可以帮我们维护内核外的驱动程序，在内核版本变动之后可以自动重新生成新的模块。  

    sudo yum install DKMS  

5、执行显卡驱动安装脚本（如果内核版本一致，就不需要指定--kernel-source-path和-k）  

    ./NVIDIA-Linux-x86_64-384.98.run --kernel-source-path=/usr/src/kernels/3.10.0-693.11.1.el7.x86_64/ -k $(uname -r) --dkms -s

6、若步骤5执行过程中没报错，则安装成功。重启，执行nvidia-smi可查看相关信息。  
如若出现重启系统驱动找不到的情况，在装完驱动后，切记，先不要重启，使用

    init5
和  

    init 3
交替切换，几次后，会进入图形界面（其中init 5为进入图形界面的命令），之后，在图形界面，重新编译一下启动项。


# CUDA&&CUDNN
　　关于cuda和cudnn的安装，这一点尤其要注意。我在CentOS7.5(更新后的版本，初始装的时候为7.4)上安装CUDA9.2时，无法与NVIDIA的驱动匹配，因此退而求其次，选择了CUDA9.1。