---
title: 搭建python沙盒环境
date: 2019-04-01 10:23:52
categories: 环境配置
tags: [沙盒,python]
---

### 一. 先update    sudo apt update

### 二. 安装pip       sudo apt install python-pip

### 三.  安装virtualenv    sudo pip install virtualenv  virtualenvwrapper

### 四. 创建目录用来存放虚拟环境    sudo mkdir -p $WORKON_HOME

### 五. 在用户目录下中的 .bashrc 中添加以下内容并保存(通过ll 可查看到 .bashrc 文件)

if [ -f /usr/local/bin/virtualenvwrapper.sh ]; then
    export WORKON_HOME=$HOME/.virtualenvs
    source /usr/local/bin/virtualenvwrapper.sh
fi  

### 六. 运行  source .bashrc   重新加载到环境变量中

### 七. 创建虚拟环境     mkvirtualenv  test1             关于虚拟环境的命令如下

mkvirtualenv wxhpython01：创建运行环境wxhpython01  
workon wxhpython01: 工作在 zqxt 环境 或 从其它环境切换到 wxhpython01环境  
deactivate: 退出终端环境  
rmvirtualenv ENV：删除运行环境ENV  
mkproject mic：创建mic项目和运行环境mic  
mktmpenv：创建临时运行环境  
lsvirtualenv: 列出可用的运行环境  
lssitepackages: 列出当前环境安装了的包  

八.在虚拟环境中可以通过pip安装其他的所需包  例如 pip install django==1.9.8
