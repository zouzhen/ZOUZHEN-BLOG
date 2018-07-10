---
title: Nodejs与Django的跨域问题
date: 2018-07-08 20:23:10
categories: Web开发
tags: [Nodejs, Django]
---
## Nodejs与Django的跨域问题

由于采用前后端分离的编程方式，Django的csrf_token验证失效，出现跨域问题，在此记录一下解决方法。

* 1 安装django-cors-headers  

    pip install django-cors-headers



* 2 配置settings.py文件  
![Nodejs与Django的跨域问题](Nodejs与Django的跨域问题/跨域问题模块.png)  

![Nodejs与Django的跨域问题](Nodejs与Django的跨域问题/跨域问题设置.png)  

OK！问题解决！

