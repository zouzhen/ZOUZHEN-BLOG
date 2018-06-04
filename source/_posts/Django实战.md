---
title: Django的网站逻辑
date: 2018-05-29 15:38:48
categories: Web开发
tags: Django
---
# Django实战——后台逻辑

## Django用户 （登录 注册 找回密码）

### 登录
***
后台的操作放在对应app的views文件下，当我们在路由中添加一个url，django会自动为我们生成一个request，  
并添加到
函数里面。首先判断请求方法，是POST还是GET。然后跳转到对应的页面进行操作。  

#### 对登录账户进行验证（采用类来做）
1. 得到用户名和密码后，使用django.contrib.auth.authenticate进行验证，验证成功的话得到一个对象，  
然后进行对应后台逻辑的编写，即调用django.contrib.auth.login进行验证。  
2. 对登录成功后返回index.html文件的状态处理，需要在html文件中进行判断，用户是否登录，调用  
request.user.user.is_authenticated来进行判断。决定显示哪一行代码。
3. 自定义认证方法，实现邮箱的登录方式（重定义方式）


#### Seesion和Cookie机制
1. 无状态请求
2. 有状态请求 
***
### 注册
---
#### 准备工具
1. 添加插件（captcha）

#### 提交注册信息（包括注册码）

#### 发送邮件验证注册信息

#### 激活账户
---
### 找回密码
***

采用类似于激活账户的方式来实现


