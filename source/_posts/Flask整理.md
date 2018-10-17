---
title: Flask整理
date: 2018-10-17 15:50:11
categories: Python后端
tags: flask框架
---
## **使用工具**

#### Flask后端 + Postgresql数据库 + JS前端（我未使用）


## Flask搭建

- **确定确定目录结构**
    
    1. app/algorithms: 用来存放相关的算法文件
    2. app/models: 用来存放数据库的操作
    3. app/web: 用来存放路由和视图函数
    4. manage: flask的启动文件

- **确定路由注册方式**

    1. 使用蓝图形式来注册路由

- **确定数据库操作方式**

    1. 使用sqlalchemy及psycopg2来控制Postgresql数据库
    2. 由于主要是用来进行数据读取的，所以采用非ORM方式构建的表结构，这种方式方便进行查询过滤操作

## 基于sqlalchemy的Postgresql数据库访问操作

- **创建表结构**

```py
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import MetaData, Table, Column, ForeignKey, Sequence
from sqlalchemy.types import *
from sqlalchemy.sql.expression import select,and_
from datetime import datetime
engine = create_engine('postgres://user:password@hosts/builder', echo=True)

metadata = MetaData()
metadata.bind = engine
    # 创建桥梁索引表
bridges_table = Table('bridges', metadata,
                      Column('id', Integer, primary_key=True),
                      Column('org_id', Integer, nullable=False),
                      Column('user_id', Integer, nullable=False),
                      Column('name', VARCHAR(length=255), nullable=False),
                      Column('created_date', TIMESTAMP, nullable=False),
                      Column('finished_date', TIMESTAMP, nullable=True),
                    #   autoload=True,
                      )
```
这种方式，有助于进行表查询，具体的相关API介绍及使用放那格式可[点此](https://docs.sqlalchemy.org/en/latest/genindex.html)查看。


- **相关操作**

```py
# 添加数据
def add():
    s = book_table.insert().values(title='测试写入2',time=datetime.now())
    c = engine.execute(s)
    c.close()
    return c.inserted_primary_key

# 查询数据
def query_code(id):
    info = {'id': '', 'title': ''}
    s = select([bridge_jobs_table.c.id.label('name')]).where(and_(bridge_jobs_table.c.kind=='桩基',bridge_jobs_table.c.name=='起钻'))
    codename_query = engine.execute(s)
    print(codename_query.keys())
    for row in codename_query:
        print(row[0])
    codename_query.close()
    return info

# 更新数据
def updata(id, title):
    s = book_table.update().where(book_table.c.id == id).values(title=title, id=id)
    c = engine.execute(s)
    c.close()
```

## Flask相关知识

- **路由操作**
    
    1. 静态路由
    ```py
    @web.route('/hello', methods=['POST', 'GET'])
    def hello():
        return 'hello world!'
    ```

    2. 参数路由
    ```py
    @web.route('/hello/<string:name>', methods=['POST', 'GET'])
    def hello(name):
        return 'hello %d '% name
    ```

    3. JSON返回
    ```py
    @web.route('/hello/<string:name>', methods=['POST', 'GET'])
    def hello(name):
        return jsonify('hello %d '% name)
    ```

    4. 使用蓝图方式注册路由
    ```py
    from flask import Flask


    def create_app():
        app = Flask(__name__)
        app.config.from_object('app.setting')

        register_blueprint(app)
        # db.init_app(app)
        # db.create_app(app=app)
        return app


    def register_blueprint(app):
        from app.web.view import web
        app.register_blueprint(web)

    ```