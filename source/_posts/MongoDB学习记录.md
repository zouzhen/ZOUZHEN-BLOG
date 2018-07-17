---
title: MongoDB学习记录
date: 2018-07-16 11:51:17
categories: 数据库
tags: mongodb
---
MongoDB
=== 
转载自https://github.com/zxhyJack/MyBlog/blob/master/mongodb/mongodb.md

###  基本概念

- 文档  
    是键值对的有序集合，这是MongDB的核心概念  

- 集合  
    集合就是一组文档  
    - 动态模式  
        集合是动态模式的，这意味着集合里面的文档可以是是各式各样的
    - 命名  
        集合使用名称进行命名  

- 数据库  
    由多个集合构成数据库，一个MongDB实例可以承载多个数据库，每个数据库拥有0个或多个集合  

- 基本操作  
    在终端运行mongod命令，启动时，shell将自动连接MongDB数据库，需确保数据库已启动，可充分利用Javascript的标准库，还可定义和调用Javascript函数。
    
    - 创建数据库

            use database_name

    如果数据库存在，则进入指定数据库，否则，创建数据库
    此时需要写入数据，数据库才能真正创建成功

    - 查看所有数据库

            show databases | dbs

    - 创建集合

            db.createCollection(collection_name)

    - 删除数据库
    先进入要删除的数据库，然后执行命令

            db.dropDatabase()

    - 删除集合

            db.collection_name.drop()

    - 增

            db.collection_name.insert(document)

        exp:
            
            db.students.insert({
            name:'James',
            age: 32,
            gender:'man',
            career:'player'
            })

    - 查

            db.collection.find(<query>,<projection>)
            - query: 查询条件
            - projection: 投影操作

        exp:

            db.students.find()

    - 改

            db.collection.updateOne(<query>,<update>) // 更新第一个符合条件的集合
            db.collection.updateMany(<query>,<update>)  // 更新所有符合条件的集合

        - query: 查询条件
        - update： 更新的内容

        exp: 

            db.students.update({name:'James'},{$set:{gender:'woman'}})

    - 删

            db.collection_name.deleteOne(<query>) // 删除第一个符合条件的集合
            db.collection_name.deleteMany(<query>) // 删除所有符合条件的集合

        exp:

            db.students.deleteOne({name:'James'})

## 数据操作（重点）

数据库的核心——CRUD，增加和删除较为简单，查询和修改较复杂

### 查询

#### 关系运算符

- $gt 大于

- $lt 小于

- $gte  大于等于

- $lte  小于等于

- $eq | (key: value)  等于

- $ne 不等于

先往数据库中添加一些数据

    db.students.insert({'name':'张三','sex':'男','age':19,'score': 89,'address': '海淀区'})
    db.students.insert({'name':'李四','sex':'女','age':20,'score': 100,'address': '朝阳区'})
    db.students.insert({'name':'王五','sex':'男','age':22,'score': 50,'address': '西城区'})
    db.students.insert({'name':'赵六','sex':'女','age':21,'score': 60,'address': '东城区'})
    db.students.insert({'name':'孙七','sex':'男','age':19,'score': 70,'address': '海淀区'})
    db.students.insert({'name':'王八','sex':'女','age':23,'score': 90,'address': '海淀区'})
    db.students.insert({'name':'刘九','sex':'女','age':35,'score': 56,'address': '朝阳区'})
    db.students.insert({'name':'钱十','sex':'男','age':27,'score': 89,'address': '海淀区'})

exp:

1. 查询姓名是张三的学生信息

        db.students.find({name:’张三’}).pretty()

2. 查询性别是男的学生信息

        db.students.find({sex:’男’}).pretty()

3. 查询年龄大于19岁的学生

        db.students.find({age:{$gt:19}}).pretty()

4. 查询成绩大于等于60分的学生

        db.students.find({score:{$gte:60}}).pretty() 

5. 查询姓名不是王五的信息

        db.students.find({name:{$ne:’王五’}}).pretty()

#### 逻辑运算符

- `$and`   与

- `$or`   或

- `$not | $nor`  非

exp:

1. 查询年龄在19 ~ 22岁的学生信息

        db.students.find({age:{$gte:19,$lte:22}}).pretty()

逻辑运算中与连接是最容易的，只需要利用`,`分割多个条件即可

2. 查询年龄小于20岁，或者成绩大于90分的学生信息

        db.students.find(
        {$or:
            [ 
            {age:{$lt:20}},
            {score:{$gt:90}}
            ]
        }).pretty()
    
3. 查询年龄大于等于20岁，且成绩小于等于90分的学生信息

        db.students.find(
        {$and:
            [ 
            {age:{$gte:20}},
            {score:{$lte:90}}
            ]
        }).pretty()

4. 查询年龄小于20岁的学生信息

        db.students.find({age:{$lt:20}}).pretty()
        db.students.find({age:{$not:{$gte:20}}}).pretty()

#### 取模

`$mod:[除数，余数]`

exp: 查询年龄除以20余1的学生信息

    db.students.find({age:{$mod:[20,1]}}).pretty()

#### 范围查询

$in: 在范围之中
$nin: 不在范围之中

exp:

1. 查询姓名是”张三“、”李四、”王五“的学生

        db.students.find({name: {$in:['张三','李四','王五']}}).pret ty()

2. 查询姓名不是”张三“、”李四、”王五“的学生

        db.students.find({name: {$nin:['张三','李四','王五']}}).pretty()

#### 数组查询

- $all 

- $size 

- $slice 

- $elemMatch

首先在数据库中新增一些数据

        db.students.insert({name:'a',sex:'男',age:19,score:89,address:'海淀区',course:['语文','数学','英语','音乐','政治']})
        db.students.insert({name:'b',sex:'男',age:19,score:89,address:'海淀区',course:['语文','数学']})
        db.students.insert({name:'c',sex:'男',age:19,score:89,address:'海淀区',course:['语文','数学','英语']})
        db.students.insert({name:'d',sex:'男',age:19,score:89,address:'海淀区',course:['英语','音乐','政治']})
        db.students.insert({name:'e',sex:'男',age:19,score:89,address:'海淀区',course:['语文','政治']})

`$all`: 表示全都包括，用法：

        {$all:[内容1,内容2]}

exp:

查询同时参加语文和数学的学生

        db.students.find({course:{$all:['语文','数学']}}).pretty()

数组的操作，可以利用索引，使用`key.index`的方式来定义索引

查询数组中第二个内容是数学的学生(sh)

        db.students.find({'course.1':'数学'}).pretty()

`$size`: 控制数组元素数量

exp:

查询只有两门课程的学生

        db.students.find({course:{$size: 2}}).pretty()

`$slice`: 控制查询结果的返回数量

exp:

查询年龄是19岁的学生，要求之显示两门参加的课程

        db.students.find({age:19},{course:{$slice:2}}).pretty()

此时查询返回的是前两门课程，可以设置参数来取出想要的内容

        $slice:-2   //后两门
        $slice: [1,2]   // 第一个参数表示跳过的数据量，第二个参数表示返回的数据量

#### 嵌套集合运算

对象里面套对象

在数据库中新增数据

        db.students.insert(
        {
            name:'A',sex:'男',age:19,score:89,address:'海淀区',
            course:['语文','数学','英语','音乐','政治'],
            parents:[
                {name:'A(father)',age:50,job:'工人'},
                {name:'A(mother)',age:50,job:'职员'}
            ]
        })
        db.students.insert(
        {
            name:'B',sex:'男',age:19,score:89,address:'海淀区',
            course:['语文','数学'],
            parents:[
                {name:'B(father)',age:50,job:'处长'},
                {name:'B(mother)',age:50,job:'局长'}
            ]
        })
        db.students.insert(
        {
            name:'C',sex:'男',age:19,score:89,address:'海淀区',
            course:['语文','数学','英语'],
            parents:[
                {name:'C(father)',age:50,job:'工人'},
                {name:'C(mother)',age:50,job:'局长'}
                ]
        })

对于嵌套的集合中数据的判断只能通过`$elemMatch`完成

语法：`{ <field>: { $elemMatch: { <query1>, <query2>, ... } } } `

exp:

查询父母中有人是局长的信息

        db.students.find({parents: {$elemMatch: {job: '局长'}}}).pretty()

#### 判断某个字段是否存在

`{$exists:flag}`  flag为true表示存在，false表示不存在

exp:

1. 查询具有parents成员的学生

        db.students.find({parents:{$exists: true}}).pretty()

2. 查询不具有course成员的学生

        db.students.find({course: {$exists: false}}).pretty()

#### 排序

`sort({ field: value }) ` value是1表示升序，-1表示降序

exp:

学生信息按照分数降序排列

        db.students.find().sort({score:-1}).pretty()

#### 分页显示

`skip(n)`: 跳过n条数据

`limit(n)`: 返回n条数据

exp:

1. 分页显示，第一页，每页显示5条数据

        db.students.find({}).skip(0).limit(5).pretty()

2. 分页显示，第二页，每页显示5条数据

        db.students.find({}).skip(5).limit(5).pretty()

### 数据修改 | 更新

`updateOne()`     修改匹配的第一条数据

`updateMany()`    修改所有匹配的数据

格式：`updateOne(<filter>,<update>)`

#### 修改器

`$inc`: 操作数字字段的数据内容

语法: `{"$inc" : {成员 : 内容}}`

exp: 

将所有年龄为19岁的学生成绩一律减少30分，年龄增加1

        db.students.updateMany({age:19},{$inc:{score:-30,age:1}})

`$set`: 更新内容

语法：`{$set: :{属性: 新内容}}`

exp: 

将20岁学生的成绩修改为89

        db.students.updateMany({age: 20},{$set: {score: 89}})

`$unset`: 删除某个属性及其内容

语法：`{$unset: {属性: 1}}`

exp:

删除张三的年龄和成绩信息

        db.students.updateOne({name:'张三'},{$unset: {age: 1,score: 1}})

`$push`: 向数组中添加数据

语法：`{$push: {属性: value}}`

exp:

在李四的课程中添加语文

        db.students.updateOne({name: '李四'},{$push: {course: '语文'}})

如果需要向数组中添加多个数据，则需要用到`$each`

exp: 

在李四的课程中添加数学、英语

        db.students.updateOne(
            {name:'李四'},
            {$push:
                {
                    course:{$each: ['数学','英语']}
                }
            }
        )

`$addToSet`: 向数组里面添加一个新的数据

与`$push`的区别，`$push`添加的数据可能是重复的，`$addToSet`只有这个数据不存在时才会添加（去重）

语法：`{$addToSet: {属性：value}}`

exp:

王五新增一门舞蹈课程

        db.students.updateOne(
            {name:'王五'},
            {$addToSet: {course:'舞蹈'}}
        )

`$pop`: 删除数组内的数据

语法：`{$pop: {field: value}}`,value为-1表示删除第一个，value为1表示删除最后一个

exp:

删除王五的第一个课程

        db.students.updateOne({name:'王五'},{$pop:{course:-1}})

只是删除属性的内容，属性还在

`$pull`: 从数组中删除一个指定内容的数据

语法：`{$pull: {field：value}}` 进行数据比对，如果是该数据则删除

exp:

删除李四的语文课程

        db.students.updateOne({name: '李四'},{$pull:{course:'语文'}})

`$pullAll`: 一次删除多个数据

语法：`{$pullAll:{field:[value1,value2...]}}`

exp:

删除a的语文数学英语课程

        db.students.updateOne({name:'a'},{$pullAll:{course:['语文','数学','英语']}})

`$rename`: 属性重命名

语法： `{$rename: {旧属性名：新属性名}}`

exp:

把张三的name属性名改为姓名

        db.students.updateOne({name:'张三'},{$rename:{name:'姓名'}})