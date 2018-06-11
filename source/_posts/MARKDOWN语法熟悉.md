---
title: MARKDOWN语法熟悉
date: 2018-05-28 17:03:28
categories: 工具
tags: 工具 Markdown
mathjax: true
---
# 基础

### **标题测试**


# 一级标题
## 二级标题
### 三级标题
#### 四级标题
##### 五级标题
###### 六级标题

### 换行和分段

#### 未换行
未换行测试
示例

#### 换行后
已换行测试（后有两个空格）  
示例

#### 分段
分段测试

分段

### 文本样式

**加粗(使用两个*号)**  
*斜体（使用一个*号）*  
~~删除线(使用两个波浪线)~~  
'底纹（单引号）'

### 列表

在Markdown 下，无序列表直接在文字前加 「 - 」 或者 「 * 」 即可，有序列表则直接在文字前加 「1.」「2.」「3.」 。符号要和文字之间加上一个字符的空格。

无序列表： 在文本前加 「 * 」 即可生成一个无序列表。快捷键：control + L （只能生成列表，不能生成子列表）
在 「 * 」 前加两个空格键或者一个 tab 键就可以产生一个子列表。
有序列表： 在文本前加 「字母.」 或 「数字.」 即可生成一个有序列表。
注意，当你第一个序号使用什么作为标记的，那么同级别的列表就会自动使用其作为标记。

#### 无序列表

* 1  
* 2  
    * 2.1  
        * 2.1.1  
            * 2.1.1.1
* 3

#### 有序列表

1. 1
2. 2
    a. 2.1
    b. 2.2
3. 3

#### 有序列表与无序列表混排

1. 1
2. 2
    a. 2.1
    b. 2.2
        * 2.2.1

### 引用

只要在文本内容之前加 「 > （大于号）」 即可将文本变成引用文本。

> 这是引用文本

### 图片与链接

#### 图片

![Mou icon](http://mouapp.com/Mou_128.png)

#### 链接

[Mou](http://25.io/mou/)

### 水平线

三个「 - 」或「 * 」都可以画出一条水平分割线

---
使用（---）的水平分割线

***
使用（***）的水平分割线

### 代码框

两对「 ``` 」包裹  
代码前加四个空格键  
代码前加一个 tab 键

#### 两对‘ ``` ’包裹

```print('Hello Word!');```

#### 四个空格

    print('Hello Word!');

#### 一个 tab 键

    print('Hello Word!');


### 表格

在 Markdown 下制作表格，是非常非常麻烦的一件事——你必须画出来！  
（本人比较懒，故略去 -_-*）


### 脚注

脚注总是成对出现的，「 [^1] 」作为标记，可以点击跳至末尾注解。「 [^1]: 」填写注解，不论写在什么位置，都会出现在文章的末尾。

点击右上方的小数字有注解[^1]  
[^1]:这里是注解  
这是随机文本  
这是随机文本  
这是随机文本  

### 注释

注释是给自己看的，预览时也不会出现，当然发布出去别人也不会看见。  
<!--注释，无法预览-->

### 首行缩进

关于首行缩进，网上争议很多，而技术本身并没有错，不是吗？在输入法的「全角」模式下，输入两个空格键即可。  

### 引号

在网页上写文章建议使用直角引号『「」』。

# 利用Markdown创建表格

Markdown作为一种轻量级书写/写作语言，并没有提供很好的排版、编辑等功能。因此，如果想要利用Markdown创建表格（特别是复杂表格），其实是一项不太轻松的事情。 
经过笔者在简书平台上的测试与其他若干帖子的表述，Markdown应是只提供了最简单的创建表格与内容对齐方式的功能。总结而言，有如下两种最为直观的创建表格方式:

### 简单方式

    Name | Academy | score  
    - | :-: | -:  
    Harry Potter | Gryffindor| 90  
    Hermione Granger | Gryffindor | 100  
    Draco Malfoy | Slytherin | 90

Name | Academy | score  
- | :-: | -:  
Harry Potter | Gryffindor| 90  
Hermione Granger | Gryffindor | 100  
Draco Malfoy | Slytherin | 90  

### 原生方式

    | Name | Academy | score |  
    | - | :-: | -: |  
    | Harry Potter | Gryffindor| 90 |  
    | Hermione Granger | Gryffindor | 100 |  
    | Draco Malfoy | Slytherin | 90 |
| Name | Academy | score |  
| - | :-: | -: |  
| Harry Potter | Gryffindor| 90 |  
| Hermione Granger | Gryffindor | 100 |  
| Draco Malfoy | Slytherin | 90 |

### 语法说明：

1. 不管是哪种方式，第一行为表头，第二行分隔表头和主体部分，第三行开始每一行代表一个表格行； 
2. 列与列之间用管道符号 “|” 隔开，原生方式的表格每一行的两边也要有管道符。 
3. 可在第二行指定不同列单元格内容的对齐方式，默认为左对齐，在 “-” 右边加上 “:” 为右对齐，在 “-” 两侧同时加上 “:” 为居中对齐。

这样傻瓜的表格创建方式十分符合Markdown简小精悍的语言气质，具有上手快、即学即用的优势。但傻瓜的定义方式显然不能满足很多处女座的要求，比如 
文章——“Linux备忘录-Linux中文件/文件夹按照时间顺序升序/降序排列”的表格如下：

    | 参数 |详细解释|备注| 
    | - | :-: | -: | 
    | -l | use a long listing format |以长列表方式显示（显示出文件/文件夹详细信息） | 
    | -t | sort by modification time |按照修改时间排序（默认最近被修改的文件/文件夹排在最前面） | 
    |-r | reverse order while sorting |逆序排列|  

| 参数 |详细解释|备注| 
| - | :-: | -: | 
| -l | use a long listing format |以长列表方式显示（显示出文件/文件夹详细信息） | 
| -t | sort by modification time |按照修改时间排序（默认最近被修改的文件/文件夹排在最前面） | 
|-r | reverse order while sorting |逆序排列|


单元格排列不齐整、第一列太窄而第三列略宽，如此不堪的视觉效果着实让强迫症患者们难以忍受。还好，利用HTML可以弥补Markdown这一缺陷，甚至可以在创建表格时其他诸多表现方面锦上添花。

# Markdown 添加 MathJax 数学公式

添加公式的方法
行内公式

    $行内公式$
行间公式

    $$行间公式$$

### MathJax 数学公式语法

#### 呈现位置

**注意:**  在公式的前一行和后一行，要注意空一行，否则公式会出错。

- 所有公式定义格式为  

    \$...$
- 具体语句例如  

    \$\sum_{i=0}^N\int_{a}^{b}g(t,i)\text{d}t$  
    显示为：  

    $\sum_{i=0}^N\int_{a}^{b}g(t,i)\text{d}t$  
- 居中并放大显示

    \$\$\sum_{i=0}^N\int_{a}^{b}g(t,i)\text{d}t$$  
    显示为：

    $$\sum_{i=0}^N\int_{a}^{b}g(t,i)\text{d}t$$

#### 希腊字母

|显示|命令|显示|命令|
| - | :-: | :-: | -:|
|α 	|\$\alpha\$ |β	|\$\beta\$|
|γ	|\$\gamma\$	|δ	|\$\delta\$|
|ϵ	|\$\epsilon\$	|ζ	|\$\zeta\$|
|η	|\$\eta\$	    |θ	|\$\theta\$|
|ι	|\$\iota\$	|κ	|\$\kappa\$|
|λ	|\$\lambda\$	|μ	|\$\mu\$|
|ν	|\$\nu\$	    |ξ	|\$\xi\$|
|π	|\$\pi\$	    |ρ	|\$\rho\$|
|σ	|\$\sigma\$	|τ	|\$\tau\$|
|υ	|\$\upsilon\$	|ϕ	|\$\phi\$|
|χ	|\$\chi\$	    |ψ	|\$\psi\$|
|ω	|\$\omega\$

- 如果需要大写的希腊字母，只需将命令的首字母大写即可(有的字母没有大写)，如  
    \$\gamma$ & \$\Gamma$

$\gamma$ & $\Gamma$

- 若需要斜体希腊字母，在命令前加上var前缀即可(大写可斜)，如  
    \$\Gamma$ & \$\varGamma$  

$\Gamma$ & $\varGamma$


#### 字母修饰

##### 上下标

- 上标：^  
- 下标：_  

\$C_n^2$  

$$C_n^2$$

##### 矢量

- 例1  

\$\vec a$  

$\vec a$

- 例2  

\$\overrightarrow a$  

$\overrightarrow xy$

##### 字体 - Typewriter

\$\mathtt {ABCDEFGHIJKLMNOPQRSTUVWXYZ}$  

$\mathtt {ABCDEFGHIJKLMNOPQRSTUVWXYZ}$  

\$\mathbb {ABCDEFGHIJKLMNOPQRSTUVWXYZ}$  

$\mathbb {ABCDEFGHIJKLMNOPQRSTUVWXYZ}$  

\$\mathsf {ABCDEFGHIJKLMNOPQRSTUVWXYZ}$  

$\mathsf {ABCDEFGHIJKLMNOPQRSTUVWXYZ}$  

##### 分组 - {}有分组功能，如

\$10^{10}\$ \& \$10^10\$  
$10^{10}$ & $10^10$

##### 括号

- 小括号：\$()$呈现为

$()$  

- 中括号：\$[]$呈现为

$[]$  

- 尖括号：\$\langle\rangle$呈现为

$\langle\rangle$  

    - 此处为与分组符号{}相区别，使用转义字符\

- 使用\left(或\right)使符号大小与邻近的公式相适应；该语句适用于所有括号类型
    - \$(\frac{x}{y})$呈现为

    $(\frac{x}{y})$  

    - 而\$\left(\frac{x}{y}\right)$呈现为

    $\left(\frac{x}{y}\right)$

**注意:** 在公式的前后，必须留有一个空格或者换行，否则无法识别。

##### 求和、极限与积分

求和：\sum  

- 举例：\$\sum_{i=1}^n{a_i}$  
    - $\sum_{i=1}^n{a_i}$ 

极限：\$\lim_{x\to 0}$  
- $\lim_{x\to 0}$ 

积分：\$\int$  
- $\int$  

    举例：\$\int_0^\infty{fxdx}$  
    - $\int_0^\infty{fxdx}$

##### 分式与根式

分式(fractions)：\$\frac{公式1}{公式2}$  

- $\frac{公式1}{公式2}$  

根式：\$\sqrt[x]{y}$

- $\sqrt[x]{y}$

##### 特殊符号

|显示|命令|
| - | -:|
|∞   |               \$\infty$ 
|∪   |               \$\cup$
|∩   |               \$\cap$ 
|⊂   |               \$\subset$ 
|⊆   |               \$\subseteq$ 
|⊃   |               \$\supset$ 
|∈   |               \$\in$ 
|∉   |                \$\notin$ 
|∅   |                \$\varnothing$ 
|∀   |                \$\forall$
|∃   |                \$\exists$ 
|¬   |                \$\lnot$ 
|∇   |                \$\nabla$ 
|∂   |                \$\partial$


##### 空格

- LaTeX语法本身会忽略空格的存在
- 小空格：\$a\ b$呈现为

$a\ b$
- 4格空格：\$a\quad b$呈现为

$a\quad b$

##### 矩阵边框
在起始、结束标记处用下列词替换matrix  

- pmatrix：小括号边框 
- bmatrix：中括号边框 
- Bmatrix：大括号边框 
- vmatrix：单竖线边框 
- Vmatrix：双竖线边框

##### 省略元素

横省略号：\cdots  
竖省略号：\vdots  
斜省略号：\ddots  
举例  

    $$\begin{bmatrix}{a_{11}}&{a_{12}}&{\cdots}&{a_{1n}}\\\
                    {a_{21}}&{a_{22}}&{\cdots}&{a_{2n}}\\\
                    {\vdots}&{\vdots}&{\ddots}&{\vdots}\\\
                    {a_{m1}}&{a_{m2}}&{\cdots}&{a_{mn}}\\\
                    \end{bmatrix}$$

$$\begin{bmatrix}{a_{11}}&{a_{12}}&{\cdots}&{a_{1n}}\\\  
    {a_{21}}&{a_{22}}&{\cdots}&{a_{2n}}\\\  
    {\vdots}&{\vdots}&{\ddots}&{\vdots}\\\  
    {a_{m1}}&{a_{m2}}&{\cdots}&{a_{mn}}\\\  
    \end{bmatrix}$$


##### 阵列
需要array环境：起始、结束处以{array}声明  
- 对齐方式：在{array}后以{}逐行统一声明  
- 左对齐：l；居中：c；右对齐：r  
- 竖直线：在声明对齐方式时，插入|建立竖直线  
- 插入水平线：\hline


方程组
需要cases环境：起始、结束处以{cases}声明  
举例  

    $$\begin{cases}
        a_1x+b_1y+c_1z=d_1\\\
        a_2x+b_2y+c_2z=d_2\\\
        a_3x+b_3y+c_3z=d_3\\\
        \end{cases}$$


$$\begin{cases}  
    a_1x+b_1y+c_1z=d_1\\\
    a_2x+b_2y+c_2z=d_2\\\  
    a_3x+b_3y+c_3z=d_3\\\  
    \end{cases}$$

