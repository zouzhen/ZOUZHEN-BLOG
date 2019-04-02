---
title: pandas常用方法整理
date: 2019-04-02 15:32:28
categories: Python第三方包
tags: [pandas]
---
## pandas dataframe的合并（append, merge, concat）
创建2个DataFrame：

    >>> df1 = pd.DataFrame(np.ones((4, 4))*1, columns=list('DCBA'), index=list('4321'))  
    >>> df2 = pd.DataFrame(np.ones((4, 4))*2, columns=list('FEDC'), index=list('6543'))  
    >>> df3 = pd.DataFrame(np.ones((4, 4))*3, columns=list('FEBA'), index=list('6521'))  
    >>> df1  
        D    C    B    A  
    4  1.0  1.0  1.0  1.0    
    3  1.0  1.0  1.0  1.0  
    2  1.0  1.0  1.0  1.0
    1  1.0  1.0  1.0  1.0
    >>> df2
        F    E    D    C
    6  2.0  2.0  2.0  2.0
    5  2.0  2.0  2.0  2.0
    4  2.0  2.0  2.0  2.0
    3  2.0  2.0  2.0  2.0
    >>> df3
        F    E    B    A
    6  3.0  3.0  3.0  3.0
    5  3.0  3.0  3.0  3.0
    2  3.0  3.0  3.0  3.0
    1  3.0  3.0  3.0  3.0
    　　

### 1，concat

#### pd.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, copy=True)
 
 示例：

    >>> pd.concat([df1, df2])
        A    B    C    D    E    F
    4  1.0  1.0  1.0  1.0  NaN  NaN
    3  1.0  1.0  1.0  1.0  NaN  NaN
    2  1.0  1.0  1.0  1.0  NaN  NaN
    1  1.0  1.0  1.0  1.0  NaN  NaN
    6  NaN  NaN  2.0  2.0  2.0  2.0
    5  NaN  NaN  2.0  2.0  2.0  2.0
    4  NaN  NaN  2.0  2.0  2.0  2.0
    3  NaN  NaN  2.0  2.0  2.0  2.0
　　

#### 1.1，axis

默认值：axis=0

axis=0：竖方向（index）合并，合并方向index作列表相加，非合并方向columns取并集

axis=1：横方向（columns）合并，合并方向columns作列表相加，非合并方向index取并集

axis=0：

    >>> pd.concat([df1, df2], axis=0)
        A    B    C    D    E    F
    4  1.0  1.0  1.0  1.0  NaN  NaN
    3  1.0  1.0  1.0  1.0  NaN  NaN
    2  1.0  1.0  1.0  1.0  NaN  NaN
    1  1.0  1.0  1.0  1.0  NaN  NaN
    6  NaN  NaN  2.0  2.0  2.0  2.0
    5  NaN  NaN  2.0  2.0  2.0  2.0
    4  NaN  NaN  2.0  2.0  2.0  2.0
    3  NaN  NaN  2.0  2.0  2.0  2.0　
axis=1：

    >>> pd.concat([df1, df2], axis=1)
        D    C    B    A    F    E    D    C
    1  1.0  1.0  1.0  1.0  NaN  NaN  NaN  NaN
    2  1.0  1.0  1.0  1.0  NaN  NaN  NaN  NaN
    3  1.0  1.0  1.0  1.0  2.0  2.0  2.0  2.0
    4  1.0  1.0  1.0  1.0  2.0  2.0  2.0  2.0
    5  NaN  NaN  NaN  NaN  2.0  2.0  2.0  2.0
    6  NaN  NaN  NaN  NaN  2.0  2.0  2.0  2.0
备注：原df中，取并集的行/列名称不能有重复项，即axis=0时columns不能有重复项，axis=1时index不能有重复项：

    >>> df1.columns = list('DDBA')
    >>> pd.concat([df1, df2], axis=0)
    ValueError: Plan shapes are not aligned
 

#### 1.2，join

默认值：join=‘outer’

非合并方向的行/列名称：取交集（inner），取并集（outer）。

axis=0时join='inner'，columns取交集：

    >>> pd.concat([df1, df2], axis=0, join='inner')
        D    C
    4  1.0  1.0
    3  1.0  1.0
    2  1.0  1.0
    1  1.0  1.0
    6  2.0  2.0
    5  2.0  2.0
    4  2.0  2.0
    3  2.0  2.0
axis=1时join='inner'，index取交集：

    >>> pd.concat([df1, df2], axis=1, join='inner')
        D    C    B    A    F    E    D    C
    4  1.0  1.0  1.0  1.0  2.0  2.0  2.0  2.0
    3  1.0  1.0  1.0  1.0  2.0  2.0  2.0  2.0
    

#### 1.3，join_axes

默认值：join_axes=None，取并集

合并后，可以设置非合并方向的行/列名称，使用某个df的行/列名称

axis=0时join_axes=[df1.columns]，合并后columns使用df1的：

    >>> pd.concat([df1, df2], axis=0, join_axes=[df1.columns])
        D    C    B    A
    4  1.0  1.0  1.0  1.0
    3  1.0  1.0  1.0  1.0
    2  1.0  1.0  1.0  1.0
    1  1.0  1.0  1.0  1.0
    6  2.0  2.0  NaN  NaN
    5  2.0  2.0  NaN  NaN
    4  2.0  2.0  NaN  NaN
    3  2.0  2.0  NaN  NaN
axis=1时axes=[df1.index]，合并后index使用df2的：

    >>> pd.concat([df1, df2], axis=1, join_axes=[df1.index])
        D    C    B    A    F    E    D    C
    4  1.0  1.0  1.0  1.0  2.0  2.0  2.0  2.0
    3  1.0  1.0  1.0  1.0  2.0  2.0  2.0  2.0
    2  1.0  1.0  1.0  1.0  NaN  NaN  NaN  NaN
    1  1.0  1.0  1.0  1.0  NaN  NaN  NaN  NaN

同时设置join和join_axes的，以join_axes为准：

    >>> pd.concat([df1, df2], axis=0, join='inner', join_axes=[df1.columns])
        D    C    B    A
    4  1.0  1.0  1.0  1.0
    3  1.0  1.0  1.0  1.0
    2  1.0  1.0  1.0  1.0
    1  1.0  1.0  1.0  1.0
    6  2.0  2.0  NaN  NaN
    5  2.0  2.0  NaN  NaN
    4  2.0  2.0  NaN  NaN
    3  2.0  2.0  NaN  NaN
　　

#### 1.4，ignore_index

默认值：ignore_index=False

合并方向是否忽略原行/列名称，而采用系统默认的索引，即从0开始的int。

axis=0时ignore_index=True，index采用系统默认索引：

    >>> pd.concat([df1, df2], axis=0, ignore_index=True)
        A    B    C    D    E    F
    0  1.0  1.0  1.0  1.0  NaN  NaN
    1  1.0  1.0  1.0  1.0  NaN  NaN
    2  1.0  1.0  1.0  1.0  NaN  NaN
    3  1.0  1.0  1.0  1.0  NaN  NaN
    4  NaN  NaN  2.0  2.0  2.0  2.0
    5  NaN  NaN  2.0  2.0  2.0  2.0
    6  NaN  NaN  2.0  2.0  2.0  2.0
    7  NaN  NaN  2.0  2.0  2.0  2.0
 axis=1时ignore_index=True，columns采用系统默认索引：

    >>> pd.concat([df1, df2], axis=1, ignore_index=True)
        0    1    2    3    4    5    6    7
    1  1.0  1.0  1.0  1.0  NaN  NaN  NaN  NaN
    2  1.0  1.0  1.0  1.0  NaN  NaN  NaN  NaN
    3  1.0  1.0  1.0  1.0  2.0  2.0  2.0  2.0
    4  1.0  1.0  1.0  1.0  2.0  2.0  2.0  2.0
    5  NaN  NaN  NaN  NaN  2.0  2.0  2.0  2.0
    6  NaN  NaN  NaN  NaN  2.0  2.0  2.0  2.0
    

#### 1.5，keys

默认值：keys=None

可以加一层标签，标识行/列名称属于原来哪个df。

axis=0时设置keys：

    >>> pd.concat([df1, df2],  axis=0, keys=['x', 'y'])
          A    B    C    D    E    F
    x 4  1.0  1.0  1.0  1.0  NaN  NaN
      3  1.0  1.0  1.0  1.0  NaN  NaN
      2  1.0  1.0  1.0  1.0  NaN  NaN
      1  1.0  1.0  1.0  1.0  NaN  NaN
    y 6  NaN  NaN  2.0  2.0  2.0  2.0
      5  NaN  NaN  2.0  2.0  2.0  2.0
      4  NaN  NaN  2.0  2.0  2.0  2.0
      3  NaN  NaN  2.0  2.0  2.0  2.0
 axis=1时设置keys：

    >>> pd.concat([df1, df2], axis=1, keys=['x', 'y'])
        x                   y              
        D    C    B    A    F    E    D    C
    1  1.0  1.0  1.0  1.0  NaN  NaN  NaN  NaN
    2  1.0  1.0  1.0  1.0  NaN  NaN  NaN  NaN
    3  1.0  1.0  1.0  1.0  2.0  2.0  2.0  2.0
    4  1.0  1.0  1.0  1.0  2.0  2.0  2.0  2.0
    5  NaN  NaN  NaN  NaN  2.0  2.0  2.0  2.0
    6  NaN  NaN  NaN  NaN  2.0  2.0  2.0  2.0　
也可以传字典取代keys：

    >>> pd.concat({'x': df1, 'y': df2}, axis=0)
          A    B    C    D    E    F
    x 4  1.0  1.0  1.0  1.0  NaN  NaN
      3  1.0  1.0  1.0  1.0  NaN  NaN
      2  1.0  1.0  1.0  1.0  NaN  NaN
      1  1.0  1.0  1.0  1.0  NaN  NaN
    y 6  NaN  NaN  2.0  2.0  2.0  2.0
      5  NaN  NaN  2.0  2.0  2.0  2.0
      4  NaN  NaN  2.0  2.0  2.0  2.0
      3  NaN  NaN  2.0  2.0  2.0  2.0
　　

#### 1.6，levels

默认值：levels=None

明确行/列名称取值范围：

    >>> pd.concat([df1, df2], axis=0, keys=['x', 'y'], levels=[['x', 'y', 'z', 'w']])
    >>> df.index.levels
    [['x', 'y', 'z', 'w'], ['1', '2', '3', '4', '5', '6']]
    

#### 1.7，sort

默认值：sort=True，提示新版本会设置默认为False，并取消该参数

但0.22.0中虽然取消了，还是设置为True

非合并方向的行/列名称是否排序。例如1.1中默认axis=0时columns进行了排序，axis=1时index进行了排序。

axis=0时sort=False，columns不作排序：

    >>> pd.concat([df1, df2], axis=0, sort=False)
        D    C    B    A    F    E
    4  1.0  1.0  1.0  1.0  NaN  NaN
    3  1.0  1.0  1.0  1.0  NaN  NaN
    2  1.0  1.0  1.0  1.0  NaN  NaN
    1  1.0  1.0  1.0  1.0  NaN  NaN
    6  2.0  2.0  NaN  NaN  2.0  2.0
    5  2.0  2.0  NaN  NaN  2.0  2.0
    4  2.0  2.0  NaN  NaN  2.0  2.0
    3  2.0  2.0  NaN  NaN  2.0  2.0
axis=1时sort=False，index不作排序：

    >>> pd.concat([df1, df2], axis=1, sort=False)
        D    C    B    A    F    E    D    C
    4  1.0  1.0  1.0  1.0  2.0  2.0  2.0  2.0
    3  1.0  1.0  1.0  1.0  2.0  2.0  2.0  2.0
    2  1.0  1.0  1.0  1.0  NaN  NaN  NaN  NaN
    1  1.0  1.0  1.0  1.0  NaN  NaN  NaN  NaN
    6  NaN  NaN  NaN  NaN  2.0  2.0  2.0  2.0
    5  NaN  NaN  NaN  NaN  2.0  2.0  2.0  2.0
　

#### 1.8，concat多个DataFrame

    >>> pd.concat([df1, df2, df3], sort=False, join_axes=[df1.columns])
        D    C    B    A
    4  1.0  1.0  1.0  1.0
    3  1.0  1.0  1.0  1.0
    2  1.0  1.0  1.0  1.0
    1  1.0  1.0  1.0  1.0
    6  2.0  2.0  NaN  NaN
    5  2.0  2.0  NaN  NaN
    4  2.0  2.0  NaN  NaN
    3  2.0  2.0  NaN  NaN
    6  NaN  NaN  3.0  3.0
    5  NaN  NaN  3.0  3.0
    2  NaN  NaN  3.0  3.0
    1  NaN  NaN  3.0  3.0
　　

### 2，append

#### append(self, other, ignore_index=False, verify_integrity=False)
竖方向合并df，没有axis属性

不会就地修改，而是会创建副本

示例：

    >>> df1.append(df2)    # 相当于pd.concat([df1, df2])
        A    B    C    D    E    F
    4  1.0  1.0  1.0  1.0  NaN  NaN
    3  1.0  1.0  1.0  1.0  NaN  NaN
    2  1.0  1.0  1.0  1.0  NaN  NaN
    1  1.0  1.0  1.0  1.0  NaN  NaN
    6  NaN  NaN  2.0  2.0  2.0  2.0
    5  NaN  NaN  2.0  2.0  2.0  2.0
    4  NaN  NaN  2.0  2.0  2.0  2.0
    3  NaN  NaN  2.0  2.0  2.0  2.0　　
　　

#### 2.1，ignore_index属性

    >>> df1.append(df2, ignore_index=True)
        A    B    C    D    E    F
    0  1.0  1.0  1.0  1.0  NaN  NaN
    1  1.0  1.0  1.0  1.0  NaN  NaN
    2  1.0  1.0  1.0  1.0  NaN  NaN
    3  1.0  1.0  1.0  1.0  NaN  NaN
    4  NaN  NaN  2.0  2.0  2.0  2.0
    5  NaN  NaN  2.0  2.0  2.0  2.0
    6  NaN  NaN  2.0  2.0  2.0  2.0
    7  NaN  NaN  2.0  2.0  2.0  2.0
    

#### 2.2，append多个DataFrame

和concat相同，append也支持append多个DataFrame

    >>> df1.append([df2, df3], ignore_index=True)
        A    B    C    D    E    F
    0   1.0  1.0  1.0  1.0  NaN  NaN
    1   1.0  1.0  1.0  1.0  NaN  NaN
    2   1.0  1.0  1.0  1.0  NaN  NaN
    3   1.0  1.0  1.0  1.0  NaN  NaN
    4   NaN  NaN  2.0  2.0  2.0  2.0
    5   NaN  NaN  2.0  2.0  2.0  2.0
    6   NaN  NaN  2.0  2.0  2.0  2.0
    7   NaN  NaN  2.0  2.0  2.0  2.0
    8   3.0  3.0  NaN  NaN  3.0  3.0
    9   3.0  3.0  NaN  NaN  3.0  3.0
    10  3.0  3.0  NaN  NaN  3.0  3.0
    11  3.0  3.0  NaN  NaN  3.0  3.0
    　　

### 3，merge

#### pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=True, suffixes=('_x', '_y'), copy=True, indicator=False,validate=None)
示例：　　

    >>> left = pd.DataFrame({'A': ['a0', 'a1', 'a2', 'a3'],
                            'B': ['b0', 'b1', 'b2', 'b3'],
                            'k1': ['x', 'x', 'y', 'y']})
    >>> right = pd.DataFrame({'C': ['c1', 'c2', 'c3', 'c4'],
                              'D': ['d1', 'd2', 'd3', 'd4'],
                              'k1': ['y', 'y', 'z', 'z']})
    >>> left
        A   B  k1
    0  a0  b0  x
    1  a1  b1  x
    2  a2  b2  y
    3  a3  b3  y
    >>> right
        C   D  k2
    0  c1  d1  y
    1  c2  d2  y
    2  c3  d3  z
    3  c4  d4  z
对df1和df2进行merge：

    >>> pd.merge(left, right)
        A   B  k1  C   D
    0  a2  b2  y  c1  d1
    1  a2  b2  y  c2  d2
    2  a3  b3  y  c1  d1
    3  a3  b3  y  c2  d2
可以看到只有df1和df2的key1=y的行保留了下来，即默认合并后只保留有共同列项并且值相等行（即交集）。

本例中left和right的k1=y分别有2个，最终构成了2*2=4行。

如果没有共同列会报错：

    >>> del left['k1']
    >>> pd.merge(left, right)
    pandas.errors.MergeError: No common columns to perform merge on
 

#### 3.1，on属性

新增一个共同列，但没有相等的值，发现合并返回是空列表，因为默认只保留所有共同列都相等的行：

    >>> left['k2'] = list('1234')
    >>> right['k2'] = list('5678')
    >>> pd.merge(left, right)
    Empty DataFrame
    Columns: [B, A, k1, k2, F, E]
    Index: []
可以指定on，设定合并基准列，就可以根据k1进行合并，并且left和right共同列k2会同时变换名称后保留下来：

    >>> pd.merge(left, right, on='k1')
        A   B  k1  k2_x   C   D   k2_y
    0  a2  b2  y     3    c1  d1   5
    1  a2  b2  y     3    c2  d2   6
    2  a3  b3  y     4    c1  d1   5
    3  a3  b3  y     4    c2  d2   6
默认值：on的默认值是所有共同列，本例为：on=['k1', 'k2']

 

3.2，how属性

how取值范围：'inner', 'outer', 'left', 'right'

默认值：how='inner'

‘inner’：共同列的值必须完全相等：

    >>> pd.merge(left, right, on='k1', how='inner')
        A   B  k1  k2_x   C   D   k2_y
    0  a2  b2  y     3    c1  d1   5
    1  a2  b2  y     3    c2  d2   6
    2  a3  b3  y     4    c1  d1   5
    3  a3  b3  y     4    c2  d2   6
‘outer’：共同列的值都会保留，left或right在共同列上的差集，会对它们的缺失列项的值赋上NaN：

    >>> pd.merge(left, right, on='k1', how='outer')
        A    B k1   k2_x  C    D  k2_y
    0   a0   b0  x    1  NaN  NaN  NaN
    1   a1   b1  x    2  NaN  NaN  NaN
    2   a2   b2  y    3   c1   d1    5
    3   a2   b2  y    3   c2   d2    6
    4   a3   b3  y    4   c1   d1    5
    5   a3   b3  y    4   c2   d2    6
    6  NaN  NaN  z  NaN   c3   d3    7
    7  NaN  NaN  z  NaN   c4   d4    8
‘left’：根据左边的DataFrame确定共同列的保留值，右边缺失列项的值赋上NaN：

    pd.merge(left, right, on='k1', how='left')
        A   B k1  k2_x  C    D   k2_y
    0  a0  b0  x    1  NaN  NaN  NaN
    1  a1  b1  x    2  NaN  NaN  NaN
    2  a2  b2  y    3   c1   d1    5
    3  a2  b2  y    3   c2   d2    6
    4  a3  b3  y    4   c1   d1    5
    5  a3  b3  y    4   c2   d2    6
‘right’：根据右边的DataFrame确定共同列的保留值，左边缺失列项的值赋上NaN：

    >>> pd.merge(left, right, on='k1', how='right')
        A    B k1  k2_x  C   D   k2_y
    0   a2   b2  y    3  c1  d1    5
    1   a3   b3  y    4  c1  d1    5
    2   a2   b2  y    3  c2  d2    6
    3   a3   b3  y    4  c2  d2    6
    4  NaN  NaN  z  NaN  c3  d3    7
    5  NaN  NaN  z  NaN  c4  d4    8
　　

#### 3.3，indicator
默认值：indicator=False，不显示合并方式

设置True表示显示合并方式，即left / right / both：

    >>> pd.merge(left, right, on='k1', how='outer', indicator=True)
        A    B k1  k2_x  C    D   k2_y     _merge
    0   a0   b0  x    1  NaN  NaN  NaN   left_only
    1   a1   b1  x    2  NaN  NaN  NaN   left_only
    2   a2   b2  y    3   c1   d1    5        both
    3   a2   b2  y    3   c2   d2    6        both
    4   a3   b3  y    4   c1   d1    5        both
    5   a3   b3  y    4   c2   d2    6        both
    6  NaN  NaN  z  NaN   c3   d3    7  right_only
    7  NaN  NaN  z  NaN   c4   d4    8  right_only
