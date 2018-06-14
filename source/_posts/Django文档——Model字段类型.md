---
title: Django文档——Model字段类型
date: 2018-06-14 17:13:15
categories:
tags:
---
字段类型(Field types)
AutoField
它是一个根据 ID 自增长的 IntegerField 字段。通常，你不必直接使用该字段。如果你没在别的字段上指定主 键，Django 就会自动添加主键字段。

BigIntegerField
64位整数，类似于IntegerField，范围从-9223372036854775808 到9223372036854775807。默认的form widget 是TextInput。

BooleanField
一个布尔值(true/false)字段。

默认的form widget是CheckboxInput。

如果要使用null作为空值，可使用NullBooleanField。

CharField
  class CharField(max_length=None[, **options])

它是一个字符串字段，对小字符串和大字符串都适用。

对于更大的文本，应该使用TextField 。

默认的form widget是TextInput。

CharField 有一个必须传入的参数：max_length,字段的最大字符数。它作用于数据库层级和 Django 的数据验证层级。

CommaSeparatedInterField
class CommaSeparatedIntegerField(max_length=None[, **options])

它用来存放以逗号间隔的整数序列。和 CharField 一样，必须为它提供 max_length 参数。而且要注意不同数据库对 max_length 的限制。

DateField
  class DateField([auto_now=False, auto_now_add=False, **options])

该字段利用 Python 的 datetime.date 实例来表示日期。下面是它额外的可选参数：

DateField.auto_now：每一次保存对象时，Django 都会自动将该字段的值设置为当前时间。一般用来表示 "最后修改" 时间。要注意使用的是当前日期，而并非默认值，所以

不能通过重写默认值的办法来改变保存时间。

DateField.auto_now_add：在第一次创建对象时，Django 自动将该字段的值设置为当前时间，一般用来表示对象创建时间。它使用的同样是当前日期，而非默认值。

默认的form widget是TextInput。

Note:当auto_now或者auto_now_add设置为True时，字段会有editable=True和blank=True的设定。

DateTimeField
  class DateTimeField([auto_now=False, auto_now_add=False, **options])

该字段利用 datetime.datetime 实例表示日期和时间。该字段所按受的参数和 DateField 一样。

默认的form widget是TextInput。Django 的admin使用两个带有 JavaScript 快捷选项TextInput分别表示日期和时间。

DecimalField
  class DecimalField(max_digits=None, decimal_places=None[, **options])

它是使用 Decimal 实例表示固定精度的十进制数的字段。它有两个必须的参数：

DecimalField.max_digits：数字允许的最大位数

DecimalField.decimal_places：小数的最大位数

例如，要存储的数字最大值是999，而带有两个小数位，你可以使用：

1
models.DecimalField(..., max_digits=5, decimal_places=2)
要存储大约是十亿级且带有10个小数位的数字，就这样写：

1
models.DecimalField(..., max_digits=19, decimal_places=10)
默认的form widget是TextInput。

EmailField
  class EmailField([max_length=75, **options])

它是带有 email 合法性检测的A CharField 。

Note：最大长度默认为75，并不能存储所有与RFC3696/5321兼容的email地址。如果要存储所有，请设置 

max_length=254。设置为75是历史遗留问题。

FileField
class FileField(upload_to=None[, max_length=100, **options])

文件上传字段

Note：该字段不支持 primary_key 和 unique 参数，否则会抛出 TypeError 异常。

它有一个必须的参数：

FileField.upload_to

用于保存文件的本地文件系统。它根据 MEDIA_ROOT 设置确定该文件的 url 属性。

该路径可以包含 时间格式串strftime()，可以在上传文件的时候替换成当时日期／时间(这样，就不会出现在上传文件把某个目录塞满的情况了)。

该参数也可以是一个可调用项，比如是一个函数，可以调用函数获得包含文件名的上传路径。这个可调用项必须要接受两个参数，

并且返回一个保存文件用的 Unix-Style 的路径(用 / 斜杠)。两个参数分别是：

    instance ：定义了当前 FileField 的 model 实例。更准确地说，就是以该文件为附件的 model 实例。

大多数情况下，在保存该文件时， model 实例对象还并没有保存到数据库，这是因为它很有可能使用默认的 AutoField，而此时它还没有从数据库中获得主键值。

    filename ：上传文件的原始名称。在生成最终路径的时候，有可能会用到它。

还有一个可选的参数：

FileField.storage

负责保存和获取文件的对象。

默认的form widget是FileInput。

Note：在 model 中使用 FileField 或 ImageField 要按照以下的步骤：

1.在项目settings文件中，你要定义 MEDIA_ROOT ，将它的值设为用来存放上传文件的目录的完整路径。(基于性能的考虑，Django 没有将文件保存在数据库中).

然后定义 MEDIA_URL ，将它的值设为表示该目录的网址。 

要确保 web 服务器所用的帐号拥有对该目录的写权限。

2.在 model 里面添加 FileField 或 ImageField ，并且确认已定义了 upload_to 项，让 Django 知道应该用 

MEDIA_ROOT 的哪个子目录来保存文件。

3.存储在数据库当中的仅仅只是文件的路径(而且是相对于 MEDIA_ROOT 的相对路径)。你可能已经想到利用 

Django 提供的 url 这个方便的属性。举个例子，如果你的 ImageField 名称是 mug_shot，那么你可以在模板 

中使用

1
{{ object.mug_shot.url }}
就能得到图片的完整网址。

例如，假设你的 MEDIA_ROOT 被设为 '/home/media'，upload_to 被设为 'photos/%Y/%m/%d'。 upload_to 中 

的 '%Y/%m/%d' 是一个strftime()， '%Y' 是四位的年份，'%m' 是两位的月份， '%d' 是两位的日子。如果你 

在2007年01月15号上传了一个文件，那么这个文件就保存在 /home/media/photos/2007/01/15 目录下。

如果你想得到上传文件的本地文件名称，文件网址，或是文件的大小，你可以使用 name, url 和 size 属性。

Note：在上传文件时，要警惕保存文件的位置和文件的类型，这么做的原因是为了避免安全漏洞。对每一个上传 

文件都要验证，这样你才能确保上传的文件是你想要的文件。举个例子，如果你盲目地让别人上传文件，而没有 

对上传文件进行验证，如果保存文件的目录处于 web 服务器的根目录下，万一有人上传了一个 CGI 或是 PHP 

脚本，然后通过访问脚本网址来运行上传的脚本，那可就太危险了。千万不要让这样的事情发生！

默认情况下，FileField 实例在数据库中的对应列是 varchar(100) ，和其他字段一样，你可以利用max_length 参数改变字段的最大长度。

FileField and FieldFile
  class FieldFile

当你访问一个Model的FileField字段时，会得到一个FieldFile的实例作为代理去访问底层文件。实例有几种属性和方法可以用来和文件数据进行互动。

FieldFile.url

通过只读的方式调用底层存储(Storage)类的 url() 方法，来访问该文件的相对URL。

FieldFile.open(mode='rb')

类似于python的open()方法。

FieldFile.close()

类似于python的close()方法。

FieldFile.save(name,content,save=True)

这种方法将filename和文件内容传递到该字段然后存储到该模型。该方法需要两个必须的参数：name， 文件的名称， content， 包含文件内容的对象。save 参数是可选的，主 

要是控制文件修改后实例是否保存。默认是 True 。需要注意的是，content 参数是 django.core.files.File 的一个实例，不是Python的内置File对象。你可以使 

用他从现有的Python文件对象中构建一个文件，如下所示：

1
2
3
4
from django.core.files import File
# Open an existing file using Python's built-in open()
f = open('/tmp/hello.world')
myfile = File(f)
或者从字符串中构造：

1
2
from django.core.files.base import ContentFile
myfile = ContentFile("hello world")
FieldFile.delete(save=True)

删除此实例相关的文件，并清除该字段的所有属性。

Note：当delete()被调用时，如果文件正好是打开的，该方法将关闭文件。

save 参数是可选的，控制文件删除后实例是否保存。默认是 True 。

需要注意的是，当一个模型被删除时，相关文件不被删除。如果想删除这些孤立的文件，需要自己去处理（比如，可以手动运行命令清理，也可以通过cron来定期执行清理命令）

FilePathField
class FilePathField(path=None[, match=None, recursive=False, max_length=100, **options])

它是一个 CharField ，它用来选择文件系统下某个目录里面的某些文件。它有三个专有的参数，只有第一个参 

数是必须的：

FilePathField.path

这个参数是必需的。它是一个目录的绝对路径，而这个目录就是 FilePathField 用来选择文件的那个目录。比 

如： "/home/images".

FilePathField.match

可选参数。它是一个正则表达式字符串， FilePathField 用它来过滤文件名称，只有符合条件的文件才出现在 

文件选择列表中。要注意正则表达式只匹配文件名，而不是匹配文件路径。例如："foo.*\.txt$" 只匹配名为 

foo23.txt 而不匹配 bar.txt 和 foo23.gif。

FilePathField.recursive

可选参数。它的值是 True 或 False。默认值是 False。它指定是否包含 path 下的子目录。

FilePathField.allow_files

该项属于Django1.5新增内容。可选参数，它的值是 True 或 False。默认值是 True。它指定是否包含指定位置的文件。该项与allow_folders 必须有一个是 True。

FilePathField.allow_folders

Django1.5新增内容。可选参数，它的值是True或False。默认是False。它指定是否包含指定位置的目录。该项与allow_files必须有一个是 True。

前面已经提到了 match 只匹配文件名称，而不是文件路径。所以下面这个例子：

1
FilePathField(path="/home/images", match="foo.*", recursive=True)
将匹配 /home/images/foo.gif ，而不匹配 /home/images/foo/bar.gif。这是因为 match 只匹配文件名 
(foo.gif 和 bar.gif).

默认情况下， FilePathField 实例在数据库中的对应列是varchar(100) 。和其他字段一样，你可以利用 max_length 参数改变字段的最大长度。

FloatField
 class FloatField([**options])

该字段在 Python 中使用float 实例来表示一个浮点数。

默认的form widget是TextInput。

请注意FloatField与DecimalField的区别。

ImageField
  class ImageField(upload_to=None[, height_field=None, width_field=None, max_length=100,**options])

和 FileField 一样，只是会验证上传对象是不是一个合法的图象文件。

除了那些在 FileField 中有效的参数之外， ImageField 还可以使用 File.height and File.width 两个属性 。

它有两个可选参数：

ImageField.height_field

保存图片高度的字段名称。在保存对象时，会根据该字段设定的高度，对图片文件进行缩放转换。

ImageField.width_field

保存图片宽度的字段名称。在保存对象时，会根据该字段设定的宽度，对图片文件进行缩放转换。

默认情况下， ImageField 实例对应着数据库中的varchar(100) 列。和其他字段一样，你可以使 用 max_length 参数来改变字段的最大长度。

IntegerField
  class IntegerField([**options])

整数字段。默认的form widget是TextInput。

IPAddressField
 class IPAddressField([**options])

以字符串形式(比如 "192.0.2.30")表示 IP 地址字段。默认的form widget是TextInput。

GenericIPAddressField
 class GenericIPAddressField([**options])

Django1.4新增。

以字符串形式(比如 "192.0.2.30"或者"2a02:42fe::4")表示 IP4或者IP6 地址字段。默认的form widget是TextInput。

IPv6的地址格式遵循RFC 4291 section 2.2。比如如果这个地址实际上是IPv4的地址，后32位可以用10进制数表示，例如 "::ffff:192.0.2.0"。

2001:0::0:01可以写成2001::1,而::ffff:0a0a:0a0a可以写成::ffff:10.10.10.10。字母都为小写。

GenericIPAddressField.protocol

验证输入协议的有效性。默认值是 ‘both’ 也就是IPv4或者IPv6。该项不区分大小写。

GenericIPAddressField.unpack_ipv4

解释IPv4映射的地址，像   ::ffff:192.0.2.1  。如果启用该选项，该地址将必解释为 192.0.2.1 。默认是禁止的。只有当 protocol 被设置为 ‘both’ 时才可以启用。

NullBooleanField
 class NullBooleanField([**options])

与 BooleanField 相似，但多了一个 NULL 选项。建议用该字段代替使用 null=True 选项的 BooleanField 。 

默认的form widget是NullBooleanSelect。

PositiveIntegerField
class PositiveIntegerField([**options])

和 IntegerField 相似，但字段值必须是非负数。

PositiveSmallIntegerField
 class PositiveSmallIntegerField([**options])

和 PositiveIntegerField 类似，但数值的取值范围较小，受限于数据库设置。

SlugField
 class SlugField([max_length=50, **options])

Slug 是一个新闻术语，是指某个事件的短标签。它只能由字母，数字，下划线或连字符组成。通赏情况下，它被用做网址的一部分。

和 CharField 类似，你可以指定 max_length (要注意数据库兼容性和本节提到的 max_length )。如果没有指定 max_length ，Django 会默认字段长度为50。

该字段会自动设置 Field.db_index to True。

基于其他字段的值来自动填充 Slug 字段是很有用的。你可以在 Django 的管理后台中使用prepopulated_fields 来做到这一点。

SmallIntegerField
 class SmallIntegerField([**options])

和 IntegerField 类似，但数值的取值范围较小，受限于数据库的限制。

TextField
 class TextField([**options])

大文本字段。默认的form widget是Textarea。

TimeField
 class TimeField([auto_now=False, auto_now_add=False, **options])

该字段使用 Python 的 datetime.time 实例来表示时间。它和 DateField 接受同样的自动填充的参数。

默认的form widget是TextInput。

URLField
 class URLField([max_length=200, **options])

保存 URL 的 CharField 。

和所有 CharField 子类一样，URLField 接受可选的 max_length 参数，该参数默认值是200。
