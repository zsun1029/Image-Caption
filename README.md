需要安装的依赖有
===
python 							   3.6

torch							   0.3.1

Pillow							   4.2.1

nltk                               3.2.4

Cython                             0.26.1

pycocotools                        2.0

pyparsing                          2.2.0

cycler                             0.10.0

torchvision                        0.2.0

数据集
===
MSCOCO-caption

各个文件的作用
===

util.py用于dataset部分

savemodel.py用于手动转化模型存储方式

plot.py用于毕设论文画图

testone.py用于单条测试，用于和在线demo连接

language.lua用于在单条测试时将英文数据集翻译为中文

训练与测试方法在main.py中
