## 1. 目标检测的基本介绍

https://zhuanlan.zhihu.com/p/34142321

### 1.1 目标检测的任务

**图像理解的层次**：图像分类>>目标检测>>语义分割，图片级->区域级->像素级

- 分类：根据给定的的类别或实例ID，判断图像所属类型

  过闸机人脸检测、手写数字识别、智能分拣水果

- 检测：找出图像中的单个或多个目标，判断类别、位置

  行人识别、医学图像（细胞计数、病灶检测）、农作物、森林火灾

- 分割：将图像中属于不同语义部分（物体、背景等）的像素进行分割。包含语义分割（区分不同种类）和实例分割（区分每一个物体）

  医学（计算机引导手术、病灶体积测量）、智能驾驶（建筑物、车辆、行人、道路分割）、生物识别（指纹、虹膜）



**目标检测的基本要素**：标注框、物体类别、置信度

- 标注框：物体轮廓的外接矩形，形状不固定

- 物体类别：在给定的若干种类别范围中进行判断

- 置信度：框内物体属于某种类别的准确度，如果置信度过低则可能不属于该分类

<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*OCsh4qf4lLoRAY-rlSZmJw.png" style="zoom: 80%;" />

<img src="https://www.researchgate.net/publication/342409316/figure/fig5/AS:905798087630850@1592970501809/b-semantic-segmentation-c-instance-segmentation-and-d-panoptic-segmentation-for.png" style="zoom: 67%;" />

​	

**图像的基本单元**：像素，RGB值，三个通道

​		数字图像是二维像素的表示，每个像素具有 RGB 值 [0-255, 0-255, 0-255]

​		RGB 三种颜色的单色图片 = 三个通道

<img src="https://www.malaoshi.top/upload/0/0/1EF49Kr1WGvI.png" style="zoom: 80%;" />

<img src="https://img-blog.csdnimg.cn/20190902194522797.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDIyNTE4Mg==,size_16,color_FFFFFF,t_70" style="zoom:50%;" />



### 1.2 目标检测的算法：传统方法vs神经网络

**传统方法**：选择候选区域、在候选区域中提取特征、进行分类

- 候选区域：滑动窗口


- 提取特征：SIFT、HOG、color names

  SIFT：尺度不变特征变换，HOG：方向梯度直方图

  https://zhuanlan.zhihu.com/p/85829145

  <img src="https://img-blog.csdn.net/20180418141658272?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2ppYW95YW5nd20=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" style="zoom:50%;" />

  <img src="https://picx.zhimg.com/v2-8fbabd144e2651fb58bf0533ead90af6_1440w.jpg?source=172ae18b" style="zoom:50%;" />

  

  Color Names：图像颜色空间直方图（11种颜色空间）+PCA 降维

  https://lear.inrialpes.fr/people/vandeweijer/papers/icip07.pdf

  <img src="https://www.researchgate.net/publication/340281516/figure/fig3/AS:875316083961858@1585703025154/The-feature-combines-the-Color-Name-feature-with-the-FHOG-feature.png" style="zoom: 67%;" />

- 分类：SVM、AdaBoost

  SVM：根据图片的不同滑窗，形成正负样本，然后用滑窗的区域特征训练出单个分类下的 SVM 模型。如果多个类别则使用多个分类器即可

  AdaBoost：弱分类器使用特征值进行简单的阈值对比，按照权重线性相加/树状组合形成强分类器

  <img src="https://pic2.zhimg.com/80/v2-fb32a9f1246a572c59d9c0aef96f08e9_720w.webp" style="zoom:80%;" />

  https://zhuanlan.zhihu.com/p/31427728



**神经网络方法**

- 两阶段模型 RCNN：使用CNN特征进行目标检测


​				选择性搜索筛选出候选框，候选框使用CNN提取特征+SVM分类器进行预测 AlexNet

​				选择性搜索：通过纹理、边缘、颜色等进行自底向上的分割，然后不同尺度进行合并

​							贪心策略计算相邻区域的相似度，每一次合并最相似的两块，最后合并成原图

​							相似度：颜色距离+纹理距离+区域大小+区域外接矩形重合面积

​							属于传统机器学习方法，后来被RPN（Region Proposal Network）所取代

​				先定位，后识别，需要训练RPN和区域检测两个网络

- 单阶段模型 YOLO


​				将图像分割成NxN区域，预测每个区域的边界框和概率

​				与滑窗和后续区域划分的检测方法不同，他把检测任务当做一个regression问题来处理，使用一个神经网络，直接从一整张图像来预测出bounding box 的坐标、box中包含物体的置信度和物体所属类别概率，可以实现端到端的检测性能优化

​				直接使用主干网络给出预测

​				学习任务：前景背景、分类、边界框坐标。与two-stage中的区域检测不同，区域检测/分类网络不需要学习前景和背景也不需要定位，而是交给RPN来做

### 1.4 目标检测的常见数据集

​	性能判断标准：IoU（DIoU、CIoU）、Precision、Recall ......

​	MS-COCO、Pascal VOC(Visual Object Classes)、ImageNet

## 2. Yolo检测框架介绍

基本特点：单阶段、快

**整体框架**

​	backbone特征提取 + neck特征进一步整理 + head定位和分类

​	backbone		

​			一般使用大型数据集如 ImageNet COCO 等进行预训练，拥有预训练的参数而不是随机参数

​			用卷积神经网络进行提取特征 VGG ResNet-50 Darknet53等

​	neck

​			FPN特征金字塔等整理backbone提取的所有特征信息

​			上下采样/尺寸变换、多尺度特征加权融合

​	head

​			根据backbone和neck提供的特征进行定位和分类，找到 bounding box

​			实现：CBL（Conv+BN+Leaky relu）+ Conv

​			多头机制

**Yolov3 检测方式**

​	图片输入 416 x 416 x 3

​	图片经过 backbone 变成特征图

​	特征图采样成大中小三份

​	划分网格：大 13 x 13，中 26 x 26，小 52 x 52 + 标准预测框：每个 size 3 种，一共 9 种

​	每个网格根据标准框，预测中心落在网格内的物体：3 x (4 xywh + 1 score + 80 one-hot 分类)

​				(13 x 13 + 26 x 26 + 52 x 52) * 3 = 10467 个预测框

​				4 + 1 + 80 = 85 每个预测框的输出维度

​	NMS：根据置信度、IoU 等排除重复框

<img src="https://pic1.zhimg.com/80/v2-4cf1b6f6afec393122305ca2bb2725a4_720w.webp" style="zoom: 67%;" />

## 3. Yolov5的应用

**Yolov5的项目基本介绍**

​	github主页 https://github.com/ultralytics/yolov5

​	v1-v3 作者 Joseph Redmon，v5 作者 Glenn Jocher	

​	安装方式 git clone + pip install

​	总体使用：detect 输入图片视频检测目标、train 从配置文件和数据集训练模型、val 使用数据集验证模型、export 将模型导出成不同格式

​	使用便捷：基于四个脚本完成，有规范的readme等文档说明

​	讨论度高+长期支持：在 GitHub 的 issue 知乎各大博客B站等都有资料，作者团队长期活跃维护更新和解答问题

​	框架稳定：Yolo系列的框架保持整体一致，只是不同单元有变化，使用起来的脚本和接口也是，对比其他五花八门的算法可能在依赖包、推理方式等都有很大不同

**Yolov5的检测功能展示**

​	使用样例图片/视频检测目标

```shell
python detect.py --weights yolov5s.pt --source img.jpg
```

​	调用 detect.py，实现单张图片、视频、多张图片的检测，标注图片+txt

```shell
python detect.py --weights test/yolov5s.pt --source test/bus.jpg
```

​	调用 val.py，实现数据集的检测和结果统计

```shell
python val.py --weights yolov5s.pt --data coco128.yaml --img 640
```

**Yolov5自定义物料模型的训练流程**

​	自制数据集：视频素材收集、labelme标注、文件夹整理

​	调用 train.py 训练模型

​	训练的后续工作：数据验证、查看模型效果

**后面的改进**

​	遮罩：修改 detect.py，用黑白图实现遮罩

​	Yolov8：有更规范的程序结构、文档、新版的模型结构和预训练模型

​	目标跟踪：在检测的基础上跟踪物体的运动信息，可能还要结合 re-ID

​	工具化：结合网盘、脚本实现素材管理、半自动标注、自动训练等内容





网上总结的比较好的：

官网介绍文档 https://github.com/ultralytics/yolov5/blob/master/README.zh-CN.md

简易使用教程 https://lightningleader.github.io/posts/15.html

数据增强 https://blog.csdn.net/weixin_41868104/article/details/114685071

推理、训练、验证脚本参数调整 https://blog.csdn.net/weixin_43694096/article/details/124378167

区域遮罩（实现方式比较麻烦，后面我修改了遮罩方式） https://blog.csdn.net/qq_39740357/article/details/125149010

one-stage vs two-stage detection https://zhuanlan.zhihu.com/p/161194349

Yolov1-v5 原理详解 https://zhuanlan.zhihu.com/p/183261974

训练时遇到的连接超时问题 https://github.com/ultralytics/yolov5/issues/1062