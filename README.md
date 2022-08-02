# 飞桨学习赛：遥感影像地块分割——22年7月第一名：66分方案

<font size=5>比赛链接：[飞桨学习赛：遥感影像地块分割](https://aistudio.baidu.com/aistudio/competition/detail/63/0/introduction)
</font>
    
## 赛题介绍
本赛题由 2020 CCF BDCI 遥感影像地块分割 初赛赛题改编而来。遥感影像地块分割, 旨在对遥感影像进行像素级内容解析，对遥感影像中感兴趣的类别进行提取和分类，在城乡规划、防汛救灾等领域具有很高的实用价值，在工业界也受到了广泛关注。现有的遥感影像地块分割数据处理方法局限于特定的场景和特定的数据来源，且精度无法满足需求。因此在实际应用中，仍然大量依赖于人工处理，需要消耗大量的人力、物力、财力。本赛题旨在衡量遥感影像地块分割模型在多个类别（如建筑、道路、林地等）上的效果，利用人工智能技术，对多来源、多场景的异构遥感影像数据进行充分挖掘，打造高效、实用的算法，提高遥感影像的分析提取能力。 赛题任务 本赛题旨在对遥感影像进行像素级内容解析，并对遥感影像中感兴趣的类别进行提取和分类，以衡量遥感影像地块分割模型在多个类别（如建筑、道路、林地等）上的效果。

## 数据说明
本赛题提供了多个地区已脱敏的遥感影像数据，各参赛选手可以基于这些数据构建自己的地块分割模型。

![](https://ai-studio-static-online.cdn.bcebos.com/ed833cc5d6fd4d03a8fcb8628ba47e9d2d471e5e606c45f2976c61eb76ede711)

其中有效的标签被定义为0，1，2，3。255像素值区域为未标记区域。

## 比赛难点

本比赛实质上还是一个语义分割的任务，只是相对于在遥感影像上的迁移。

* 个人认为本次比赛的难点在于类与类之间的分割界面不容易区分，对于模型细粒度的分割提出了要求。

* 而且对于后处理部分来说（分割任务涨点比较多的办法）由于类与类之间界限比较模糊，加上类别形状的不规则，难以找到有效的后处理办法涨分。

## 本项目亮点

* 整体基于PaddleSeg套件进行完成，易于上手改进，学习。

* 有效的数据增强方法

* 构建类别3难与训练的有效重采样办法，一定程度上解决类别3难以学习的问题。

* 构建基于SegFormer改进的Decoder，一定程度上解决模型细粒度不够的问题。

### PaddleSeg套件开发

本项目使用的模型及数据集处理方法，均为在PaddleSeg原有的基础上进行改进，主要的改进部分有数据集加载处理及模型部分结构更改部分。

* 对于数据集的处理，主要从paddleseg/datasets/dataset.py更改

* 对于模型的选择及更改，则主要从paddleseg/models/segformer.py更改


### 数据增强部分

本项目从已有开源项目，和自己的消融实验，主要确立使用数据增强方法如下：



| 编号 | 数据增强方法 |得分：MIOU |
| -------- | -------- | -------- |
| A     | RandomHorizontalFlip+RandomVerticalFlip+RandomPaddingCrop    | 56.48     |
| B     | A+RandomBlur    | 59.04     |
| C     | B+ResizeStepScalin(0.75:1.25:0.25)     | 62.91     |
| D     | B+ResizeStepScalin(0.75:1.25:0.25)     | 63.41     |
| E     | D+RandomDistort     | 64.35     |
| F     | F+RandomRotation     | 64.72     |

**上述方法都基于未更改模型和数据集处理的得分,且都使用了TTA的预测方法**

### 数据集处理部分

这里的数据集处理，主要是针对预测结果进行分析，并且获得。

**从几次训练结果分析，可以知道模型对于类别3的学习，存在一定的困难，但其实如果从面积的角度去分析的话，三个类别的分布是相当的，但鉴于此，我选择了在训练过程中，对类别3单独进行重采样的处理方法**

![](https://ai-studio-static-online.cdn.bcebos.com/3fe37dea5f494172849860a28d916c625ccfe421f2274fa285d748f44f016bc6)

具体的代码在myconfig里的mydataset.py文件里。大致的思路就是去构建含有3类别的数据，在训练中以一定的概率对这些数据进行采样。相关代码如下，其实也很简单，就是一定概率下，这次的采样数据从含有类别3的数据里找。

```
            if np.random.random()<self.sample_prob:
                idx = idx%len(self.sample_list)
                image_path, label_path = self.sample_list[idx]
            im, label = self.transforms(im=image_path, label=label_path)
            if self.edge:
                edge_mask = F.mask_to_binary_edge(
                    label, radius=2, num_classes=self.num_classes)
                return im, label, edge_mask
            else:
                return im, label
```

**这里的采样概率，采用的是以10%的概率对每次采样时，从含类别3中的数据中进行采样，当然这里也可以以递增的方法进行采样，但要改的代码比较多且最后都结果并没有以稳定10%概率采样效果好。**

### SegFormer模型改进

SegFormer是一款基于Transformer构建的具有简单结构的语义分割网络。其基础网络结构如下图所示。

![](https://ai-studio-static-online.cdn.bcebos.com/eef5b861bcfc40c7a8c6d92184d5e6a2518c2260e2ba4c0086dc15e6553a7b91)

这里重点关注模型的Decodr部分。原基础模型的Decoder部分，简单的说就是一个MLP来将不同特种层输出的特征进行融合，然后经过一个MLP再进行上采样。

这里给出两个改进的思路：

1. 对MLP特征融合进行改进

具体可以参考像FPN和PAN结构的金字塔特征融合，来对特征进行充分挖掘。

2. 对上采样部分进行改进


在原基础SegFormer结构里，上采样部分，用的是最简单的双线性插值的方法。在本项目中，针对上采样方法进行改进，旨在增强模型的细节还原能力。具体来说，即将上采样的部分更换成具有学习能力的转置卷积，来提升模型对细节的还原。

![](https://ai-studio-static-online.cdn.bcebos.com/13c11b12ab044b8a9d91c0667e9c7e22ae5cb6f217634cfcaa7a44a32a8b3875)


具体代码详见myconfig/segformer.py文件。

经过上述两部分操作，得分概览如下。



| 编号 | 方法| 得分：MIOU |
| -------- | -------- | -------- |
| G     | F+数据集重采样     | 66.04     |
| H     | G+SegFormer模型改进     | 66.18     |

### 其他细节

本项目还存在以下一些细节部分

* 训练时采用混合Loss：分别为0.8：CrossEntropyLoss和0.2：LovaszSoftmaxLoss

* 预测时采用了TTA方法，分别有垂直翻转和0.75，1.0和1.25倍尺寸的多尺度预测。

下面则是本项目的训练过程

