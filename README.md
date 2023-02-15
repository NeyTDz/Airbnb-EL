# Airbnb短期旅行租金分析及预测

## 示例

### 依赖

训练模型：sklearn >= 1.0.x, xgboost == 1.7.3, mlxtend == 0.21.0

可视化：matplotlib, plotly >= 5.6.0

### 运行

```python
python main.py
```

可通过 `params.py` 中参数控制训练-测试集划分比例，是否输出训练结果等

## 数据描述

数据集来源为从知名租房网站 Airbnb 上所获取的悉尼周边租房服务的房源信息，具体的数据属性如下：

- $\texttt{description}$：具体描述，即用户评价文本信息

- $\texttt{neighbourhood}$：所在地区，包括悉尼城区及郊区 33 个区域

- $\texttt{latitude},\texttt{longitude}$：纬度，经度

- $\texttt{type}$​：房屋类型，取值：整间（Entire home/apt）、单间（Private room）、

  旅馆房间（Hotel room）和合租间（Shared room）

- $\texttt{accommodates}$：可容纳人数

- $\texttt{bathrooms}$：盥洗室数量

- $\texttt{bedrooms}$：卧室数量

- $\texttt{amenities}$：设施信息，列出房间所含有的基本设施，生活电器等

- $\texttt{reviews}$：评分人数

- $\texttt{review rating}$：评分总分 (0∼100)

- $\texttt{review scoresA}$：指标 A 评分 (0∼10)

- $\texttt{review scoresB}$：指标 B 评分 (0∼10)

- $\texttt{review scoresC}$：指标 C 评分 (0∼10)

- $\texttt{review scoresD}$：指标 D 评分 (0∼10)

- $\texttt{instant bookable}$：是否可以立即预订

- $\texttt{target}$：租金级别（0~5）

其中指标A、B、C、D 的评分经过了匿名化处理，本对应租客对某些指标（如房间卫生程度等）的评分。租金同样被分级为6个级别。

## 数据分析

在对以上数据属性进行分析后，大致将其分为三类：

1. 类别型属性。如 $\texttt{type}$ 和 $\texttt{neighbourhood}$ 等，表示房屋在不同分类下的类别属性，需要先经过编码等数值化处理。
2. 数值型属性。以 $\texttt{review rating}$ 和 $\texttt{accommodates}$ 为代表，数据为明确的评分分值或数量值，只需要进行标准化处理即可。
3. 文本型属性。有且只有 $\texttt{description}$ 和 $\texttt{amenities}$ 两种，为大段的文字信息，需要经过词频统计、关键词提取等 NLP 方法处理。

### 类别型属性分析

首先对不同地区和房屋类型下的数据分布情况进行统计如图1，顶部的饼状图展示了数据在两种属性下各自的分布情况，底部的柱状图展示了综合分布情况：

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="pics_output\dis_type_neigh.png" width = "90%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      图1 不同地区和房屋类型下的数据分布情况
  	</div>
</center>

对于房屋类型的分布，整间和单间的房源占到了所有房源的近99%，整间较多于单间，只有极少数的房源是旅馆房间和合租间，在不同地区下的分布也符合这一规律，因此考虑将不同的房屋类型**重新分为三种**，分别为**整间、单间和其他类型**。图2统计了不同房屋类型的租金级别分布情况，其差异性同样支持上述划分。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="pics_output\type_violin.png" width = "80%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      图2 不同房屋类型的价格级别分布情况
  	</div>
</center>

对于所在地区的分布，实际上代表着地缘因素对价格的影响，因此针对其的分析需要结合经纬度信息在悉尼当地的地图上展开分析。图3在地图上绘制了房源价格的地理分布情况，其中子图 (a)展示的是悉尼及周边地区的整体分布情况。可以看到，作为有名的海滨旅游城市，价格较高的房源无疑在东南沿海地区高度集中，相比之下，西部的郊区和城镇虽也有一部分房源分布，但价格明显远远低于沿海地区。同时，连接悉尼主城区和悉尼北部的海港大桥周边是悉尼的经济和商业中心，人口高度集中，因此也有相当一部分房源位于此处。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="pics_output\neig_tar_dis.png" width = "90%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      图3 房源价格的地理分布情况. 子图 (a) 为房源价格在悉尼及周边地区的总体分布情况，
子图 (b)(c)(d) 分别为高价区、杂价区、低价区的代表性地区的价格分布情况，外围柱状图为详细统计数据
  	</div>
</center>

根据上述分布情况，将 33 个所在地区**分为三类**，分别为**低价区、杂价区和高价区**。低价区即对应西部的郊区和城镇，除去本身样本量较少的地区，如子图 (b) 所展示的 Parramatta，作为西悉尼的市中心，样本总量在所有地区中排到第六位（见图1），但所分布的房源平均价格却相对较低。对比明显的即是分布在东部沿海的高价区，以样本量与其相当的 Manly 为例（如子图 c），凭借其优秀的海滩旅游资源，平均价格在所有地区中高居第四，Wavely 和 Pittwater 也是这一区域的代表。悉尼主城区（如子图 d）则是房源价格呈综合性分布的典型例子，虽然样本数量遥遥领先其他地区，但因为各个价格级别的样本分布较均匀，因而平均价格并不高，其周围的不少小区域，如 Leichhardt 等，也呈此分布态势，因此可以归类于杂价区。

### 数值型数据分析

数值型属性的处理主要需要对数值进行统一标准化和缺失值补全。对于缺失较少的属性，如 $\texttt{bedrooms}$ 等，可以使用平均取值补全。但对于含有大量缺失值的评分总分 $\texttt{review rating}$ 属性，按照平均取值补全的方式会造成较为严重的使数据特征淹没，从而则需要更合理的补全方式。因此此处在无缺失属性的样本上先对已经数值化的属性值进行相关性分析，得到的结果如图 4：

<center>
    <img 
    src="pics_output\corr_notext.jpg" width = "90%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      图4 各属性相关性分析
  	</div>
</center>

可见评分总分 $\texttt{review rating}$ 和指标 A、B、C、D 的分别评分相关性较大，结合数据分析中评分计算的特性，可判断评分总分由上述4类指标评分和可能存在的其他指标评分加权求和得到，故此处在无缺失属性的样本上进行了线性回归拟合，得到下式：
$$
R=3.52R_A+3.23R_B+0.72R_C+2.64R_D-2.71
$$
其中 $R$ 表示评分总分，$R_A ∼ R_D$​ 则表示指标A∼D 的评分。由该式即可实现对评分总分缺失值的补充。

### 文本型数据分析

对 于 文 本 型 属 性 $\texttt{description}$ 和 $\texttt{amenities}$​​​，首先去除停用词以提取关键词，其次使用词频统计和[TF-IDF](https://en.wikipedia.org/wiki/Tf–idf)方法构建文本词向量。对于TF-IDF方法，将取TF-IDF总值最高的n个词作为新的属性替代原有属性，以完成文本信息的数值化。词频统计法则直接取频数最高的n个词即可。图5展示了当取n= 10时，描述属性（左）和设施信息属性（右）按照词频统计法和 TF-IDF 方法计算词向量后排名较高的词语。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="pics_output\des_ame_bar.png" width = "75%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      图5 描述属性和设施信息属性的文本词向量统计
  	</div>
</center>


## 租金预测

对于上述结构化数据，使用逻辑回归（LR）、随机森林（RF)、两种梯度提升算法GBoost和[XGBoost]([[1603.02754\] XGBoost: A Scalable Tree Boosting System (arxiv.org)](https://arxiv.org/abs/1603.02754))进行预测，并最终使用Stacking模型集成以上四个预测模型。预测结果如图6：

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="pics_output\predict.png" width = "85%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      图6 不同模型性能比较
  	</div>
</center>











