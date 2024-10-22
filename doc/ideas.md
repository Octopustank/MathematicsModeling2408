# B题：高速公路车辆轨迹数据的分析应用

## 背景资料与数据一览

* **ETC门架**是车辆通过的检验，但是**存在可能会“漏过”车辆**
* **收费站**是车辆在高速的起点、终点

* 进入高速分为`补录入站`、`ETC入站`（电子收费系统）、`MTC入站`（手动收费系统） 3种
* 出高速分为`补录出站`、`MTC入站` 2种
* 关于出入站的关系：
  * 使用`MTC入站`，则可能存在`补录出站`、`MTC出站` 2种出站可能
  * 使用`ETC入站`，则仅可能`补录出站`
  * 使用`补录入站`，则仅可能`补录出站`
* 任何入站都会记录时间，但是出站**仅有**`MTC出站`有时间记录
* 所有`补录入站`、`补录出站`都不会记录收费站信息
* 车型有：`其他`、`货车`、`客车`
* 行驶方向为小门架号向大门架号（记为正方向）。收费站排列不按号大小排列

## 思考

* 假设高速路上车始终在行驶，并基于我国高速公路的限速`120km/h`（`33.3m/s`），可以进行大致推算：

  用`补录入站`**入站的时间**和经过的下一个**门架**的**位置**和**时间**，先推算行驶**里程**，再得出可能的**入站收费站**

  > 例如`轨迹表1`的`车000001`，`补录入站`时间和经过下一个门架`门架7`的时间间隔为$2437s$，以`120km/s`记行驶距离约$2437 \times 33.3 = 81,233m$，大于了`门架7`和`门架1`的距离$72790m$。考虑实际行驶情况，可以推断是在`门架1`负方向的`收费站13`进入的。

* 在保证相对位置的前提下，收费站可以视为在两个门架之间的任何位置。

  门架是检测车辆经过的手段，门架之间没有确定车辆位置的手段。收费站都分布在门架之间。收费站在确定的门架之间时，因为不能确定车辆位置，所以收费站具体在哪里没有意义。并且唯一相邻而不被门架分隔的收费站15和10在图中十分靠近。所以讨论收费站的具体位置并没有意义，只需要确定收费站和门架的相对位置即可。

## 问题1

### 数据处理

提取所有车辆经过门架的信息，进行绘图。X轴为1-15号门架，Y轴为时间区间的中点，Z轴为对应门架、对应时间区间内通过的车辆的计数。时间区间划分依据所有数据的最大时间、最小时间，进行10或60等分。

### 得出结论

#### 时空分布

**初步观察：**时间区间为10等分时的图像，可以发现`门架7`、`门架8`、`门架9`、`门架10`相对来说通过的车辆最多。即：**靠近路段中央，通过车辆较多；靠近路段两端，通过车辆较少**。同时关注到大约在`2022年3月2日 03:53:30`**之后，车辆通过数几乎全部归0**。

**具体观察：**再观察时间区间为60等分时的图像局部图，可以发现**在一天内，车辆通过数存在约为`24h`的周期性变化，凌晨2点前后时段车辆通过数最少，中午1点前后车辆通过数最多**。大约在`2022年2月28日 06:11:15`**之后，车辆通过数几乎全部归0**。

#### 关于门架维修

**初步观察：**时间区间为10等分时的图像，可见存在1处异常低的点，出现在`门架6`，时间约在`2022年2月24日 17:18:30`前后。再观察时间区间为60等分时的图像，可以确定存在一段时间，`门架6`记录通过的车数量为0。故**存在1个门架`门架6`有一段时间在维修**。

**具体观察：**再观察时间区间为60等分时的图像局部图，可以读出，`门架6`记录通过的车数量为0的时间区间约为`2022年2月23日 17:22:05`到`2022年2月25日 08:32:35`。故可以初步确定，**`门架6`大约在`2022年2月23日 17:22:05`到`2022年2月25日 08:32:35`这段时间内在维修**。
