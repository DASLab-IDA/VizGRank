# VizGRank

#### 介绍
这是一种基于可视化内在关系的上下文感知的可视化推荐方法。该方法将可视化之间的关系建模为图，并采用一种基于图形的算法来计算可视化的重要性。在这个模型中，来自可视化编码和底层数据模式的关系被用于构建图并完成可视化排序从而生成推荐。


#### 依赖环境

1.	Python 3.7.3
2.	numpy 1.19.4
3.	pandas 1.1.5
4.	matplotlib 3.3.0
5.	networkx 2.5
6.	other dependencies in ‘requirement.txt’

#### 主要文件

VizGRank主要包含文件vizrank.py，node_relation.py和personalization.py。vizrank.py包含可视化生成和推荐的主要逻辑，node_relation.py包含定义不同的可视化间关系，personalization.py包含的则是ranking过程中的排序偏好。

#### 操作方法（API）

a)第一步，导入VizGRank类。
![导入VizGRank类](https://images.gitee.com/uploads/images/2021/0603/150725_b155a843_9100839.png "f1.png")

b)第二步，指定输入数据文件和数据类型文件。

![指定输入数据文件和数据类型文件](https://images.gitee.com/uploads/images/2021/0603/150851_a14f7bf7_9100839.png "f2.png")

c)第三步，读取数据文件，生成候选可视化，对可视化进行排序，并输出结果

![输出结果](https://images.gitee.com/uploads/images/2021/0603/150910_92fb9ae1_9100839.png "f3.png")



#### 输入包括数据文件路径和指定数据类型。

a)	数据文件路径csv_path，对应文件格式为csv格式。样例见example下data.csv
b)	以json格式指定csv中各列的数据类型，使用Pandas中的数据类型。样例见example下types.json

####  输出：html格式的渲染完成的可视化结果，样例见html下的data_all.html

### Citation

Qianfeng Gao, Zhenying He, Yinan Jing, Kai Zhang, and X. Sean Wang. 2021. VizGRank: A Context-Aware Visualization Recommendation Method Based on Inherent Relations Between Visualizations. In Database Systems for Advanced Applications: 26th International Conference, DASFAA 2021, Taipei, Taiwan, April 11–14, 2021, Proceedings, Part III. Springer-Verlag, Berlin, Heidelberg, 244–261. https://doi.org/10.1007/978-3-030-73200-4_16
