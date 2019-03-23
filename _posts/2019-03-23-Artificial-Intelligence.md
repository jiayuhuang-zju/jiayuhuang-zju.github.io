## 人工智能

[TOC]



### 搜索求解

> - Uninformed Search and Informed (Heuristic) Search 
> - Adversarial Search: Minimax Search, Evaluation Functions, Alpha-Beta Search, Stochastic Search
> - Adversarial Search: Multi-armed bandits, Upper Confidence Bound (UCB),Upper Confidence Bounds on Trees, Monte-Carlo Tree Search (MCTS)

#### 搜索算法的形式化描述:

<状态、动作、状态转移、路径、测试目标>

#### 启发式搜索(有信息搜索)：

- 代表算法为**贪婪最佳优先搜索**和**A*搜索**

![image-20190320014330080](/Users/miracle/Library/Application Support/typora-user-images/image-20190320014330080.png)

- **贪婪最佳优先搜索**(Greedy best-first search): 评价函数f(n)=启发函数h(n)

  - 不足之处： 

    - 贪婪最佳优先搜索不是最优的
    - 启发函数代价最小化这一目标会对错误的起点比较敏感
    - 婪最佳优先搜索也是不完备的。所谓不完备即它可能沿着一条无限的路径走下去而不回来
      做其他的选择尝试，因此无法找到最佳路径这一答案。
    - 在最坏的情况下，贪婪最佳优先搜索的时间复杂度和空间复杂度都是$O(𝑏^𝑚)$，其中𝑏是节点的
      分支因子数目、𝑚是搜索空间的最大深度

    原因在于贪婪优先搜索没有好的启发函数，因此我们要**设计更好的启发函数**

- **A*算法**：

  - 定义评价函数 $f(𝑛) =g(𝑛) +h(𝑛)$. g(n)表示从起始节点到节点𝑛的开销代价值，h(n)表示从节点𝑛到目标节点路径中所估算的最小开销代价值。f(n)可视为经过节点𝑛 、具有最小开销代价值的路径。

    ![image-20190320015706463](/Users/miracle/Library/Application Support/typora-user-images/image-20190320015706463.png)

  

  - 为了保证A*算法是最优 (optimal) ，需要启发函数h(𝑛)是可容的(admissible heuristic)和一致(consistency，或者也称单调性，即 monotonicity)。

    ![image-20190320015424922](/Users/miracle/Library/Application Support/typora-user-images/image-20190320015424922.png)

  - A*算法保持最优的条件:启发函数具有可容性(admissible)和一致性(consistency)。
    - 将直线距离作为启发函数h(𝑛)，则启发函数一定是可容的，因为其不会高估开销代价。
    - 𝑔(𝑛) 是从起始节点到节点𝑛的实际代价开销，且𝑓(𝑛) = 𝑔(𝑛) + h(𝑛)，因此𝑓(𝑛)不会高估经过节点𝑛路径的实际开销。
    - $h(𝑛) ≤ 𝑐(𝑛, 𝑎, 𝑛′) + h(𝑛′)$ 构成了三角不等式。这里节点𝑛、节点𝑛′和目标结点𝐺𝑛之间组成了一个三角形。如果存在一条经过节点𝑛′ ，从节点𝑛到目标结点𝐺𝑛的路径，其代价开销小于h(𝑛)，则破坏了h(𝑛)是从节点𝑛到目标结点𝐺𝑛 所形成的具有最小开销代价的路径这一定义。
  - Tree-search的A*算法中，如果启发函数h(n)是可容的，则A\*算法是最优的和完备的;在Graph-search的A\*算法中，如果启发函数h(n)是一致的，A*算法是最优的。
  - 如果函数满足一致性条件，则一定满足可容条件;反之不然。
  - 直线最短距离函数既是可容的，也是一致的
  - 如果h(n)是一致的(单调的)，那么𝑓(𝑛) 一定是非递减的(non-decreasing)
  - 如果A*算法将节点n选择作为具有最小代价开销的路径中一个节点，则𝑛一定是最优路径中的一个节点。即最先被选中扩展的节点在最优路径中。



#### 对抗搜索

主要讨论在确定的、全局可观察的、竞争对手轮流行动、零和游戏(zero-sum)下的对抗搜索

两人对决游戏 (MAX and MIN, MAX先走) 可如下形式化描述，从而将其转换为对抗搜索问题

![image-20190320020901860](/Users/miracle/Library/Application Support/typora-user-images/image-20190320020901860.png)

- **minimax算法**：给定一个游戏搜索树，minimax算法通过每个节点的minimax值来决定最优策略。当然，MAX希望最大化minimax值，而MIN则相反。

  ![image-20190320021206003](/Users/miracle/Library/Application Support/typora-user-images/image-20190320021206003.png)

  ![image-20190320021239472](/Users/miracle/Library/Application Support/typora-user-images/image-20190320021239472.png)
  - Time complexity $O(𝑏^𝑚)$, Space complexity  $O(b × 𝑚)$ (depth-first exploration)。 m是游戏树的最大深度，在每个节点存在b个有效走法。
  - 优点:
    - 算法是一种简单有效的对抗搜索手段 
    - 在对手也“尽力而为”前提下，算法可返回最优结果 
  - 缺点
    - 如果搜索树极大，则无法在有效时间内返回结果 
  - 改善
    - 使用alpha-beta pruning算法来减少搜索节点 
    -  对节点进行采样、而非逐一搜索 (i.e., MCTS) 

- **Alpha-Beta 剪枝搜索**

  在极小化极大算法(minimax算法)中减少所搜索的搜索树节点数。该算法和极小化极大算法所得结论相同，但剪去了不影响最终结果的搜索分枝。

  ![image-20190320022149759](/Users/miracle/Library/Application Support/typora-user-images/image-20190320022149759.png)

  - 𝛼 为可能解法的最大上界，𝛽 为可能解法的最小下界
  - 如果节点 𝑁 是可能解法路径中的一个节点，则其产生的收益一定满足如下条件: 𝛼 ≤ 𝑟𝑒𝑤𝑎𝑟𝑑(𝑁) ≤ 𝛽(其中𝑟𝑒𝑤𝑎𝑟𝑑(𝑁)是节点𝑁产生的收益) 
  - 剪枝本身不影响算法输出结果
  - 节点先后次序会影响剪枝效率
  - 如果节点次序“恰到好处”，Alpha-Beta剪枝的时间复杂度为$O(𝑏^{\frac{m}{2}})$，最小最大搜索的时间复杂度为$O(𝑏^𝑚)$ 

  ![image-20190320022805350](/Users/miracle/Library/Application Support/typora-user-images/image-20190320022805350.png)

- **蒙特卡洛树搜索**

  - 单一状态蒙特卡洛规划: 多臂赌博机 (multi-armed bandits)

    多臂赌博机问题是一种序列决策问题，这种问题需要在利用(exploitation)和探索(exploration) 之间保持平衡。 利用(exploitation) :保证在过去决策中得到最佳回报；探索(exploration) :寄希望在未来能够得到更大回报。

    ![image-20190320023417875](/Users/miracle/Library/Application Support/typora-user-images/image-20190320023417875.png)

    

  - 上限置信区间 (Upper Confidence Bound, UCB)

    ![image-20190320023528489](/Users/miracle/Library/Application Support/typora-user-images/image-20190320023528489.png)

    ![image-20190320023611622](/Users/miracle/Library/Application Support/typora-user-images/image-20190320023611622.png)

- **蒙特卡洛树搜索 Monte-CarloTreeSearch(MCTS)**

  蒙特卡洛树搜索基于采样来得到结果、而非穷尽式枚举(虽然在枚举过程中也可剪掉若干不影响结果的分支)。

  - 选择
    - 从根节点 R 开始，向下递归选择子节点，直至选择一个叶子节点L。 
    - 具体来说，通常用UCB1(Upper Confidence Bound，上限置信区间)选择最具“潜力”的后续节点

  - 扩展
    - 如果 L 不是一个终止节点(即博弈游戏不)，则随机创建其 后的一个未被访问节点，选择 该节点作为后续子节点C。 
  - 模拟
    - 从节点 C出发，对游戏进行模拟， 直到博弈游戏结束。 
  - 反向传播
    - 用模拟所得结果来回溯更新导致这个结果的每个节点中获胜次数和访问次数

  两种策略学习机制

  - 搜索树策略: 从已有的搜索树中选择或创建一个叶子结点(即蒙特卡洛中选择和拓展两 个步骤).搜索树策略需要在利用和探索之间保持平衡。 
  - 模拟策略:从非叶子结点出发模拟游戏，得到游戏仿真结果。

- **蒙特卡洛树搜索算法 (Upper Confidence Bounds on Trees , UCT)**
  - 𝑺 : The set of states. 

  - 𝑨(𝒔) : The set of valid and unvisited actions for state 𝑠 . 

  - 𝒔(𝒗) : The state that 𝑣 represents. 

  - 𝒂(𝒗) : The action that leading to 𝑣 . 

  - 𝒇 : 𝑆 × A → 𝑆 , the state transition function. 

  - 𝑵(𝒗) : The number of times node 𝑣 has been visited. 

  - 𝑸(𝑽) : The total reward of the action leading to 𝑣 . 

  - 𝚫(𝒗, 𝒑) : The reward for the player 𝑝 to move at node 𝑣 . 

    

### 逻辑与推理

#### 命题逻辑

![image-20190320030135170](/Users/miracle/Library/Application Support/typora-user-images/image-20190320030135170.png)

![image-20190320030148674](/Users/miracle/Library/Application Support/typora-user-images/image-20190320030148674.png)

“如果𝑝那么𝑞(𝑝 ⟶ 𝑞)”定义的是一种蕴涵关系(即充分条件)，也就是命题𝑞 包含着命题𝑝 ( 𝑝是𝑞的子集) 

𝑝不成立相当于𝑝是一个空集，空集可被其他所有集合所包含，因此当𝑝不成立时，“如果𝑝那么𝑞”永远为真。 

逻辑等价:给定命题𝑝和命题𝑞，如果𝑝和𝑞在所有情况下都具有同样真假结果，那么𝑝和𝑞在逻辑上等价，一般用≡来表示，即𝑝 ≡ 𝑞。

- 命题逻辑中的推理规则

  ![image-20190320030635502](/Users/miracle/Library/Application Support/typora-user-images/image-20190320030635502.png)

- 命题范式
  - 有限个简单合取式构成的析取式称为析取范式，有限个简单析取式构成的合取式称为合取范式，析取范式与合取范式统称为范式 (normal form)。
  - 假设$𝛼_𝑖(𝑖 = 1,2, ... , 𝑘)$为简单的合取式，则$𝛼 = 𝛼_1 ∨ 𝛼_2 ∨ ⋯ ∨ 𝛼_𝑘$为析取范式
  - 假设$𝛼_𝑖(𝑖 = 1,2, ... , 𝑘)$为简单的析取式，则$𝛼 = 𝛼_1 ∧ 𝛼_2 ∧ ⋯ ∧ 𝛼_𝑘$为合取范式
  - 一个析取范式是不成立的，当且仅当它的每个简单合取式都不成立 
  - 一个合取范式是成立的，当且仅当它的每个简单析取式都是成立的。 
  - 任一命题公式都存在着与之等值的析取范式与合取范式

#### 谓词逻辑

核心概念：个体、谓词(predicate)和量词(quantifier)

- 全称量词(universal quantifier, ∀)：∀𝑥𝑃(𝑥)表示定义域中的所有个体具有性质𝑃
- 存在量词(existential quantifier, ∃)：∃𝑥𝑃(𝑥)表示定义域中存在一个个体或若干个体具有性质𝑃 



#### 知识图谱

- 知识图谱的的构成

  - 概念:层次化组织
  - 实体:概念的示例化描述
  - 属性:对概念或实体的描述信息
  - 关系:概念或实体之间的关联
  - 推理规则:可产生语义网络中上述新的元素

  知识图谱一般可通过标注多关系图(labeled multi-relational graph)来表示

  知识图谱中存在连线的两个实体可表达为形如<left_node, relation, right_node >的三元组形式，这种三元组也可以表示为一阶逻辑(first order logic, FOL)的形式。

- 知识图谱推理:FOIL (First Order Inductive Learner)

  归纳逻辑程序设计 (inductive logic programming，ILP)算法

  ILP使用一阶谓词逻辑进行知识表示，通过修改和扩充逻辑表达式对现有知识归纳，完成推理任务。

  作为ILP的代表性方法，FOIL(First Order Inductive Learner)通过**序贯覆盖**实现规则推理。

  只能在已知两个实体的关系且确定其关系与目标谓词相悖时，才能将这两个实体用于构建目标谓词的反例，而不能在不知两个实体是否满足目标谓词前提下将它们来构造目标谓词的反例。

  推理思路:从一般到特殊，逐步给目标谓词添加前提约束谓词，直到所构成的推理规则不覆盖任何反例。 

  ![image-20190320033841372](/Users/miracle/Library/Application Support/typora-user-images/image-20190320033841372.png)



### 监督学习

#### 机器学习基本概念

- 损失函数

  训练集中一共有𝑛个标注数据，第𝑖个标注数据记为$(𝑥_𝑖,𝑦_𝑖)$,其中第𝑖个样本数据为$𝑥_𝑖$，$𝑦_𝑖$是$𝑥_𝑖$的标注信息。从训练数据中学习得到的映射函数记为𝑓,$f$对$𝑥_𝑖$的预测结果记为$𝑓(𝑥_𝑖)$ 。损失函数就是用来计算$𝑥_𝑖$真实值$𝑦_𝑖 $与预测值$𝑓(𝑥_𝑖 )$之间差值的函数。在训练过程中希望映射函数在训练数据集上得到 “损失”之和最小，即$min\sum_{i=1}^{n}Loss(f(x_i),y_i)$

  ![image-20190320044841500](/Users/miracle/Library/Application Support/typora-user-images/image-20190320044841500.png)

- 经验风险(empirical risk ):训练集中数据产生的损失。经验风险越小说明学习模型对训练数据拟合程度越好。 

​       期望风险(expected risk):当测试集中存在无穷多数据时产生的损失。期望风险越小，学习所得模型越好。

​       过学习(over-fitting)与欠学习(under-fitting)

​       结构风险最小化(structural risk minimization)：为了防止过拟合，在经验风险上加上表示模型复杂度的正则	化项(regulatizer)或惩罚项(penalty term): 在最小化经验风险与降低模型复杂度之间寻找平衡。

- 判别模型与生成模型

  监督学习方法又可以分为生成方法(generative approach)和判别方法(discriminative approach)。所学到的模型分别称为生成模型(generative model)和判别模型(discriminative model).

  - 判别方法直接学习判别函数𝑓(𝑋) 或者条件概率 分布𝑃(𝑌|𝑋) 作为预测的模型，即判别模型。 
  - 判别模型关心在给定输入数据下，预测该数据的输出是什么。典型判别模型包括回归模型、神经网络、支持向量机和Ada boosting等。 
  - 生成模型从数据中学习联合概率分布𝑃(𝑋, 𝑌)(通过似然概率𝑃(𝑋|𝑌) 和类概率𝑃(𝑌) 的乘积来求取)，典型方法为贝叶斯方法、隐马尔可夫链 

#### 线性回归

#### 自适应提升

- 霍夫丁不等式(Hoeffding’s inequality)：

  ![image-20190320050310653](/Users/miracle/Library/Application Support/typora-user-images/image-20190320050310653.png)

- 概率近似正确 (probably approximately correct, PAC)

  ![image-20190320050424940](/Users/miracle/Library/Application Support/typora-user-images/image-20190320050424940.png)

- Ada Boosting算法核心问题

  - 在每个弱分类器学习过程中，如何改变训练数据的权重:提高在上一轮中分类错误样本的权重。
  - 如何将一系列弱分类器组合成强分类器:通过加权多数表决方法来提高分类误差小的弱分类器的权重，让其在最终分类中起到更大作用。同时减少分类误差大的弱分类器的权重，让其在最终分类中仅起到较小作用。

- Ada Boosting:算法描述

  - 给定包含𝑁个标注数据的训练集合Γ，$Γ=\{ (𝑥_1,𝑦_1) ,…, (𝑥_𝑁,𝑦_𝑁) \}。𝑥_𝑖 (1≤𝑖≤𝑁) ∈𝑋⊆𝑅^𝑛,𝑦_𝑖 ∈𝑌=\{−1,1\}$

  - Ada Boosting算法将从这些标注数据出发，训练得到一系列弱分类器，并将这些弱分类器线性组合得到一个强分类器。

    1. 初始化每个训练样本的权重:$𝐷1=( 𝑤_{11},…,𝑤_{1𝑖},…,𝑤_{1𝑁}) $，其中$𝑤_{1𝑖}=\frac{1}{N}(1≤𝑖≤𝑁)$

    ![image-20190320051250165](/Users/miracle/Library/Application Support/typora-user-images/image-20190320051250165.png)

    ![image-20190320051324140](/Users/miracle/Library/Application Support/typora-user-images/image-20190320051324140.png)

- Ada Boosting:算法解释

  - 第m个弱分类器$G_m(x)$ 在训练数据集上产生的分类误差: 该误差为被错误分类的样本所具有权重的累加

    ![image-20190320051529143](/Users/miracle/Library/Application Support/typora-user-images/image-20190320051529143.png)

  - ![image-20190320051607417](/Users/miracle/Library/Application Support/typora-user-images/image-20190320051607417.png)

  - ![image-20190320051728749](/Users/miracle/Library/Application Support/typora-user-images/image-20190320051728749.png)
  - ![image-20190320051803920](/Users/miracle/Library/Application Support/typora-user-images/image-20190320051803920.png)
  - ![image-20190320051821626](/Users/miracle/Library/Application Support/typora-user-images/image-20190320051821626.png)
  - ![image-20190320051914617](/Users/miracle/Library/Application Support/typora-user-images/image-20190320051914617.png)



### 无监督学习

数据特征和相似度函数都很重要

#### K均值聚类	

#### 	![image-20190320052616105](/Users/miracle/Library/Application Support/typora-user-images/image-20190320052616105.png)		

1. 初始化聚类质心

   ![image-20190320052704132](/Users/miracle/Library/Application Support/typora-user-images/image-20190320052704132.png)

2. 将每个待聚类数据放入唯一一个聚类集合中

   ![image-20190320052803041](/Users/miracle/Library/Application Support/typora-user-images/image-20190320052803041.png)

3. 根据聚类结果、更新聚类质心

   ![image-20190320052846611](/Users/miracle/Library/Application Support/typora-user-images/image-20190320052846611.png)

4. 算法循环迭代，直到满足条件。聚类迭代满足如下任意一个条件，则聚类停止:

   - 已经达到了迭代次数上限

   - 前后两次迭代中，聚类质心基本保持不变

     

- K均值聚类算法的另一个视角: **最小化每个类簇的方差**

  ![image-20190320053059581](/Users/miracle/Library/Application Support/typora-user-images/image-20190320053059581.png)

  

- K均值聚类算法的不足

  - 需要事先确定聚类数目，很多时候我们并不知道数据应被聚类的数目

  - 需要初始化聚类质心，初始化聚类中心对聚类结果有较大的影响

  - 算法是迭代执行，时间开销非常大

  - 欧氏距离假设数据每个维度之间的重要性是一样的

    

#### 主成分分析

- 一种特征降维方法

- 降维后的结果要保持原始数据固有结构

  - 图像数据中结构: 视觉对象区域构成的空间分布
  - 文本数据中结构: 单词之间的(共现)相似或不相似

- 方差与协方差

  - 方差等于各个数据与样本均值之差的平方和之平均数，描述了样本数据的波动程度

  - 协方差衡量两个变量之间的相关度。对于一组两维变量，可通过计算它们之间的协方差值来判断这组数据给出的两维变量是否存在关联关系。

  - 当协方差$𝑐𝑜𝑣(𝑋,𝑌)>0​$时 , 称𝑋 与𝑌 正相关

    当协方差$𝑐𝑜𝑣(𝑋,𝑌)<0$时 , 称𝑋 与𝑌 负相关

    当协方差$𝑐𝑜𝑣(𝑋,𝑌)=0​$时 , 称𝑋 与𝑌 不相关

- 从协方差到相关系数

  - 通过皮尔逊相关系数(Pearson Correlation coefficient )将两组变量之间的关联度规整
    到一定的取值范围内。皮尔逊相关系数定义如下:

    ![image-20190320054523540](/Users/miracle/Library/Application Support/typora-user-images/image-20190320054523540.png)

  - ![image-20190320054634588](/Users/miracle/Library/Application Support/typora-user-images/image-20190320054634588.png)
  - ![image-20190320054803285](/Users/miracle/Library/Application Support/typora-user-images/image-20190320054803285.png)

- 主成份分析: 算法动机

  - 在数理统计中，方差被经常用来度量数据和其数学期望(即均值)之间偏离程度，这个偏离程 度反映了数据分布结构。 
  - 研究数据和其均值之间的偏离程度有着很重要的意义。 
  - 在降维之中，需要尽可能将数据向方差最大方向进行投影，使得数据所蕴含信息没有丢失，彰显个性。 
  - 主成份分析思想是将𝑛维特征数据映射到𝑙维空间( 𝑛 ≫ 𝑙)，去除原始数据之间的冗余性(通过去除相关性手段达到这一目的)
  - 将原始数据向这些数据方差最大的方向进行投影。一旦发现了方差最大的投影方向， 则继续寻找保持方差第二的方向且进行投影。 
  - 将每个数据从𝑛维高维空间映射到𝑙维低维空间，每个数据所得到最好的𝑘维特征就 是使得每一维上样本方差都尽可能大 。

  

- 主成份分析: 算法描述

  ![image-20190320055141238](/Users/miracle/Library/Application Support/typora-user-images/image-20190320055141238.png)

  ![image-20190320055521479](/Users/miracle/Library/Application Support/typora-user-images/image-20190320055521479.png)

  ![image-20190320055539576](/Users/miracle/Library/Application Support/typora-user-images/image-20190320055539576.png)

  ![image-20190320055548964](/Users/miracle/Library/Application Support/typora-user-images/image-20190320055548964.png)

  ![image-20190320055558687](/Users/miracle/Library/Application Support/typora-user-images/image-20190320055558687.png)

  

  

  

### 深度学习

#### 前馈神经网络

- 神经元是深度学习模型中基本单位

  - ![image-20190320060626354](/Users/miracle/Library/Application Support/typora-user-images/image-20190320060626354.png)

  - 常用激活函数

    ![image-20190320060715860](/Users/miracle/Library/Application Support/typora-user-images/image-20190320060715860.png)

- 前馈神经网络

  - 各个神经元接受前一级的输入，并输出到下一级，模型中没有反馈
  - 层与层之间通过“全连接”进行链接，即两个相邻层之间的神经元完全成对连接，但层内的神经元不相互连接。
  - 感知机网络(Perceptron Networks)是一种特殊的前馈神经网络:
    - 无隐藏层，只有输入层/输出层
    - 无法拟合复杂的数据
  -  $𝑤_{𝑖𝑗}(1 ≤ 𝑖 ≤ 𝑛,1 ≤ 𝑗 ≤ m)$构成了感知机模型参数，𝑛为神经网络层数、 𝑚为每层中神经元个数

- 如何优化网络参数

  ![image-20190320061713025](/Users/miracle/Library/Application Support/typora-user-images/image-20190320061713025.png)

- 梯度下降

  ![image-20190320062035491](/Users/miracle/Library/Application Support/typora-user-images/image-20190320062035491.png)

- 误差反向传播(error back propagation, BP)

  - BP算法是一种将输出层误差反向传播给隐藏层进行参数更新的方法。

  - 将误差从后向前传递，将误差分摊给各层所有单元，从而获得各层单元所产生的误差，进而依据这个误差来让各层单元负起各自责任、修正各单元参数。

  - ![image-20190320062933354](/Users/miracle/Library/Application Support/typora-user-images/image-20190320062933354.png)

    ![image-20190320063259491](/Users/miracle/Library/Application Support/typora-user-images/image-20190320063259491.png)

    ![image-20190320063412961](/Users/miracle/Library/Application Support/typora-user-images/image-20190320063412961.png)

    ![image-20190320063450143](/Users/miracle/Library/Application Support/typora-user-images/image-20190320063450143.png)

#### 卷积神经网络

- 图像经过特定卷积矩阵滤波后，所得到的卷积结果可认为是保留了像素点所构成的特定空间分布模式

- 学习一个卷积核，不同卷积核提取图像中不同视觉模式

- 在对原始图像做卷积操作后，可使用ReLu激活函数对卷积后结果进行处理

- **池化(pooling)**操作是对输入的特征图进行下采样，以获得最主要信息
- **全连接层与分类层**





#### 自然语言与视觉分析

- 学习单词的表达-----词向量(Word2Vec)

  - 在基于规则和统计的自然语言传统方法中，将单词视为独立符号
  - One-hot向量。
  - 缺点
    - 维数灾难的困扰
    - 无法刻画词与词之间的相似性:任意两个词之间都是孤立的
  - 通过深度学习方法，将单词表征为K维实数值向量(distribution representation)。这样，把对文本内容分析简化为 K 维向量空间中的向量运算，而向量空间上的相似度可以用来表示文本语义上的相似。用深度学习算法生成每个单词的向量表达所有单词的向量表达组成了一个“词向量空间” 。

- 词向量模型的训练

  ![image-20190320071634339](/Users/miracle/Library/Application Support/typora-user-images/image-20190320071634339.png)

- 词向量模型的基本思想

  - 词向量模型由一层输入层，一层隐藏层，一层输出层构成:实现了每个单词𝑁维向量的表达

- 词向量模型:两种训练模式

  - Continue Bag-of-Words (CBoW): 根据某个单词的上下文单词来预测该单词

  - Skip-gram:利用某个单词来分别预测该单词的上下文单词

    





















