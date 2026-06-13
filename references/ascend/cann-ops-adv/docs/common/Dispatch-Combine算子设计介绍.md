# MoeDistributeDispatch和MoeDistributeCombine算子设计介绍

说明：本文档图片正在准备中

# 一、算子背景介绍
## MoE架构带来的通信优化挑战

在大模型训练和推理领域，MoE架构凭借其动态专家激活机制带来的计算稀疏性优势，以及千亿参数规模下的高推理吞吐能力，已然成为超大规模模型的核心技术方案。该架构通过分发（Dispatch）和组合（Combine）这两个关键操作，实现了输入数据的动态分配与多专家输出的高效整合，从而，在维持海量参数规模的同时确保了高性能计算。然而，随着专家并行（EP）规模的不断扩大，专家间频繁交互带来的高额通信开销，逐渐演变成为制约大模型推理性能的关键瓶颈。

## 创新通算融合算子破局MoE通信瓶颈

针对MoE架构，我们的技术团队深入分析后发现，高额的通信开销主要源于两大技术挑战：

(1) AllToAllV通信的缺陷

动态专家选择机制下，每个Token分发的目标专家呈现离散分布特征，导致两个关键问题：

* 数据分发不均匀：不同专家接收的Token长度存在差异，需依赖低效的AllToAllV通信；
* 元数据同步开销：获取收发信息需调用前置AllGather算子收集路由表，并在Host侧完成同步，引入额外通信开销和Stream同步延迟；

(2) 小数据包与Host Bound困境

推理场景中的Token数据量小，引发双重挑战：

* 算子下发延迟：传统Host驱动通信需要构造子图并调度，其下发时延随EP规模线性增长；
* RDMA同步开销：RDMA前同步和后同步引入额外的RTT时延。

分析这些瓶颈点后，我们创新性地开发了MoeDistributeDispatch和MoeDistributeCombine两个融合算子，并历经三个阶段的迭代优化，最终实现模型推理吞吐提升50%的显著收益。

## Dispatch/Combine双算子协同优化

DeepSeekV3模型的MoE架构创新性地采用了动态路由机制，每个Token会动态选择topK个专家进行处理。在这一架构中，Dispatch操作发挥核心调度作用，基于Token与专家的路由对应关系表，采用分布式计算策略：首先将各专家节点需处理的Token数量计算任务下沉到对应设备执行，随后通过AllToAllV通信完成Token的跨设备传输，同时预计算Combine阶段所需参数；而Combine操作则扮演着整合者的角色，负责对各专家输出的计算结果执行加权求和，并通过逆向的AllToAllV通信将处理后的Token数据恢复至原始位置，从而完成整个分布式专家计算的协同与整合。

由上可知，Dispatch/Combine操作本质上是计算与通信的结合。因此，我们创新型地实现了MoeDistributeDispatch和MoeDistributeCombine这两个通算融合算子，将路由计算等Host侧逻辑下沉至Device侧，消除了Host与Device间的同步开销。同时，将Combine操作中所需的部分计算操作与AllToAllV通信流水并行，也得以实现计算与通信的耗时掩盖。

## 基于AIV+AICPU融合架构的RDMA全互联（Fullmesh）方案

AIV作为昇腾硬件NPU主要的计算单元，负责完成数据的预处理、RDMA通信元数据的准备、数据接收Flag的轮询以及接收数据的后处理等关键环节。

在预处理阶段，首先会获取每个Token的路由信息，依照专家索引对Token进行重排，将发往同一个目标rank的数据汇聚。这样，仅一次通信就可以完成目标rank上所有专家的数据发送，以减少RDMA下发时延。

与此同时，AIV将数据在共享内存中的地址、数据长度信息通过GM直接传递给同处Device侧上的AICPU，由AICPU直接驱动RDMA通信，彻底摒弃了传统需要Host侧构造子图和调度RDMA任务的繁琐流程，不仅解决了Host侧处理耗时长的问题，更消除了传统调度方式带来的额外时延。在通信环节，AIV轮询数据接收Flag，以确保所有rank的Token数据全部接收完成，从而消除RDMA同步带来的通信时延。最后，AIV将共享内存中的数据按照专家汇总搬出，为后续FFN层的计算提供数据准备。这一系列优化形成了完整的低延迟处理闭环。

# 二、MoeDistributeDispatch实现方案
### 索引计算与Token重排
#### 特性背景
索引计算主要是围绕expertIds这个输入展开，该输入是一个BS\*K的矩阵，其中expertIds(i,j)表示第i个专家被放松给的第j个专家的索引。首先，Combine算子需要Dispatch算子提供一个expandIdx的输出，该输入也是一个BS\*K的矩阵，其中expandIdx(i,j)的含义是全局来看第i个token是发送给专家expertIds(i,j)的第几个token，下图给出了一个BS=4，K=4，moeExpertNum=8的示例。  
Token重排主要是因为每次通信下发均有时间开销（1us），为了尽可能减少下发次数，我们希望将需要发送的数据一次性发送给对应rank，为了做到这一点，我们需要重排使得发送给每个rank的Token在HBM上连续，本质上就是要做一个MoePermute的操作。下图给出了一个BS=4，K=4，moeExpertNum=8，localExpertNum=2的示例。

#### 特性实现
为了计算expertIds，同时也考虑到后面发送信息的需要，我们构建矩阵sendStatus矩阵，该矩阵是一个worldSize*STATUS_ENTRY_COUNT的矩阵，其中STATUS_ENTRY_COUNT为常量，当前定义为32。sendStatus矩阵的布局如下  
sendStatus矩阵的每一行都分为count区和flag区，前FLAG_OFFSET个数为count区，每个数表示发送的Token数量，即sendStatus(i,j)表示本卡往第i张卡的第j个专家发送的Token数量，其中j < localExpertNum。由以上布局可以看到，localExpertNum需要不大于FLAG_OFFSET，FLAG_OFFSET当前定义为常量，值为24。第FLAG_OFFSET个数（从0计数）为flag，是一个特殊值，该值用于后续接受同步，此处暂不做介绍。  

具体在计算时，我们对expertIds做循环，循环到第i行第j列时，expertIds(i,j)即为要发送的专家索引，Ceil(expertIds(i,j)/localExpertNum)即为要发送的rank索引，expertIds(i,j)%localExpertNum即为要发送的专家在其所在rank上的排序，那么sendStatus(expertIds(i,j)/localExpertNum, expertIds(i,j)%localExpertNum)即为之前已经往该专家发送的Token数量，所以expandIdx(i,j)=sendStatus(expertIds(i,j)/localExpertNum, expertIds(i,j)%localExpertNum)，同时需要更新sendStatus(Ceil(expertIds(i,j)/localExpertNum), expertIds(i,j)%localExpertNum)++。  
为了实现token重排，我们还需要对给所有专家发送Token数量做一个前缀和数组expertCumsum，例如4.1.1节的例子计算的到的前缀和数组如下  

有了这个数组，我们就可以进行Token重排了。还是对expertIds做循环，循环到第i行第j列时，我们需要明确这个Token副本要重排到什么位置。为了确定这个位置，需要知道三个值，第一个是要发送给哪个rank，这个可以通过Ceil(expertIds(i,j)/localExpertNum)获得，第二个是发送给这张卡上排序在这个Token要发送的专家之前的所有专家的Token数量之和，这个可以通过计算expertCumsum(expertIds(i,j))-expertCumsum(Ceil(expertIds(i,j)/localExpertNum)*localExpertNum)获得，第三个是这个Token副本在要所有要发送给对应专家的Token里排第几个，这个可以通过expandIdx(i,j)获得。由以上三个值，即可获得重排后的位置。  
### 发送数据与接受同步
#### 特性背景
在前期准备工作完成后，就要进行数据的发送了。本算子使用BatchWrite接口发送数据，该接口的入参是一个GM指针，该指针指向一个结构体数组，数组内有多少个结构体，就会通信下发多少次。单个结构体包含的字段如下：  
这里HCCLBUFFER指由HCCL管理的两块GM，分别叫做WindowsIn和WindowsOut，其中WindowsIn用于接受，WindowsOut用于发送。
BatchWrite没有同步机制，且每次下发有时间开销（1us），所以我们需要做到：  
* 尽可能少的下发次数
* 接收端在算子侧实现同步，保证数据接受齐备后再进行后续处理
#### 特性实现
我们将WindowsIn和WindowsOut都平均分成worldSize个窗口，每个窗口存放发送给对应rank的所有数据，这些数据连续，这样对于每个rank只需要一次下发即可。每个窗口内的数据结构如下：  
第i个窗口的发送Token数量数组加上FLAG 1即为上面提到的sendStatus的第i行，长度固定，发送的Tokens即为上一节重排的结果，FLAG 2需要单独写入，其位置与Tokens数量有关，不固定。按照下图来进行发送：  
在接受数据时，我们采取分核双循环等待，将所有rank平均分配给各个核，每个核依次处理若干个对应的rank，对于每个rank，采取双循环等待：  
while循环反复读取FLAG 1的值，直到被刷新为特定值，说明前面的Token数量数组已经接受齐备；  
对Token数量数组求和获取总的Tokens数量，得到FLAG2的位置；  
while循环反复读取FLAG 1的值，直到被刷新为特定值，说明所有数据都已收齐。  
### 发送后处理
#### 特性背景
dispatch算子在收齐信息后，除了需要依据元数据将实际数据内容进行重排，使得属于同一个专家的token在GM上保序连续，同时还需要依据元数据计算epRecvCount和expertTokenNum这两个输出，该过程较为复杂，本章进行详细说明。具体来说，假设总共有w张卡，每张卡上放e个专家，收到的各rank发来的元数据按顺序搬运到ub上（不搬运flag）以后可以形成如下矩阵：
直观上来说，epRecvCount这个输出需要将上面这个矩阵先进行转置，再按行主序进行前缀和。如下图所示：
其中红色箭头为前缀和的方向，实际即为将转置后的矩阵按行主序展开成一维后进行前缀和。为更形象地说明，举个具体的小规模case（3卡，单卡4专家）：
expertTokenNum是本地每个专家的token数量前缀和数组，其实就是epRecvCount的最后一列，此处不再赘述。
#### 特性实现
本特性主要需要考虑的地方在于如何尽可能高效实现转置+累加和的操作，避免过多的scalar操作影响性能。此处我们选择Add+GatherMask+Adds接口来实现这一功能，GatherMask这个接口的功能相对复杂，详见GatherMask-CANN商用版8.1.RC1-昇腾社区
流程框架如下：
首先使用Add接口对转置前的矩阵从0行开始，按顺序将上一行的数据对应加到下一行上，都是向量操作。还是以上面的小规模case为例，这一步作用是转置后就不用再在列方向上做前缀和了，效果如下：
使用GatherMask接口，一次性转置一列数据：
使用Adds接口，从0行开始，按顺序将上一行的最后一个数加到下一行上：
最终即得到需要得epRecvCount输出。
### 算子间协同
#### 特性背景
 
图1：时序示意图  
如上面的时序图所示，由于不同卡的数据量可能存在不平衡的情况，快卡（rank b）在执行到combine的时候，慢卡（rank a）dispatch的后处理部分还没有结束，此时并没有同步接口可以用来阻塞快卡的combine发送数据，那么就可能会写脏慢卡还没有来得及后处理的数据，造成精度问题或者是卡死。  
#### 特性实现  
核心思路是把WindowIn（接受数据的Buffer）和WindowOut（发送数据的Buffer）都平均分成两块，并在WindowIn的特定位置（第一块的最后1M起始处）设置一个名为bufferChosen的flag（如图2所示）。在dispatch和combine算子初始化的时候，都会去这个位置读取这个值，如果是0，则使用WindowIn和WindowOut的第一块，如果是1，则使用第二块。在算子执行结束前，刷新bufferChosen=bufferChosen^1。  
图2：双Buffer示意图  
此处可能会有疑问：为什么只把数据空间划分为两块就可以解决问题，而不需要更多？同样参考图1，可以看到由于快慢卡的原因，不同卡combine算子的启动时间无法做到同步，但是combine算子发完后有一轮等待同步WaitCombine，可以保证快卡的combine算子WaitCombine结束时，慢卡的combine算子一定发送结束了，这也就意味着，快卡的第二次dispatch启动时，慢卡的第一次dispatch一定已经结束了，那么第一次dispatch使用的数据空间就可以被写入了。  


# 三、MoeDistributeCombine实现方案

实现方案中的示例如下，卡数为4，moe专家数为8，每张卡在dispatch阶段输入5个token，每个token发给2个专家  

图1 专家分布图  

图2 Dispatch 输入Token和专家索引表  

图3 Combine expandX、expandIdx、sendCounts输入表  


### 1.Token重排（ReorderToken）

#### 特性背景

Combine算子在Fullmesh方案中使用HCCL高阶API中的BatchWrite接口进行卡间通信，该接口每次下发通信任务的时间开销均有时间开销（1us），为了尽可能减少下发次数，需要通过重排Token使得发送给每个卡的Token在GM上连续，从而能够让要发送的数据一次性发送给目标卡，同时，由于数据需要搬运到通信域的缓冲区才能发送，因此，在Token重排阶段，需要按照Dispatch传递给Combine的sendCounts，确定Token需要发送的目标卡，并将Token追加到目标卡在WinOut对应区域的尾部。WinIn和WinOut指由HCCL管理的两块GM，统称为HCCLBUFF，其中WindowsIn用于接受，WindowsOut用于发送。  

#### 特性实现

以卡0为例，输入的expandX和sendCounts如下图  

sendCounts的shape为(moeExperNum,0)，其中的内容可以理解为（单卡专家数，卡数）形状的二维矩阵，在非前缀和的形式下，第[i, j]个元素表示当前卡第i个专家要发给第j张卡的Token数。  
以示例中卡0发给卡1的数据搬运到WinOut的过程为例，要发给卡1的专家0和专家1的Token个数为sendCounts的[0,1]和[1,1]位置的元素的值（非前缀和），即0和5。前缀和形式下前一个值分别为1和3，说明需要从expandX的第1个Token开始取0个Token、从expandX的第3个Token开始取5个Token，分别追加到WinOut分配给卡1的区域尾部。  

伪代码如下

```
for dstRankId from 0 to worldSize:
    // 设置当前目标Rank的输出窗口地址
    localOutWindow.SetGlobalBuffer(windowOutGM + dstRankId * rankSizeOnWin)
    rankTokenNum = 0 // 表示目标卡当前Token的计数

    for expertId from 0 to localMoeExpertNum:
        // 计算前序token数量
        if expertId != 0 or dstRankId != 0:
            preCount = sendCountLocal[expertId * worldSize_ + dstRankId - 1]
        else:
            preCount = 0

        startTokenAddr = preCount * axisH
        tokenNum = sendCountLocal[expertId * worldSize + dstRankId] - preCount

        for tokenId from 0 to tokenNum:
            DataCopy(ubTensor, expandXGlobal + startTokenAddr)  // 从输入GM搬入UB
            DataCopy(localOutWindow + rankTokenNum * axisH, ubTensor)  // 从UB追加到目标卡的尾部
            startTokenAddr += axisH
            rankTokenNum += 1
```

### 2. 发送数据和接收同步（AllToAllV+WaitFlag）

#### 特性背景

在Token重排完成后，便进行数据的发送。本算子使用BatchWrite接口发送数据，该接口的入参是一个GM指针，该指针指向一个结构体数组，数组内有多少个结构体，就会通信下发多少次。单个结构体包含内存排布和含义如下  

```text
UINT64 该次发送数据在HCCLBUFFER上的首地址
UINT64 接收端的HCCLBUFFER首地址
UINT64 发送数据大小
UINT32 发送数据类型
UINT32 发送RANK号
```

BatchWrite没有同步机制，每个结构体都表示一个通信任务，BatchWrite每下发一个通信任务到RDMA的时间开销约为1us，所以我们需要做到：

* 接收端在算子侧实现同步，保证数据接受齐备后再进行后续处理
* 尽可能减少通信任务下发次数，即结构体数量
  因此，减少下发通信任务次数由前述的Token重排实现，算子侧实现同步则通过在WinOut中要发给目标卡的数据尾部加一个Flag，发送数据时将Data和Flag一起发送，接收端轮询这个Flag实现。
  由于BatchWrite只能在0核执行，仅起到通知aicpu可以开始工作的功能，aicpu在收到该消息后会开始解析并下发通信任务

#### 特性实现

我们将WindowsIn和WindowsOut都平均分成worldSize个窗口，每个窗口存放发送给对应rank的所有数据，这些数据连续，这样对于每个rank只需要一次下发即可。  
以卡0为例，发送端WinOut和接收端WinIn的数据最终如下



在接受数据时，我们采取分核循环等待，将所有rank平均分配给各个核，每个核依次处理若干个对应的rank，对于每个rank，轮询FLAG，直到被刷新为特定值，说明数据已收到。

### 3. 加权求和（Sum）

#### 特性背景

在Dispatch算子的过程中，每个Token都被发给K个专家，因此，Combine原路返还Token到WinIn的数据实际上是Dispatch输入Token个数的K倍。Combine的最后一步就是将K个Token加权求和为原来的1个Token，并按照原来Dispatch Token输入的顺序进行输出。

#### 特性实现

Token在WinIn中的排布不是按Dispatch输入的Token顺序而连续排布的。需要通过expertIds专家索引表和Dispatch给Combine的输入expandIdx计算出Token的位置。expandIdx表示了对应专家索引表同位置的Token是该专家收到的第几个Token，以下图中token2，其位置在专家2的第2个Token（从0开始计数，下同）和专家3的第1个Token，这个符合Dispatch输入的Token2发送的专家索引是2和3且expandId对应位置是2和1的信息。


Dispatch输入Token、专家索引表和expandIdx示例图，


因此，只要知道每个专家相对于WinIn首地址的偏移量，即可计算出Token在WinIn的实际位置，计算专家相对于WinIn首地址的偏移量的方法如下。
1、根据expertIds统计往每个专家发送的Token个数  
2、依据每个专家在目标卡是第几个专家，求出该专家前面的专家的累计Token数。例如，专家2和专家3是卡1是第0个和第1个专家，专家2前面没有专家，则累计Token数为0，专家3前面有专家2，对应的累计Token数为3，通过该计算步骤得到每个专家相对于该专家归属的卡在WinIn的首地址的Token数偏移量，存储于数组expertWindowOffset中。  
3、依据上述步骤计算得到每个专家之前的累计Token数后，乘以Token的字节数H，即可得到相对于该专家归属的卡在WinIn的首地址的字节数偏移量  

由此得到Token在WinIn的地址计算公式如下，TokenAddr(i, j)表示Dispatch输入的第i个Token，发送给K个专家中的第j个专家的数据，在Combine返还回来后，在WinIn的地址  
winInGM表示winIn首地址，expertId表示发送给K个专家中的第j个专家的专家编号，rankSizeOnWin表示winIn均分给worldSize个卡后的单卡大小，rank表示专家expertId归属的卡号  

$$
TokenAddr(i, j) = windowInGM + rankSizeOnWin * rank + expertWindowOffset[expertId] * H + expandIdx[i, j] * H
$$

按照以上地址计算公式，即可按专家索引表遍历所有Token数据，依据输入的每个专家的权重系数，将K个Token加权求和为1个Token，还原回Dispatch的输入格式  
