# aclnnGroupedMatmulSwigluQuant

## 支持的产品型号

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>。

## 功能说明

- 接口功能：融合GroupedMatmul 、dquant、swiglu和quant，详细解释见计算公式。
- 计算公式：
  
  - **定义**：
    
    * **⋅** 表示矩阵乘法。
    * **⊙** 表示逐元素乘法。
    * $\left \lfloor x\right \rceil$ 表示将x四舍五入到最近的整数。
    * $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
    * $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
  - **输入**：
    
    * $X∈\mathbb{Z_8}^{M \times K}$：输入矩阵（左矩阵），M是总token 数，K是特征维度。
    * $W∈\mathbb{Z_8}^{E \times K \times N}$：分组权重矩阵（右矩阵），E是专家个数，K是特征维度，N是输出维度。
    * $bias∈\mathbb{Z_{32}}^{E  \times N}$：矩阵乘计算的偏移值，E是专家个数，N是输出维度。
    * $offset∈\mathbb{R}^{E  \times N}$：per-channel非对称反量化的偏移，E是专家个数，N是输出维度。
    * $w\_scale∈\mathbb{R}^{E \times N}$：分组权重矩阵（右矩阵）的逐通道缩放因子，E是专家个数，N是输出维度。
    * $x\_scale∈\mathbb{R}^{M}$：输入矩阵（左矩阵）的逐 token缩放因子，M是总token 数。
    * $grouplist∈\mathbb{N}^{E}$：前缀和的分组索引列表。
  - **输出**：
    
    * $Q∈\mathbb{Z_8}^{M \times N / 2}$：量化后的输出矩阵。
    * $Q\_scale∈\mathbb{R}^{M}$：量化缩放因子。
    * $Q\_offset∈\mathbb{R}^{M}$：量化偏移因子。
  - **计算过程**
    
    - 1.根据groupList[i]确定当前分组的 token ，$i \in [0,Len(groupList)]$。
    
      >例子：假设groupList=[3,4,4,6]，从0开始计数。
      >
      >第0个右矩阵`W[0,:,:]`，对应索引位置[0,3)的token`x[0:3]`（共3-0=3个token），对应`x_scale[0:3]`、`w_scale[0]`、`bias[0]`、`offset[0]`、`Q[0:3]`、`Q_scale[0:3]`、`Q_offset[0:3]`；
      >
      >第1个右矩阵`W[1,:,:]`，对应索引位置[3,4)的token`x[3:4]`（共4-3=1个token），对应`x_scale[3:4]`、`w_scale[1]`、`bias[1]`、`offset[1]`、`Q[3:4]`、`Q_scale[3:4]`、`Q_offset[3:4]`；
      >
      >第2个右矩阵`W[2,:,:]`，对应索引位置[4,4)的token`x[4:4]`（共4-4=0个token），对应`x_scale[4:4]`、`w_scale[2]`、`bias[2]`、`offset[2]`、`Q[4:4]`、`Q_scale[4:4]`、`Q_offset[4:4]`；
      >
      >第3个右矩阵`W[3,:,:]`，对应索引位置[4,6)的token`x[4:6]`（共6-4=2个token），对应`x_scale[4:6]`、`w_scale[3]`、`bias[3]`、`offset[3]`、`Q[4:6]`、`Q_scale[4:6]`、`Q_offset[4:6]`；
      >
      >请注意：grouplist中未指定的部分将不会参与更新。
      >例如groupList=[12,14,18]，X的shape为[30，:]。
      >
      >则第一个输出Q的shape为[30，:]，其中Q[18:，：]的部分不会进行更新和初始化，其中数据为显存空间申请时的原数据。
      >
      >同理，第二个输出Q的shape为[30]，其中Q\_scale[18:]的部分不会进行更新或初始化，其中数据为显存空间申请时的原数据。
      >
      >即输出的Q[:grouplist[-1],:]和Q\_scale[:grouplist[-1]]为有效数据部分。

    - 2.根据分组确定的入参进行如下计算：

      $C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ BroadCast} \odot w\_scale_{i\ BroadCast}$

      $C_{i,act}, gate_{i} = split(C_{i})$

      $S_{i}=Swish(C_{i,act})\odot gate_{i}$  &nbsp;&nbsp;其中$Swish(x)=\frac{x}{1+e^{-x}}$

      >注：当前版本不支持$bias_{i}$、$offset_{i}$，未来版本将支持的计算公式如下：
      >$C_{i} =(X_{i}\cdot W_{i} + bias_{i\ BroadCast})\odot x\_scale_{i\ BroadCast} \odot w\_scale_{i\ BroadCast}+offset_{i\ BroadCast}$

    - 3.确定量化方式
      
      - 当量化方式为对称量化时：

        $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

        $Q_{i} = \left \lfloor \frac{S_{i}}{Q\_scale_{i}}\right \rceil $

      - 当量化方式为非对称量化时：(暂不支持)

        $Q\_scale_{i} = \frac{max(S_{i})-min(S_{i})}{255}$

        $Q\_offset_{i} = -128 - \left \lfloor \frac{min(S_{i})}{Q\_scale_{i}}\right \rceil$

        $Q_{i} = \left \lfloor \frac{S_{i}}{ Q\_scale_{i} } + Q\_offset_{i}\right \rceil $

## 函数原型

每个算子分为[两段式接口](./common/两段式接口.md)，必须先调用“aclnnGroupedMatmulSwigluQuantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGroupedMatmulSwigluQuant”接口执行计算。

- `aclnnStatus aclnnGroupedMatmulSwigluQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *bias, const aclTensor *offset,  const aclTensor *weightScale, const aclTensor *xScale, const aclTensor *groupList,  aclTensor *output, aclTensor *outputScale, aclTensor *outputOffset, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnGroupedMatmulSwigluQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnGroupedMatmulSwigluQuantGetWorkspaceSize

- **参数说明：**
  
  - x（aclTensor*，计算输入）：左矩阵，公式中的$X$，Device侧的aclTensor。shape支持2维，假设shape为[M,K]，则K必须小于65536，数据类型支持INT8，[数据格式](./common/数据格式.md)支持ND，支持[非连续的Tensor](common/非连续的Tensor.md)。

  - weight（aclTensor*，计算输入）：权重矩阵，公式中的$W$，Device侧的aclTensor。shape支持3维，数据类型支持INT8，[数据格式](./common/数据格式.md)支持FRACTAL\_NZ，支持[非连续的Tensor](common/非连续的Tensor.md)。

  - bias（aclTensor*，计算输入）：矩阵乘计算的偏移值，公式中的$bias$，shape支持2维，数据类型支持INT32，预留输入，暂不支持，需要传空指针。

  - offset（aclTensor*，计算输入）：per-channel非对称反量化的偏移，公式中的$offset$，shape支持2维，数据类型支持Float，预留输入，暂不支持，需要传空指针。

  - weightScale（aclTensor*，计算输入）：右矩阵的量化因子，公式中的$w\_scale$，Device侧的aclTensor。shape支持2维，首轴长度需与`weight`的首轴维度相等，尾轴长度需要与weight还原为ND格式的尾轴相同，数据类型支持FLOAT、FLOAT16、BFLOAT16，[数据格式](./common/数据格式.md)支持ND，支持[非连续的Tensor](common/非连续的Tensor.md)。

  - xScale（aclTensor*，计算输入）：左矩阵的量化因子，公式中的$x\_scale$，Device侧的aclTensor。shape支持1维，长度需与`x`的首轴维度相等，数据类型支持FLOAT，[数据格式](./common/数据格式.md)支持ND，支持[非连续的Tensor](common/非连续的Tensor.md)。

  - groupList（aclTensor*，计算输入）：指示每个分组参与计算的Token个数，公式中的$grouplist$，Device侧的aclTensor。shape支持1维，长度需与`weight`的首轴维度相等，数据类型支持INT64，[数据格式](./common/数据格式.md)支持ND，支持[非连续的Tensor](common/非连续的Tensor.md)，grouplist中的最后一个值约束了输出数据的有效部分，详见功能说明中的计算过程部分。

  - output（aclTensor*，计算输出）：输出的量化结果，公式中的$Q$，Device侧的aclTensor。数据类型支持INT8，shape支持2维，Device侧的aclTensor。[数据格式](./common/数据格式.md)支持ND，支持[非连续的Tensor](common/非连续的Tensor.md)。

  - outputScale（aclTensor*，计算输出）：输出的量化因子，公式中的$Q\_scale$，Device侧的aclTensor。数据类型支持FLOAT，shape支持1维，Device侧的aclTensor。[数据格式](./common/数据格式.md)支持ND，支持[非连续的Tensor](common/非连续的Tensor.md)。

  - outputOffset（aclTensor*，计算输出）：输出的非对称量化的偏移，公式中的$Q\_offset$，Device侧的aclTensor，shape支持1维，数据类型支持FLOAT，预留输入，暂不支持，需要传空指针。

  - workspaceSize（uint64_t*，出参）：返回用户需要在npu device侧申请的workspace大小。

  - executor（aclOpExecutor**，计算输出）：返回op执行器，包含了算子计算流程。
- **返回值：**
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](./common/aclnn返回码.md)。
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）： 1. 传入的 x、weight、weightScale、xScale、groupList、output、outputScale是空指针时。
  返回161002（ACLNN_ERR_PARAM_INVALID）： 1. 传入的x、weight、weightScale、xScale、groupList、output、outputScale的数据维度不满足约束。
                                         2. 传入的x、weight、weightScale、xScale、groupList、output、outputScale数据的shape不满足约束条件。
                                         3. 传入的x、weight、weightScale、xScale、groupList、output、outputScale数据的format不满足约束条件。
                                         4. groupList的元素个数大于weight的首轴长度。
                                         5. output的尾轴长度超过4096。
                                         6. x的尾轴长度大于等于65536。
  ```

## aclnnGroupedMatmulSwigluQuant

- **参数说明：**

    - workspace（void*，入参）：在Device侧申请的workspace内存地址。
    - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnGroupedMatmulSwigluQuantGetWorkspaceSize获取。
    - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
    - stream（aclrtStream，入参）：执行任务的AscendCL Stream流。
- **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](./common/aclnn返回码.md)。

## 约束说明

- 仅支持内部使用，支持特定shape

## 调用示例

- aclnn单算子调用方式

通过aclnn单算子调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_swiglu_quant.h"

#define CHECK_RET(cond, return_expr) \
do {                               \
  if (!(cond)) {                   \
    return_expr;                   \
  }                                \
} while (0)

#define LOG_PRINT(message, ...)     \
do {                              \
  printf(message, ##__VA_ARGS__); \
} while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    // 固定写法，AscendCL初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, 
                    void** deviceAddr, aclDataType dataType, aclFormat formatType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, formatType,
                            shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t E = 4;
    int64_t M = 8192;
    int64_t N = 4096;
    int64_t K = 7168;
    std::vector<int64_t> xShape = {M, K};
    std::vector<int64_t> weightShape = {E, N / 32 ,K / 16, 16, 32};
    std::vector<int64_t> weightScaleShape = {E, N};
    std::vector<int64_t> xScaleShape = {M};
    std::vector<int64_t> groupListShape = {E};
    std::vector<int64_t> outputShape = {M, N / 2};
    std::vector<int64_t> outputScaleShape = {M};

    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* weightScaleDeviceAddr = nullptr;
    void* xScaleDeviceAddr = nullptr;
    void* groupListDeviceAddr = nullptr;
    void* outputDeviceAddr = nullptr;
    void* outputScaleDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* weightScale = nullptr;
    aclTensor* xScale = nullptr;
    aclTensor* groupList = nullptr;
    aclTensor* output = nullptr;
    aclTensor* outputScale = nullptr;

    std::vector<int8_t> xHostData(M * K, 0);
    std::vector<int8_t> weightHostData(E * N * K, 0);
    std::vector<float> weightScaleHostData(E * N, 0);
    std::vector<float> xScaleHostData(M, 0);
    std::vector<int64_t> groupListHostData(E, 0);
    std::vector<int8_t> outputHostData(M * N / 2, 0);
    std::vector<float> outputScaleHostData(M, 0);

    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr,  aclDataType::ACL_INT8, aclFormat::ACL_FORMAT_ND, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weight aclTensor
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr,  aclDataType::ACL_INT8, aclFormat::ACL_FORMAT_FRACTAL_NZ, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weightScale aclTensor
    ret = CreateAclTensor(weightScaleHostData, weightScaleShape, &weightScaleDeviceAddr, aclDataType::ACL_FLOAT,  aclFormat::ACL_FORMAT_ND, &weightScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建xScale aclTensor
    ret = CreateAclTensor(xScaleHostData, xScaleShape, &xScaleDeviceAddr, aclDataType::ACL_FLOAT,  aclFormat::ACL_FORMAT_ND, &xScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建groupList aclTensor
    ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, aclFormat::ACL_FORMAT_ND, &groupList);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建output aclTensor
    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_INT8, aclFormat::ACL_FORMAT_ND, &output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建outputScale aclTensor
    ret = CreateAclTensor(outputScaleHostData, outputScaleShape, &outputScaleDeviceAddr, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, &outputScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 3. 调用CANN算子库API
    // 调用aclnnGroupedMatmulSwigluQuant第一段接口
    ret = aclnnGroupedMatmulSwigluQuantGetWorkspaceSize(x, weight, nullptr, nullptr, weightScale, xScale, 
                                                        groupList, output, outputScale, nullptr,
                                                        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, 
    LOG_PRINT("aclnnGroupedMatmulSwigluQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnGroupedMatmulSwigluQuant第二段接口
    ret = aclnnGroupedMatmulSwigluQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, 
    LOG_PRINT("aclnnGroupedMatmulSwigluQuant failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outputShape);
    std::vector<int8_t> out1Data(size, 0);
    ret = aclrtMemcpy(out1Data.data(), out1Data.size() * sizeof(out1Data[0]), outputDeviceAddr,
                        size * sizeof(out1Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t j = 0; j < size; j++) {
        LOG_PRINT("result[%d] is: %d\n", j, out1Data[j]);
    }
    size = GetShapeSize(outputScaleShape);
    std::vector<float> out2Data(size, 0);
    ret = aclrtMemcpy(out2Data.data(), out2Data.size() * sizeof(out2Data[0]), outputScaleDeviceAddr,
                        size * sizeof(out2Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t j = 0; j < size; j++) {
        LOG_PRINT("result[%d] is: %f\n", j, out2Data[j]);
    }
    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(weightScale);
    aclDestroyTensor(xScale);
    aclDestroyTensor(groupList);
    aclDestroyTensor(output);
    aclDestroyTensor(outputScale);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(weightScaleDeviceAddr);
    aclrtFree(xScaleDeviceAddr);
    aclrtFree(groupListDeviceAddr);
    aclrtFree(outputDeviceAddr);
    aclrtFree(outputScaleDeviceAddr);
    if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```