#pragma region print_util

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

template <typename... Ts>
struct [[deprecated]] pntS
{
};

template <typename T>
void PrintTensorHelper(T& tensor, std::vector<std::size_t>& iss)
{
    auto nDim = tensor.GetNumOfDimension();
    auto lens = tensor.GetLengths();
    auto len  = iss.size();
    if(len < nDim)
    {
        if(len == 0)
            printf("\nStart tensor print:\n");

        if(len == nDim - 1)
        {
            printf("[");
            for(auto idx : iss)
                printf("%2zu ", idx);
            printf(" 0-%zu ] ", lens[len] - 1);
        }

        for(size_t i = 0; i < lens[len]; i++)
        {
            iss.push_back(i);
            PrintTensorHelper(tensor, iss);
            iss.pop_back();
        }
        if(len == nDim - 1)
        {
            printf("\n");
        }
    }
    else
    {
        printf("%10.2f \t", (float)tensor.mData[tensor.GetOffsetFromMultiIndex(iss)]);
    }
}

template <typename T>
void PrintTensor(T tensor)
{
    std::vector<std::size_t> iss{};
    PrintTensorHelper(tensor, iss);
}

#pragma region debug_buf

__device__ constexpr int debug_line_size = DEBUG_LINE_SIZE;
constexpr int debug_buf_size             = DEBUG_MAX_ENTRY * DEBUG_LINE_SIZE;

__device__ DEBUG_DATA_TYPE dev_debug_buf[debug_buf_size];

__device__ int debug_entry               = 0;
__device__ constexpr int debug_max_entry = DEBUG_MAX_ENTRY;

void PrintDebugBuf()
{
    DEBUG_DATA_TYPE host_debug_buf[debug_buf_size];
    HIP_CHECK_ERROR(hipMemcpyFromSymbol(
        &host_debug_buf, HIP_SYMBOL(dev_debug_buf), sizeof(DEBUG_DATA_TYPE) * debug_buf_size));

    HostTensorDescriptor debug_buf_desc{DEBUG_MAX_ENTRY, DEBUG_LINE_SIZE};
    Tensor<DEBUG_DATA_TYPE> debug_buf(debug_buf_desc);

    for(int i = 0; i < debug_buf_size; i++)
        debug_buf.mData[i] = host_debug_buf[i];

    PrintTensor(debug_buf);
}

#pragma region thread_range

__device__ inline bool THREAD_IDX_AT(ck::index_t a = 0,
                                     ck::index_t b = 0,
                                     ck::index_t c = 0,
                                     ck::index_t x = 0,
                                     ck::index_t y = 0,
                                     ck::index_t z = 0)
{
    return blockIdx.x == a && blockIdx.y == b && blockIdx.z == c && threadIdx.x == x &&
           threadIdx.y == y && threadIdx.z == z;
}

__device__ inline bool THREAD_IDX_UB(ck::index_t a = 0,
                                     ck::index_t b = 0,
                                     ck::index_t c = 0,
                                     ck::index_t x = 0,
                                     ck::index_t y = 0,
                                     ck::index_t z = 0)
{
    return blockIdx.x <= a && blockIdx.y <= b && blockIdx.z <= c && threadIdx.x <= x &&
           threadIdx.y <= y && threadIdx.z <= z;
}

__device__ inline bool THREAD_IDX_LB(ck::index_t a = 0,
                                     ck::index_t b = 0,
                                     ck::index_t c = 0,
                                     ck::index_t x = 0,
                                     ck::index_t y = 0,
                                     ck::index_t z = 0)
{
    return blockIdx.x >= a && blockIdx.y >= b && blockIdx.z >= c && threadIdx.x >= x &&
           threadIdx.y >= y && threadIdx.z >= z;
}