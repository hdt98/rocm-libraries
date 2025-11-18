struct WinogradConv
{
    using InpDataType = half_t;
    using WeiDataType = half_t;
    using OutDataType = half_t;

    struct Args
    {
        WinogradConv::InpDataType* inp;
        WinogradConv::WeiDataType* wei;
        WinogradConv::OutDataType* out;

        index_t N;
        index_t C;
        index_t H;
        index_t W;

        index_t K;
        index_t n_groups;
        index_t R;
        index_t S;

        index_t pad_H;
        index_t pad_W;
        index_t out_H;
        index_t out_W;

        index_t d_N_stride;
        index_t d_C_stride;
        index_t d_H_stride;

        index_t f_K_stride;
        index_t f_C_stride;
        index_t f_R_stride;

        index_t o_N_stride;
        index_t o_K_stride;
        index_t o_H_stride;
    };

    __host__ static auto GetGridSize(Args args) { return dim3(1); }
    __host__ static auto GetBlockSize(Args args) { return dim3((256 + 128) * 2); }

    __device__ void run(const Args args)
    {
        if(THREAD_IDX_AT(0, 0, 0, 0))
        {
            printf("this is a kernel template\n");
        }
    }
};