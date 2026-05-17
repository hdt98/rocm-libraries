#include "naive_conv.hpp"

template <bool ASSUME_PACKED,
          typename src_data_t,
          typename acc_data_t,
          typename dst_data_t,
          int use_tf32 = 0>
inline __device__ void naive_conv_bwd_nchw(dst_data_t* __restrict__ p_in,
                                           const src_data_t* __restrict__ p_wei,
                                           const double alpha,
                                           const double beta,
                                           const src_data_t* __restrict__ p_out,
                                           Strides5D in_strides,
                                           Strides5D wei_strides,
                                           Strides5D out_strides,
                                           int hi,
                                           int wi,
                                           int n,
                                           int k_per_group,
                                           int c_per_group,
                                           int ho,
                                           int wo,
                                           int sy,
                                           int sx,
                                           int dy,
                                           int dx,
                                           int py,
                                           int px,
                                           int fy,
                                           int fx,
                                           int group)
{
    /*
     *  need to compute total input pixel: `group * n * c_per_group * hi * wi`.
     *  to distribute this workload, let one workgroup compute `hi * wi` pixel,
     *  hence need `group * n * c_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = hi * wi;
    int bid           = blockIdx.x;
    int ic            = bid % c_per_group;
    int in            = (bid / c_per_group) % n;
    int ig            = bid / (n * c_per_group);

    if constexpr(ASSUME_PACKED)
    {
        p_in += static_cast<size_t>(in) * c * hi * wi +
                static_cast<size_t>(ig) * c_per_group * hi * wi + static_cast<size_t>(ic) * hi * wi;

        p_wei += static_cast<size_t>(ig) * k_per_group * c_per_group * fy * fx +
                 static_cast<size_t>(ic) * fy * fx;

        p_out +=
            static_cast<size_t>(in) * k * ho * wo + static_cast<size_t>(ig) * k_per_group * ho * wo;
    }
    else
    {
        p_in += static_cast<size_t>(in) * in_strides[4] + static_cast<size_t>(ig) * in_strides[3] +
                static_cast<size_t>(ic) * in_strides[2];

        p_wei +=
            static_cast<size_t>(ig) * wei_strides[4] + static_cast<size_t>(ic) * wei_strides[2];

        p_out +=
            static_cast<size_t>(in) * out_strides[4] + static_cast<size_t>(ig) * out_strides[3];
    }

    for(int tid = threadIdx.x; tid < thread_length; tid += blockDim.x)
    {
        int ihi = tid / wi;
        int iwi = tid % wi;

        acc_data_t value = 0;

        for(int ik = 0; ik < k_per_group; ik++)
        {
            for(int iy = 0; iy < fy; iy++)
            {
                int valid_h = 1;
                int cur_ho  = ihi + py - dy * iy; // cur_h = sy*iho-py+dy*iy;
                if(cur_ho < 0 || cur_ho % sy)
                    valid_h &= 0;
                cur_ho /= sy;
                if(cur_ho >= ho)
                    valid_h &= 0;
                for(int ix = 0; ix < fx; ix++)
                {
                    int valid_w = 1;
                    int cur_wo  = iwi + px - dx * ix; // cur_w = sx*iwo-px+dx*ix;
                    if(cur_wo < 0 || cur_wo % sx)
                        valid_w &= 0;
                    cur_wo /= sx;
                    if(cur_wo >= wo)
                        valid_w &= 0;

                    if(valid_h & valid_w)
                    {
                        if constexpr(ASSUME_PACKED)
                        {
                            size_t o_idx = static_cast<size_t>(ik) * ho * wo +
                                           static_cast<size_t>(cur_ho) * wo +
                                           static_cast<size_t>(cur_wo);

                            size_t f_idx = static_cast<size_t>(ik) * c_per_group * fy * fx +
                                           static_cast<size_t>(iy) * fx + static_cast<size_t>(ix);

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_wei[f_idx]);
                        }
                        else
                        {
                            size_t o_idx = static_cast<size_t>(ik) * out_strides[2] +
                                           static_cast<size_t>(cur_ho) * out_strides[1] +
                                           static_cast<size_t>(cur_wo) * out_strides[0];

                            size_t f_idx = static_cast<size_t>(ik) * wei_strides[3] +
                                           static_cast<size_t>(iy) * wei_strides[1] +
                                           static_cast<size_t>(ix) * wei_strides[0];

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_wei[f_idx]);
                        }
                    }
                }
            }
        }

        if constexpr(ASSUME_PACKED)
        {
            size_t i_idx = static_cast<size_t>(ihi) * wi + static_cast<size_t>(iwi);
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_in, value, alpha, beta, i_idx);
        }
        else
        {
            size_t i_idx =
                static_cast<size_t>(ihi) * in_strides[1] + static_cast<size_t>(iwi) * in_strides[0];
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_in, value, alpha, beta, i_idx);
        }
    }
}
template <bool ASSUME_PACKED,
          typename src_data_t,
          typename acc_data_t,
          typename dst_data_t,
          int use_tf32 = 0>
inline __device__ void naive_conv_bwd_ncdhw(dst_data_t* __restrict__ p_in,
                                            const src_data_t* __restrict__ p_wei,
                                            const double alpha,
                                            const double beta,
                                            const src_data_t* __restrict__ p_out,
                                            Strides6D in_strides,
                                            Strides6D wei_strides,
                                            Strides6D out_strides,
                                            int di,
                                            int hi,
                                            int wi,
                                            int n,
                                            int k_per_group,
                                            int c_per_group,
                                            int do_,
                                            int ho,
                                            int wo,
                                            int sz,
                                            int sy,
                                            int sx,
                                            int dz,
                                            int dy,
                                            int dx,
                                            int pz,
                                            int py,
                                            int px,
                                            int fz,
                                            int fy,
                                            int fx,
                                            int group)
{
    /*
     *  need to compute total input pixel: `group * n * c_per_group * di * hi *
     * wi`.
     *  to distribute this workload, let one workgroup compute `di * hi * wi`
     * pixel,
     *  hence need `group * n * c_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = di * hi * wi;
    int bid           = blockIdx.x;
    int ic            = bid % c_per_group;
    int in            = (bid / c_per_group) % n;
    int ig            = bid / (n * c_per_group);

    if constexpr(ASSUME_PACKED)
    {
        p_in += static_cast<size_t>(in) * c * di * hi * wi +
                static_cast<size_t>(ig) * c_per_group * di * hi * wi +
                static_cast<size_t>(ic) * di * hi * wi;

        p_wei += static_cast<size_t>(ig) * k_per_group * c_per_group * fz * fy * fx +
                 static_cast<size_t>(ic) * fz * fy * fx;

        p_out += static_cast<size_t>(in) * k * do_ * ho * wo +
                 static_cast<size_t>(ig) * k_per_group * do_ * ho * wo;
    }
    else
    {
        p_in += static_cast<size_t>(in) * in_strides[5] + static_cast<size_t>(ig) * in_strides[4] +
                static_cast<size_t>(ic) * in_strides[3];

        p_wei +=
            static_cast<size_t>(ig) * wei_strides[5] + static_cast<size_t>(ic) * wei_strides[3];

        p_out +=
            static_cast<size_t>(in) * out_strides[5] + static_cast<size_t>(ig) * out_strides[4];
    }

    for(int tid = threadIdx.x; tid < thread_length; tid += blockDim.x)
    {
        int iwi = tid % wi;
        int ihi = (tid / wi) % hi;
        int idi = tid / (hi * wi);

        acc_data_t value = 0;

        for(int ik = 0; ik < k_per_group; ik++)
        {
            for(int iz = 0; iz < fz; iz++)
            {
                int valid_d = 1;
                int cur_do  = idi + pz - dz * iz;
                if(cur_do < 0 || cur_do % sz)
                    valid_d &= 0;
                cur_do /= sz;
                if(cur_do >= do_)
                    valid_d &= 0;
                for(int iy = 0; iy < fy; iy++)
                {
                    int valid_h = 1;
                    int cur_ho  = ihi + py - dy * iy; // cur_h = sy*iho-py+dy*iy;
                    if(cur_ho < 0 || cur_ho % sy)
                        valid_h &= 0;
                    cur_ho /= sy;
                    if(cur_ho >= ho)
                        valid_h &= 0;
                    for(int ix = 0; ix < fx; ix++)
                    {
                        int valid_w = 1;
                        int cur_wo  = iwi + px - dx * ix; // cur_w = sx*iwo-px+dx*ix;
                        if(cur_wo < 0 || cur_wo % sx)
                            valid_w &= 0;
                        cur_wo /= sx;
                        if(cur_wo >= wo)
                            valid_w &= 0;

                        if(valid_d & valid_h & valid_w)
                        {
                            if constexpr(ASSUME_PACKED)
                            {
                                size_t o_idx = static_cast<size_t>(ik) * do_ * ho * wo +
                                               static_cast<size_t>(cur_do) * ho * wo +
                                               static_cast<size_t>(cur_ho) * wo +
                                               static_cast<size_t>(cur_wo);

                                size_t f_idx =
                                    static_cast<size_t>(ik) * c_per_group * fz * fy * fx +
                                    static_cast<size_t>(iz) * fy * fx +
                                    static_cast<size_t>(iy) * fx + static_cast<size_t>(ix);

                                value += cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]) *
                                         cast_to<src_data_t, acc_data_t, use_tf32>(p_wei[f_idx]);
                            }
                            else
                            {
                                size_t o_idx = static_cast<size_t>(ik) * out_strides[3] +
                                               static_cast<size_t>(cur_do) * out_strides[2] +
                                               static_cast<size_t>(cur_ho) * out_strides[1] +
                                               static_cast<size_t>(cur_wo) * out_strides[0];

                                size_t f_idx = static_cast<size_t>(ik) * wei_strides[4] +
                                               static_cast<size_t>(iz) * wei_strides[2] +
                                               static_cast<size_t>(iy) * wei_strides[1] +
                                               static_cast<size_t>(ix) * wei_strides[0];

                                value += cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]) *
                                         cast_to<src_data_t, acc_data_t, use_tf32>(p_wei[f_idx]);
                            }
                        }
                    }
                }
            }
        }

        if constexpr(ASSUME_PACKED)
        {
            size_t i_idx = static_cast<size_t>(idi) * hi * wi + static_cast<size_t>(ihi) * wi +
                           static_cast<size_t>(iwi);
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_in, value, alpha, beta, i_idx);
        }
        else
        {
            size_t i_idx = static_cast<size_t>(idi) * in_strides[2] +
                           static_cast<size_t>(ihi) * in_strides[1] +
                           static_cast<size_t>(iwi) * in_strides[0];
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_in, value, alpha, beta, i_idx);
        }
    }
}

template <bool ASSUME_PACKED,
          typename src_data_t,
          typename acc_data_t,
          typename dst_data_t,
          int use_tf32 = 0>
inline __device__ void naive_conv_bwd_nhwc(dst_data_t* __restrict__ p_in,
                                           const src_data_t* __restrict__ p_wei,
                                           const double alpha,
                                           const double beta,
                                           const src_data_t* __restrict__ p_out,
                                           Strides5D in_strides,
                                           Strides5D wei_strides,
                                           Strides5D out_strides,
                                           int hi,
                                           int wi,
                                           int n,
                                           int k_per_group,
                                           int c_per_group,
                                           int ho,
                                           int wo,
                                           int sy,
                                           int sx,
                                           int dy,
                                           int dx,
                                           int py,
                                           int px,
                                           int fy,
                                           int fx,
                                           int group)
{
    /*
     *  need to compute total input pixel: `group * n * hi * wi * c_per_group`.
     *  to distribute this workload, let one workgroup compute `wi *
     *  group * c_per_group` pixel,
     *  hence need `n * hi` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = wi * c;
    int bid           = blockIdx.x;
    int ihi           = bid % hi;
    int in            = (bid / hi) % n;

    if constexpr(ASSUME_PACKED)
    {
        p_in += static_cast<size_t>(in) * hi * wi * c + static_cast<size_t>(ihi) * wi * c;

        p_out += static_cast<size_t>(in) * ho * wo * k;
    }
    else
    {
        p_in += static_cast<size_t>(in) * in_strides[4] + static_cast<size_t>(ihi) * in_strides[3];

        p_out += static_cast<size_t>(in) * out_strides[4];
    }

    for(int tid = threadIdx.x; tid < thread_length; tid += blockDim.x)
    {
        // We want to compute
        //      iwi = tid / c
        //      ic  = tid % c
        // , but
        //      tid = tid / c * c + tid % c = iwi * c + ic
        // so we can avoid the % operation.
        int iwi       = tid / c;
        int global_ic = tid - iwi * c;
        int ig        = global_ic / c_per_group;
        int ic        = global_ic - ig * c_per_group;

        acc_data_t value = 0;

        for(int iy = 0; iy < fy; iy++)
        {
            int valid_h = 1;
            int cur_ho  = ihi + py - dy * iy; // cur_h = sy*iho-py+dy*iy;
            if(cur_ho < 0 || cur_ho % sy)
                valid_h &= 0;
            cur_ho /= sy;
            if(cur_ho >= ho)
                valid_h &= 0;
            for(int ix = 0; ix < fx; ix++)
            {
                int valid_w = 1;
                int cur_wo  = iwi + px - dx * ix; // cur_w = sx*iwo-px+dx*ix;
                if(cur_wo < 0 || cur_wo % sx)
                    valid_w &= 0;
                cur_wo /= sx;
                if(cur_wo >= wo)
                    valid_w &= 0;
                for(int ik = 0; ik < k_per_group; ik++)
                {
                    if(valid_h & valid_w)
                    {
                        if constexpr(ASSUME_PACKED)
                        {
                            size_t o_idx = static_cast<size_t>(cur_ho) * wo * k +
                                           static_cast<size_t>(cur_wo) * k +
                                           static_cast<size_t>(ig) * k_per_group +
                                           static_cast<size_t>(ik);

                            size_t f_idx =
                                static_cast<size_t>(ig) * k_per_group * fy * fx * c_per_group +
                                static_cast<size_t>(ik) * fy * fx * c_per_group +
                                static_cast<size_t>(iy) * fx * c_per_group +
                                static_cast<size_t>(ix) * c_per_group + static_cast<size_t>(ic);

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_wei[f_idx]);
                        }
                        else
                        {
                            size_t o_idx = static_cast<size_t>(cur_ho) * out_strides[3] +
                                           static_cast<size_t>(cur_wo) * out_strides[2] +
                                           static_cast<size_t>(ig) * out_strides[1] +
                                           static_cast<size_t>(ik) * out_strides[0];

                            size_t f_idx = static_cast<size_t>(ig) * wei_strides[4] +
                                           static_cast<size_t>(ik) * wei_strides[3] +
                                           static_cast<size_t>(iy) * wei_strides[2] +
                                           static_cast<size_t>(ix) * wei_strides[1] +
                                           static_cast<size_t>(ic) * wei_strides[0];

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_wei[f_idx]);
                        }
                    }
                }
            }
        }

        if constexpr(ASSUME_PACKED)
        {
            size_t i_idx = static_cast<size_t>(iwi) * c + static_cast<size_t>(ig) * c_per_group +
                           static_cast<size_t>(ic);
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_in, value, alpha, beta, i_idx);
        }
        else
        {
            size_t i_idx = static_cast<size_t>(iwi) * in_strides[2] +
                           static_cast<size_t>(ig) * in_strides[1] +
                           static_cast<size_t>(ic) * in_strides[0];
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_in, value, alpha, beta, i_idx);
        }
    }
}

template <bool ASSUME_PACKED,
          typename src_data_t,
          typename acc_data_t,
          typename dst_data_t,
          int use_tf32 = 0>
inline __device__ void naive_conv_bwd_ndhwc(dst_data_t* __restrict__ p_in,
                                            const src_data_t* __restrict__ p_wei,
                                            const double alpha,
                                            const double beta,
                                            const src_data_t* __restrict__ p_out,
                                            Strides6D in_strides,
                                            Strides6D wei_strides,
                                            Strides6D out_strides,
                                            int di,
                                            int hi,
                                            int wi,
                                            int n,
                                            int k_per_group,
                                            int c_per_group,
                                            int do_,
                                            int ho,
                                            int wo,
                                            int sz,
                                            int sy,
                                            int sx,
                                            int dz,
                                            int dy,
                                            int dx,
                                            int pz,
                                            int py,
                                            int px,
                                            int fz,
                                            int fy,
                                            int fx,
                                            int group)
{

    /*
     *  need to compute total input pixel: `group * n * di * hi * wi *
     * c_per_group`.
     *  to distribute this workload, let one workgroup compute `hi * wi *
     * c_per_group` pixel,
     *  hence need `group * n * di` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = hi * wi * c_per_group;
    int bid           = blockIdx.x;
    int idi           = bid % di;
    int in            = (bid / di) % n;
    int ig            = bid / (n * di);

    if constexpr(ASSUME_PACKED)
    {
        p_in += static_cast<size_t>(in) * di * hi * wi * c +
                static_cast<size_t>(idi) * hi * wi * c + static_cast<size_t>(ig) * c_per_group;

        p_wei += static_cast<size_t>(ig) * k_per_group * fz * fy * fx * c_per_group;

        p_out +=
            static_cast<size_t>(in) * do_ * ho * wo * k + static_cast<size_t>(ig) * k_per_group;
    }
    else
    {
        p_in += static_cast<size_t>(in) * in_strides[5] + static_cast<size_t>(idi) * in_strides[4] +
                static_cast<size_t>(ig) * in_strides[1];

        p_wei += static_cast<size_t>(ig) * wei_strides[5];

        p_out +=
            static_cast<size_t>(in) * out_strides[5] + static_cast<size_t>(ig) * out_strides[1];
    }

    for(int tid = threadIdx.x; tid < thread_length; tid += blockDim.x)
    {
        int ic  = tid % c_per_group;
        int iwi = (tid / c_per_group) % wi;
        int ihi = (tid / (c_per_group * wi));

        acc_data_t value = 0;

        for(int iz = 0; iz < fz; iz++)
        {
            int valid_d = 1;
            int cur_do  = idi + pz - dz * iz;
            if(cur_do < 0 || cur_do % sz)
                valid_d &= 0;
            cur_do /= sz;
            if(cur_do >= do_)
                valid_d &= 0;
            for(int iy = 0; iy < fy; iy++)
            {
                int valid_h = 1;
                int cur_ho  = ihi + py - dy * iy; // cur_h = sy*iho-py+dy*iy;
                if(cur_ho < 0 || cur_ho % sy)
                    valid_h &= 0;
                cur_ho /= sy;
                if(cur_ho >= ho)
                    valid_h &= 0;
                for(int ix = 0; ix < fx; ix++)
                {
                    int valid_w = 1;
                    int cur_wo  = iwi + px - dx * ix; // cur_w = sx*iwo-px+dx*ix;
                    if(cur_wo < 0 || cur_wo % sx)
                        valid_w &= 0;
                    cur_wo /= sx;
                    if(cur_wo >= wo)
                        valid_w &= 0;
                    for(int ik = 0; ik < k_per_group; ik++)
                    {
                        if(valid_d & valid_h & valid_w)
                        {
                            if constexpr(ASSUME_PACKED)
                            {
                                size_t o_idx = static_cast<size_t>(cur_do) * ho * wo * k +
                                               static_cast<size_t>(cur_ho) * wo * k +
                                               static_cast<size_t>(cur_wo) * k +
                                               static_cast<size_t>(ik);

                                size_t f_idx =
                                    static_cast<size_t>(ik) * fz * fy * fx * c_per_group +
                                    static_cast<size_t>(iz) * fy * fx * c_per_group +
                                    static_cast<size_t>(iy) * fx * c_per_group +
                                    static_cast<size_t>(ix) * c_per_group + static_cast<size_t>(ic);

                                value += cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]) *
                                         cast_to<src_data_t, acc_data_t, use_tf32>(p_wei[f_idx]);
                            }
                            else
                            {
                                size_t o_idx = static_cast<size_t>(cur_do) * out_strides[4] +
                                               static_cast<size_t>(cur_ho) * out_strides[3] +
                                               static_cast<size_t>(cur_wo) * out_strides[2] +
                                               static_cast<size_t>(ik) * out_strides[0];

                                size_t f_idx = static_cast<size_t>(ik) * wei_strides[4] +
                                               static_cast<size_t>(iz) * wei_strides[3] +
                                               static_cast<size_t>(iy) * wei_strides[2] +
                                               static_cast<size_t>(ix) * wei_strides[1] +
                                               static_cast<size_t>(ic) * wei_strides[0];

                                value += cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]) *
                                         cast_to<src_data_t, acc_data_t, use_tf32>(p_wei[f_idx]);
                            }
                        }
                    }
                }
            }
        }

        if constexpr(ASSUME_PACKED)
        {
            size_t i_idx = static_cast<size_t>(ihi) * wi * c + static_cast<size_t>(iwi) * c +
                           static_cast<size_t>(ic);
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_in, value, alpha, beta, i_idx);
        }
        else
        {
            size_t i_idx = static_cast<size_t>(ihi) * in_strides[3] +
                           static_cast<size_t>(iwi) * in_strides[2] +
                           static_cast<size_t>(ic) * in_strides[0];
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_in, value, alpha, beta, i_idx);
        }
    }
}


// double accumulator (GPU reference mode)
DEFINE_2D_NAIVE_CONV_KERNEL(bwd, nchw, float, double, float, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(bwd, nchw, float, double, float, 1)
DEFINE_2D_NAIVE_CONV_KERNEL(bwd, nchw, half, double, half, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(bwd, nchw, ushort, double, ushort, 0)
// float accumulator (normal solver mode)
DEFINE_2D_NAIVE_CONV_KERNEL(bwd, nchw, float, float, float, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(bwd, nchw, half, float, half, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(bwd, nchw, ushort, float, ushort, 0)

// double accumulator (GPU reference mode)
DEFINE_2D_NAIVE_CONV_KERNEL(bwd, nhwc, float, double, float, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(bwd, nhwc, float, double, float, 1)
DEFINE_2D_NAIVE_CONV_KERNEL(bwd, nhwc, half, double, half, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(bwd, nhwc, ushort, double, ushort, 0)
// float accumulator (normal solver mode)
DEFINE_2D_NAIVE_CONV_KERNEL(bwd, nhwc, float, float, float, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(bwd, nhwc, half, float, half, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(bwd, nhwc, ushort, float, ushort, 0)

// double accumulator (GPU reference mode)
DEFINE_3D_NAIVE_CONV_KERNEL(bwd, ncdhw, float, double, float, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(bwd, ncdhw, float, double, float, 1)
DEFINE_3D_NAIVE_CONV_KERNEL(bwd, ncdhw, half, double, half, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(bwd, ncdhw, ushort, double, ushort, 0)
// float accumulator (normal solver mode)
DEFINE_3D_NAIVE_CONV_KERNEL(bwd, ncdhw, float, float, float, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(bwd, ncdhw, half, float, half, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(bwd, ncdhw, ushort, float, ushort, 0)

// double accumulator (GPU reference mode)
DEFINE_3D_NAIVE_CONV_KERNEL(bwd, ndhwc, float, double, float, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(bwd, ndhwc, float, double, float, 1)
DEFINE_3D_NAIVE_CONV_KERNEL(bwd, ndhwc, half, double, half, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(bwd, ndhwc, ushort, double, ushort, 0)
// float accumulator (normal solver mode)
DEFINE_3D_NAIVE_CONV_KERNEL(bwd, ndhwc, float, float, float, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(bwd, ndhwc, half, float, half, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(bwd, ndhwc, ushort, float, ushort, 0)

