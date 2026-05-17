#include "naive_conv.hpp"


template <bool ASSUME_PACKED,
          typename src_data_t,
          typename acc_data_t,
          typename dst_data_t,
          int use_tf32 = 0>
inline __device__ void naive_conv_wrw_nchw(const src_data_t* __restrict__ p_in,
                                           dst_data_t* __restrict__ p_wei,
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
     *  need to compute total filter pixel: `group * k_per_group * c_per_group *
     * fy * fx`.
     *  to distribute this workload, let one workgroup compute `c_per_group * fy
     * * fx` pixel,
     *  hence need `group * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = c_per_group * fy * fx;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int ig            = bid / k_per_group;

    if constexpr(ASSUME_PACKED)
    {
        p_in += static_cast<size_t>(ig) * c_per_group * hi * wi;

        p_wei += static_cast<size_t>(ig) * k_per_group * c_per_group * fy * fx +
                 static_cast<size_t>(ik) * c_per_group * fy * fx;

        p_out +=
            static_cast<size_t>(ig) * k_per_group * ho * wo + static_cast<size_t>(ik) * ho * wo;
    }
    else
    {
        p_in += static_cast<size_t>(ig) * in_strides[3];

        p_wei +=
            static_cast<size_t>(ig) * wei_strides[4] + static_cast<size_t>(ik) * wei_strides[3];

        p_out +=
            static_cast<size_t>(ig) * out_strides[3] + static_cast<size_t>(ik) * out_strides[2];
    }

    for(int tid = threadIdx.x; tid < thread_length; tid += blockDim.x)
    {
        int ix = tid % fx;
        int iy = (tid / fx) % fy;
        int ic = tid / (fx * fy);

        acc_data_t value = 0;

        for(int in = 0; in < n; in++)
        {
            for(int iho = 0; iho < ho; iho++)
            {
                int valid_h = 1;
                int cur_h   = sy * iho - py + dy * iy;
                if(cur_h < 0 || cur_h >= hi)
                    valid_h &= 0;
                for(int iwo = 0; iwo < wo; iwo++)
                {
                    int valid_w = 1;
                    int cur_w   = sx * iwo - px + dx * ix;
                    if(cur_w < 0 || cur_w >= wi)
                        valid_w &= 0;

                    if(valid_h & valid_w)
                    {
                        if constexpr(ASSUME_PACKED)
                        {
                            size_t i_idx = static_cast<size_t>(in) * c * hi * wi +
                                           static_cast<size_t>(ic) * hi * wi +
                                           static_cast<size_t>(cur_h) * wi +
                                           static_cast<size_t>(cur_w);

                            size_t o_idx = static_cast<size_t>(in) * k * ho * wo +
                                           static_cast<size_t>(iho) * wo + static_cast<size_t>(iwo);

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                        }
                        else
                        {
                            size_t i_idx = static_cast<size_t>(in) * in_strides[4] +
                                           static_cast<size_t>(ic) * in_strides[2] +
                                           static_cast<size_t>(cur_h) * in_strides[1] +
                                           static_cast<size_t>(cur_w) * in_strides[0];

                            size_t o_idx = static_cast<size_t>(in) * out_strides[4] +
                                           static_cast<size_t>(iho) * out_strides[1] +
                                           static_cast<size_t>(iwo) * out_strides[0];

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                        }
                    }
                }
            }
        }

        if constexpr(ASSUME_PACKED)
        {
            size_t f_idx = static_cast<size_t>(ic) * fy * fx + static_cast<size_t>(iy) * fx +
                           static_cast<size_t>(ix);
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_wei, value, alpha, beta, f_idx);
        }
        else
        {
            size_t f_idx = static_cast<size_t>(ic) * wei_strides[2] +
                           static_cast<size_t>(iy) * wei_strides[1] +
                           static_cast<size_t>(ix) * wei_strides[0];
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_wei, value, alpha, beta, f_idx);
        }
    }
}

template <bool ASSUME_PACKED,
          typename src_data_t,
          typename acc_data_t,
          typename dst_data_t,
          int use_tf32 = 0>
inline __device__ void naive_conv_wrw_ncdhw(const src_data_t* __restrict__ p_in,
                                            dst_data_t* __restrict__ p_wei,
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
     *  need to compute total filter pixel: `group * k_per_group * c_per_group *
     * fz * fy * fx`.
     *  to distribute this workload, let one workgroup compute `c_per_group * fz
     * * fy * fx` pixel,
     *  hence need `group * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = c_per_group * fz * fy * fx;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int ig            = bid / k_per_group;

    if constexpr(ASSUME_PACKED)
    {
        p_in += static_cast<size_t>(ig) * c_per_group * di * hi * wi;

        p_wei += static_cast<size_t>(ig) * k_per_group * c_per_group * fz * fy * fx +
                 static_cast<size_t>(ik) * c_per_group * fz * fy * fx;

        p_out += static_cast<size_t>(ig) * k_per_group * do_ * ho * wo +
                 static_cast<size_t>(ik) * do_ * ho * wo;
    }
    else
    {
        p_in += static_cast<size_t>(ig) * in_strides[4];

        p_wei +=
            static_cast<size_t>(ig) * wei_strides[5] + static_cast<size_t>(ik) * wei_strides[4];

        p_out +=
            static_cast<size_t>(ig) * out_strides[4] + static_cast<size_t>(ik) * out_strides[3];
    }

    for(int tid = threadIdx.x; tid < thread_length; tid += blockDim.x)
    {
        int ix = tid % fx;
        int iy = (tid / fx) % fy;
        int iz = (tid / (fx * fy)) % fz;
        int ic = tid / (fx * fy * fz);

        acc_data_t value = 0;

        for(int in = 0; in < n; in++)
        {
            for(int ido = 0; ido < do_; ido++)
            {
                int valid_d = 1;
                int cur_d   = sz * ido - pz + dz * iz;
                if(cur_d < 0 || cur_d >= di)
                    valid_d &= 0;
                for(int iho = 0; iho < ho; iho++)
                {
                    int valid_h = 1;
                    int cur_h   = sy * iho - py + dy * iy;
                    if(cur_h < 0 || cur_h >= hi)
                        valid_h &= 0;
                    for(int iwo = 0; iwo < wo; iwo++)
                    {
                        int valid_w = 1;
                        int cur_w   = sx * iwo - px + dx * ix;
                        if(cur_w < 0 || cur_w >= wi)
                            valid_w &= 0;

                        if(valid_d & valid_h & valid_w)
                        {
                            if constexpr(ASSUME_PACKED)
                            {
                                size_t i_idx = static_cast<size_t>(in) * c * di * hi * wi +
                                               static_cast<size_t>(ic) * di * hi * wi +
                                               static_cast<size_t>(cur_d) * hi * wi +
                                               static_cast<size_t>(cur_h) * wi +
                                               static_cast<size_t>(cur_w);

                                size_t o_idx = static_cast<size_t>(in) * k * do_ * ho * wo +
                                               static_cast<size_t>(ido) * ho * wo +
                                               static_cast<size_t>(iho) * wo +
                                               static_cast<size_t>(iwo);

                                value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                         cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                            }
                            else
                            {
                                size_t i_idx = static_cast<size_t>(in) * in_strides[5] +
                                               static_cast<size_t>(ic) * in_strides[3] +
                                               static_cast<size_t>(cur_d) * in_strides[2] +
                                               static_cast<size_t>(cur_h) * in_strides[1] +
                                               static_cast<size_t>(cur_w) * in_strides[0];

                                size_t o_idx = static_cast<size_t>(in) * out_strides[5] +
                                               static_cast<size_t>(ido) * out_strides[2] +
                                               static_cast<size_t>(iho) * out_strides[1] +
                                               static_cast<size_t>(iwo) * out_strides[0];

                                value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                         cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                            }
                        }
                    }
                }
            }
        }

        if constexpr(ASSUME_PACKED)
        {
            size_t f_idx = static_cast<size_t>(ic) * fz * fy * fx +
                           static_cast<size_t>(iz) * fy * fx + static_cast<size_t>(iy) * fx +
                           static_cast<size_t>(ix);
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_wei, value, alpha, beta, f_idx);
        }
        else
        {
            size_t f_idx = static_cast<size_t>(ic) * wei_strides[3] +
                           static_cast<size_t>(iz) * wei_strides[2] +
                           static_cast<size_t>(iy) * wei_strides[1] +
                           static_cast<size_t>(ix) * wei_strides[0];
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_wei, value, alpha, beta, f_idx);
        }
    }
}

template <bool ASSUME_PACKED,
          typename src_data_t,
          typename acc_data_t,
          typename dst_data_t,
          int use_tf32 = 0>
inline __device__ void naive_conv_wrw_nhwc(const src_data_t* __restrict__ p_in,
                                           dst_data_t* __restrict__ p_wei,
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
     *  need to compute total filter pixel: `group * k_per_group * fy * fx *
     * c_per_group`.
     *  to distribute this workload, let one workgroup compute `fy * fx *
     * c_per_group` pixel,
     *  hence need `group * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = c_per_group * fy * fx;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int ig            = bid / k_per_group;

    if constexpr(ASSUME_PACKED)
    {
        p_in += static_cast<size_t>(ig) * c_per_group;

        p_wei += static_cast<size_t>(ig) * k_per_group * fy * fx * c_per_group +
                 static_cast<size_t>(ik) * fy * fx * c_per_group;

        p_out += static_cast<size_t>(ig) * k_per_group + static_cast<size_t>(ik);
    }
    else
    {
        p_in += static_cast<size_t>(ig) * in_strides[1];

        p_wei +=
            static_cast<size_t>(ig) * wei_strides[4] + static_cast<size_t>(ik) * wei_strides[3];

        p_out +=
            static_cast<size_t>(ig) * out_strides[1] + static_cast<size_t>(ik) * out_strides[0];
    }

    for(int tid = threadIdx.x; tid < thread_length; tid += blockDim.x)
    {
        int ic = tid % c_per_group;
        int ix = (tid / c_per_group) % fx;
        int iy = tid / (c_per_group * fx);

        acc_data_t value = 0;

        for(int in = 0; in < n; in++)
        {
            for(int iho = 0; iho < ho; iho++)
            {
                int valid_h = 1;
                int cur_h   = sy * iho - py + dy * iy;
                if(cur_h < 0 || cur_h >= hi)
                    valid_h &= 0;
                for(int iwo = 0; iwo < wo; iwo++)
                {
                    int valid_w = 1;
                    int cur_w   = sx * iwo - px + dx * ix;
                    if(cur_w < 0 || cur_w >= wi)
                        valid_w &= 0;

                    if(valid_h & valid_w)
                    {

                        if constexpr(ASSUME_PACKED)
                        {
                            size_t i_idx = static_cast<size_t>(in) * hi * wi * c +
                                           static_cast<size_t>(cur_h) * wi * c +
                                           static_cast<size_t>(cur_w) * c + static_cast<size_t>(ic);

                            size_t o_idx = static_cast<size_t>(in) * ho * wo * k +
                                           static_cast<size_t>(iho) * wo * k +
                                           static_cast<size_t>(iwo) * k;

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                        }
                        else
                        {
                            size_t i_idx = static_cast<size_t>(in) * in_strides[4] +
                                           static_cast<size_t>(cur_h) * in_strides[3] +
                                           static_cast<size_t>(cur_w) * in_strides[2] +
                                           static_cast<size_t>(ic) * in_strides[0];

                            size_t o_idx = static_cast<size_t>(in) * out_strides[4] +
                                           static_cast<size_t>(iho) * out_strides[3] +
                                           static_cast<size_t>(iwo) * out_strides[2];

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                        }
                    }
                }
            }
        }

        if constexpr(ASSUME_PACKED)
        {
            size_t f_idx = static_cast<size_t>(iy) * fx * c_per_group +
                           static_cast<size_t>(ix) * c_per_group + static_cast<size_t>(ic);
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_wei, value, alpha, beta, f_idx);
        }
        else
        {
            size_t f_idx = static_cast<size_t>(iy) * wei_strides[2] +
                           static_cast<size_t>(ix) * wei_strides[1] +
                           static_cast<size_t>(ic) * wei_strides[0];
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_wei, value, alpha, beta, f_idx);
        }
    }
}

template <bool ASSUME_PACKED,
          typename src_data_t,
          typename acc_data_t,
          typename dst_data_t,
          int use_tf32 = 0>
inline __device__ void naive_conv_wrw_ndhwc(const src_data_t* __restrict__ p_in,
                                            dst_data_t* __restrict__ p_wei,
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
     *  need to compute total filter pixel: `group * k_per_group * fz * fy * fx
     * * c_per_group`.
     *  to distribute this workload, let one workgroup compute `fz * fy * fx *
     * c_per_group` pixel,
     *  hence need `group * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = fz * fy * fx * c_per_group;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int ig            = bid / k_per_group;

    if constexpr(ASSUME_PACKED)
    {
        p_in += static_cast<size_t>(ig) * c_per_group;

        p_wei += static_cast<size_t>(ig) * k_per_group * fz * fy * fx * c_per_group +
                 static_cast<size_t>(ik) * fz * fy * fx * c_per_group;

        p_out += static_cast<size_t>(ig) * k_per_group + static_cast<size_t>(ik);
    }
    else
    {
        p_in += static_cast<size_t>(ig) * in_strides[1];

        p_wei +=
            static_cast<size_t>(ig) * wei_strides[5] + static_cast<size_t>(ik) * wei_strides[4];

        p_out +=
            static_cast<size_t>(ig) * out_strides[1] + static_cast<size_t>(ik) * out_strides[0];
    }

    for(int tid = threadIdx.x; tid < thread_length; tid += blockDim.x)
    {
        int ic = tid % c_per_group;
        int ix = (tid / c_per_group) % fx;
        int iy = (tid / (c_per_group * fx)) % fy;
        int iz = (tid / (c_per_group * fx * fy));

        acc_data_t value = 0;

        for(int in = 0; in < n; in++)
        {
            for(int ido = 0; ido < do_; ido++)
            {
                int valid_d = 1;
                int cur_d   = sz * ido - pz + dz * iz;
                if(cur_d < 0 || cur_d >= di)
                    valid_d &= 0;
                for(int iho = 0; iho < ho; iho++)
                {
                    int valid_h = 1;
                    int cur_h   = sy * iho - py + dy * iy;
                    if(cur_h < 0 || cur_h >= hi)
                        valid_h &= 0;
                    for(int iwo = 0; iwo < wo; iwo++)
                    {
                        int valid_w = 1;
                        int cur_w   = sx * iwo - px + dx * ix;
                        if(cur_w < 0 || cur_w >= wi)
                            valid_w &= 0;

                        if(valid_d & valid_h & valid_w)
                        {

                            if constexpr(ASSUME_PACKED)
                            {
                                size_t i_idx = static_cast<size_t>(in) * di * hi * wi * c +
                                               static_cast<size_t>(cur_d) * hi * wi * c +
                                               static_cast<size_t>(cur_h) * wi * c +
                                               static_cast<size_t>(cur_w) * c +
                                               static_cast<size_t>(ic);

                                size_t o_idx = static_cast<size_t>(in) * do_ * ho * wo * k +
                                               static_cast<size_t>(ido) * ho * wo * k +
                                               static_cast<size_t>(iho) * wo * k +
                                               static_cast<size_t>(iwo) * k;

                                value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                         cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                            }
                            else
                            {

                                size_t i_idx = static_cast<size_t>(in) * in_strides[5] +
                                               static_cast<size_t>(cur_d) * in_strides[4] +
                                               static_cast<size_t>(cur_h) * in_strides[3] +
                                               static_cast<size_t>(cur_w) * in_strides[2] +
                                               static_cast<size_t>(ic) * in_strides[0];

                                size_t o_idx = static_cast<size_t>(in) * out_strides[5] +
                                               static_cast<size_t>(ido) * out_strides[4] +
                                               static_cast<size_t>(iho) * out_strides[3] +
                                               static_cast<size_t>(iwo) * out_strides[2];

                                value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                         cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                            }
                        }
                    }
                }
            }
        }

        if constexpr(ASSUME_PACKED)
        {
            size_t f_idx = static_cast<size_t>(iz) * fy * fx * c_per_group +
                           static_cast<size_t>(iy) * fx * c_per_group +
                           static_cast<size_t>(ix) * c_per_group + static_cast<size_t>(ic);
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_wei, value, alpha, beta, f_idx);
        }
        else
        {
            size_t f_idx = static_cast<size_t>(iz) * wei_strides[3] +
                           static_cast<size_t>(iy) * wei_strides[2] +
                           static_cast<size_t>(ix) * wei_strides[1] +
                           static_cast<size_t>(ic) * wei_strides[0];
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_wei, value, alpha, beta, f_idx);
        }
    }
}



DEFINE_2D_NAIVE_CONV_KERNEL(wrw, nchw, float, double, float, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(wrw, nchw, float, double, float, 1)
DEFINE_2D_NAIVE_CONV_KERNEL(wrw, nchw, half, double, half, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(wrw, nchw, ushort, double, ushort, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(wrw, nhwc, float, double, float, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(wrw, nhwc, float, double, float, 1)
DEFINE_2D_NAIVE_CONV_KERNEL(wrw, nhwc, half, double, half, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(wrw, nhwc, ushort, double, ushort, 0)

DEFINE_3D_NAIVE_CONV_KERNEL(wrw, ncdhw, float, double, float, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(wrw, ncdhw, float, double, float, 1)
DEFINE_3D_NAIVE_CONV_KERNEL(wrw, ncdhw, half, double, half, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(wrw, ncdhw, ushort, double, ushort, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(wrw, ndhwc, float, double, float, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(wrw, ndhwc, float, double, float, 1)
DEFINE_3D_NAIVE_CONV_KERNEL(wrw, ndhwc, half, double, half, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(wrw, ndhwc, ushort, double, ushort, 0)

