/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <hip/hip_runtime.h>
#include "float_types.h"

// HIP equivalents for math_ops.h functions
inline __device__ unsigned int iDiv_legacy(unsigned int v, unsigned int d)
{
    return static_cast<unsigned int>(
        static_cast<float>(v) * (1.0f / static_cast<float>(d)) + 0.00001f);
}

inline __device__ unsigned int iMod(unsigned int v, unsigned int u, unsigned int d)
{
    return v - u * d;
}

// Vector type support for MLO_READ_TYPE
#if MIOPEN_USE_FP32 == 1
#define FLOAT2 float2
#define FLOAT4 float4
#endif

// Broadcast scalar to vector type, and vector exp/log
#if MLO_READ_UNIT == 1
inline __device__ MLO_READ_TYPE MLO_MAKE_VEC(float v) { return static_cast<MLO_READ_TYPE>(v); }
inline __device__ MLO_READ_TYPE MLO_VEC_EXP(MLO_READ_TYPE v) { return expf(static_cast<float>(v)); }
inline __device__ MLO_READ_TYPE MLO_VEC_LOG(MLO_READ_TYPE v) { return logf(static_cast<float>(v)); }
#elif MLO_READ_UNIT == 2
inline __device__ float2 MLO_MAKE_VEC(float v) { return make_float2(v, v); }
inline __device__ float2 MLO_VEC_EXP(float2 v) { return make_float2(expf(v.x), expf(v.y)); }
inline __device__ float2 MLO_VEC_LOG(float2 v) { return make_float2(logf(v.x), logf(v.y)); }
#elif MLO_READ_UNIT == 4
inline __device__ float4 MLO_MAKE_VEC(float v) { return make_float4(v, v, v, v); }
inline __device__ float4 MLO_VEC_EXP(float4 v) { return make_float4(expf(v.x), expf(v.y), expf(v.z), expf(v.w)); }
inline __device__ float4 MLO_VEC_LOG(float4 v) { FLOAT vx = v.x; FLOAT vy = v.y; FLOAT vz = v.z; FLOAT vw = v.w; return make_float4(logf(vx), logf(vy), logf(vz), logf(vw)); }
#endif

#define DBG_OUT 0

#define MLO_LRN_GROUP_SZ2 1
#define MLO_LRN_STRIDE 1

#define MLO_LRN_LEFT_PAD0 (((MLO_LRN_PRE_PAD0 + MLO_READ_UNIT - 1) / MLO_READ_UNIT) * MLO_READ_UNIT)
#define MLO_LRN_RIGHT_SIDE                                                               \
    (((MLO_LRN_GROUP_SZ0 * MLO_LRN_N_HORIZ_OUT_PIX + MLO_LRN_PAD0 + MLO_READ_UNIT - 1) / \
      MLO_READ_UNIT) *                                                                   \
     MLO_READ_UNIT)
#define MLO_LRN_LCL_DATA_WIDTH (MLO_LRN_LEFT_PAD0 + MLO_LRN_RIGHT_SIDE)
#define MLO_LCL_READ4 (MLO_LRN_LCL_DATA_WIDTH / MLO_READ_UNIT)
#define MLO_LRN_LCL_DATA_HEIGHT (MLO_LRN_GROUP_SZ1 * MLO_LRN_N_VERT_OUT_PIX + MLO_LRN_KERNEL_SZ - 1)
#define MLO_LRN_GROUP_SZ (MLO_LRN_GROUP_SZ2 * MLO_LRN_GROUP_SZ1 * MLO_LRN_GROUP_SZ0)

struct LRNForwardParam
{
    FLOAT alphaoverarea;
    FLOAT alpha;
    FLOAT beta;
    FLOAT K;
};

extern "C" __global__ __launch_bounds__(MLO_LRN_GROUP_SZ0 * MLO_LRN_GROUP_SZ1 * MLO_LRN_GROUP_SZ2) void
MIOpenLRNWithinChannel_PS(const FLOAT* bot,
                          FLOAT* top,
#if MLO_LRN_DO_SCALE
                          FLOAT* scale,
#endif
                          FLOAT alphaoverarea,
                          [[maybe_unused]] FLOAT alpha,
                          FLOAT beta,
                          FLOAT K)
{
    // IT's taken from POOLING AVE with stride = 1'
    __shared__ FLOAT bot_data[MLO_LRN_LCL_DATA_WIDTH * MLO_LRN_LCL_DATA_HEIGHT];
    int x       = blockIdx.x * MLO_LRN_GROUP_SZ0 * MLO_LRN_N_HORIZ_OUT_PIX;
    int y       = blockIdx.y * MLO_LRN_GROUP_SZ1 * MLO_LRN_N_VERT_OUT_PIX;
    int lcl_id0 = threadIdx.x;
    int lcl_id1 = threadIdx.y;
    int ob      = blockIdx.z; // output * batch_sz
    int o       = iDiv_legacy(ob, MLO_LRN_BATCH_SZ);
    int b       = iMod(ob, o, MLO_LRN_BATCH_SZ);
    int bot_x   = x;
    int bot_y   = y;
    int bot_off = b * MLO_LRN_BOT_BATCH_STRIDE + o * MLO_LRN_BOT_CHANNEL_STRIDE;

    // load tile
    for(int b_j = lcl_id1; b_j < MLO_LRN_LCL_DATA_HEIGHT; b_j += MLO_LRN_GROUP_SZ1)
    {
        int bot_y_act = bot_y + b_j - MLO_LRN_PRE_PAD1;

        bool invisibleY = (bot_y_act < 0) || (bot_y_act >= MLO_LRN_BOT_HEIGHT);

        int bot_y_off = bot_y_act * MLO_LRN_BOT_STRIDE;

        int lcl_off_v = b_j * (int)MLO_LRN_LCL_DATA_WIDTH;

        for(int b_i = lcl_id0; b_i < MLO_LCL_READ4; b_i += MLO_LRN_GROUP_SZ0)
        {

            int bot_x_act = bot_x + (b_i * MLO_READ_UNIT) - MLO_LRN_LEFT_PAD0;

            bool invisibleX;
            for(int i = 0; i < MLO_READ_UNIT; ++i)
            {

                int bot_off_x = bot_off + bot_y_off + bot_x_act + i;

                invisibleX = (bot_x_act + i < 0) || (bot_x_act + i >= MLO_LRN_BOT_WIDTH);

                bot_off_x = (invisibleX || invisibleY) ? 0 : bot_off_x;

                FLOAT bot_val = bot[bot_off_x];

                bot_val = (invisibleX || invisibleY) ? 0 : bot_val;

                bot_data[lcl_off_v + (b_i * MLO_READ_UNIT) + i] = bot_val;
            }
        }
    }

    __syncthreads();
#if MLO_LRN_N_HORIZ_OUT_PIX > 1
    FLOAT partial_sum_x[MLO_LRN_N_HORIZ_OUT_PIX - 1]; // horizontal partial sum
#endif
#if MLO_LRN_N_VERT_OUT_PIX > 1
    FLOAT partial_sum_xy[MLO_LRN_N_VERT_OUT_PIX - 1]
                         [MLO_LRN_N_HORIZ_OUT_PIX]; // horizontal-vertical partial sums.
#endif
    FLOAT accum[MLO_LRN_N_VERT_OUT_PIX][MLO_LRN_N_HORIZ_OUT_PIX]; // accumulator

    int top_y = lcl_id1 * (int)MLO_LRN_N_VERT_OUT_PIX + y;
    int top_x = lcl_id0 * (int)MLO_LRN_N_HORIZ_OUT_PIX + x;

    int lcl_y = lcl_id1 * (int)MLO_LRN_N_VERT_OUT_PIX;
    int lcl_x =
        lcl_id0 * (int)(MLO_LRN_N_HORIZ_OUT_PIX) + (int)(MLO_LRN_LEFT_PAD0 - MLO_LRN_PRE_PAD0);
    int lcl_off = lcl_y * MLO_LRN_LCL_DATA_WIDTH + lcl_x;

    for(int j = 0; j < MLO_LRN_N_VERT_OUT_PIX; ++j)
    {
        for(int i = 0; i < MLO_LRN_N_HORIZ_OUT_PIX; ++i)
        {
            accum[j][i] = 0;
        }
    }
#if MLO_LRN_N_VERT_OUT_PIX > 1
    for(int j = 0; j < MLO_LRN_N_VERT_OUT_PIX - 1; ++j)
    {
        for(int i = 0; i < MLO_LRN_N_HORIZ_OUT_PIX; ++i)
        {
            partial_sum_xy[j][i] = 0;
        }
    }
#endif

    // running window  summation
    FLOAT mov_accum;
    int jj = 0;
    int ii = 0;

    // first to get vertica partial sums

#if MLO_LRN_N_VERT_OUT_PIX > 1
    for(; jj < (int)(MLO_LRN_N_VERT_OUT_PIX - 1); ++jj)
    {
        for(ii = 0; ii < (int)(MLO_LRN_N_HORIZ_OUT_PIX - 1); ++ii)
        {

            FLOAT bot_val = bot_data[lcl_off + jj * MLO_LRN_LCL_DATA_WIDTH + ii];

            FLOAT accum_tmp = bot_val * bot_val;

#if MLO_LRN_N_HORIZ_OUT_PIX > 1
            // save horizontal partial sums
            partial_sum_x[ii] = accum_tmp;
#endif
            // accumulate in vert-horizontal(0)
            partial_sum_xy[jj][0] += accum_tmp;
        }

        for(; ii < (int)MLO_LRN_KERNEL_SZ0; ++ii)
        {

            FLOAT bot_val   = bot_data[lcl_off + jj * MLO_LRN_LCL_DATA_WIDTH + ii];
            FLOAT accum_tmp = bot_val * bot_val;
            // accumulate in vert horizontal(0)
            partial_sum_xy[jj][0] += accum_tmp;
        }

        // running horizontal window

        for(; ii < (int)(MLO_LRN_KERNEL_SZ0 + MLO_LRN_N_HORIZ_OUT_PIX - 1); ++ii)
        {

            FLOAT bot_val   = bot_data[lcl_off + jj * MLO_LRN_LCL_DATA_WIDTH + ii];
            FLOAT accum_tmp = bot_val * bot_val;
            // calculate all vertical-horizontal partial sums
            partial_sum_xy[jj][ii - MLO_LRN_KERNEL_SZ0 + 1] =
                partial_sum_xy[jj][ii - MLO_LRN_KERNEL_SZ0] +
                (accum_tmp
#if MLO_LRN_N_HORIZ_OUT_PIX > 1
                 - partial_sum_x[ii - MLO_LRN_KERNEL_SZ0]
#endif
                );
        }

        // put into accumulator[0][i]
        // whatever has been accumulated so far
        for(int i = 0; i < MLO_LRN_N_HORIZ_OUT_PIX; ++i)
        {
            accum[0][i] += partial_sum_xy[jj][i];
        }
    }
#endif

    // calculate row 0 accumulators
    for(; jj < (int)MLO_LRN_KERNEL_SZ1; ++jj)
    {
        mov_accum = 0;

        for(ii = 0; ii < (int)(MLO_LRN_N_HORIZ_OUT_PIX - 1); ++ii)
        {

            FLOAT bot_val   = bot_data[lcl_off + jj * MLO_LRN_LCL_DATA_WIDTH + ii];
            FLOAT accum_tmp = bot_val * bot_val;
#if MLO_LRN_N_HORIZ_OUT_PIX > 1
            partial_sum_x[ii] = accum_tmp;
#endif
            mov_accum += accum_tmp;
        }

        for(; ii < (int)MLO_LRN_KERNEL_SZ0; ++ii)
        {

            FLOAT bot_val   = bot_data[lcl_off + jj * MLO_LRN_LCL_DATA_WIDTH + ii];
            FLOAT accum_tmp = bot_val * bot_val;
            mov_accum += accum_tmp;
        }

        accum[0][0] += mov_accum;
        // running horizontal window

        for(; ii < (int)(MLO_LRN_KERNEL_SZ0 + MLO_LRN_N_HORIZ_OUT_PIX - 1); ++ii)
        {

            FLOAT bot_val   = bot_data[lcl_off + jj * MLO_LRN_LCL_DATA_WIDTH + ii];
            FLOAT accum_tmp = bot_val * bot_val;
            // running horizontal window
            mov_accum += (accum_tmp
#if MLO_LRN_N_HORIZ_OUT_PIX > 1
                          - partial_sum_x[ii - MLO_LRN_KERNEL_SZ0]
#endif
            );
            accum[0][ii - MLO_LRN_KERNEL_SZ0 + 1] += mov_accum;
        }
    }

    // accumulate all other rows besides 0
    for(; jj < (int)(MLO_LRN_KERNEL_SZ1 + MLO_LRN_N_VERT_OUT_PIX - 1); ++jj)
    {
        // first running horizontal winodw as before
        mov_accum = 0;
        for(ii = 0; ii < (int)(MLO_LRN_N_HORIZ_OUT_PIX - 1); ++ii)
        {

            FLOAT bot_val   = bot_data[lcl_off + jj * MLO_LRN_LCL_DATA_WIDTH + ii];
            FLOAT accum_tmp = bot_val * bot_val;
#if MLO_LRN_N_HORIZ_OUT_PIX > 1
            partial_sum_x[ii] = accum_tmp;
#endif
            accum[jj - MLO_LRN_KERNEL_SZ1 + 1][0] += accum_tmp;
        }
        for(; ii < (int)MLO_LRN_KERNEL_SZ0; ++ii)
        {

            FLOAT bot_val   = bot_data[lcl_off + jj * MLO_LRN_LCL_DATA_WIDTH + ii];
            FLOAT accum_tmp = bot_val * bot_val;
            accum[jj - MLO_LRN_KERNEL_SZ1 + 1][0] += accum_tmp;
        }
        // running horizontal window

        int ii1 = ii;
        for(; ii < (int)(MLO_LRN_KERNEL_SZ0 + MLO_LRN_N_HORIZ_OUT_PIX - 1); ++ii)
        {

            FLOAT bot_val   = bot_data[lcl_off + jj * MLO_LRN_LCL_DATA_WIDTH + ii];
            FLOAT accum_tmp = bot_val * bot_val;
            //
            accum[jj - MLO_LRN_KERNEL_SZ1 + 1][ii - MLO_LRN_KERNEL_SZ0 + 1] =
                accum[jj - MLO_LRN_KERNEL_SZ1 + 1][ii - MLO_LRN_KERNEL_SZ0] + accum_tmp;
#if MLO_LRN_N_HORIZ_OUT_PIX > 1
            accum[jj - MLO_LRN_KERNEL_SZ1 + 1][ii - MLO_LRN_KERNEL_SZ0 + 1] -=
                partial_sum_x[ii - MLO_LRN_KERNEL_SZ0];
#endif
        }

        // finally running vertical window

        for(ii = ii1; ii < (int)(MLO_LRN_KERNEL_SZ0 + MLO_LRN_N_HORIZ_OUT_PIX - 1); ++ii)
        {

            // finish horizontal summation
            // add/substarct vertical patial sum
            accum[jj - MLO_LRN_KERNEL_SZ1 + 1][ii - MLO_LRN_KERNEL_SZ0 + 1] +=
                accum[jj - MLO_LRN_KERNEL_SZ1][ii - MLO_LRN_KERNEL_SZ0 + 1];
#if MLO_LRN_N_VERT_OUT_PIX > 1
            accum[jj - MLO_LRN_KERNEL_SZ1 + 1][ii - MLO_LRN_KERNEL_SZ0 + 1] -=
                partial_sum_xy[jj - MLO_LRN_KERNEL_SZ1][ii - MLO_LRN_KERNEL_SZ0 + 1];
#endif
        }
#if MLO_LRN_N_VERT_OUT_PIX > 1
        accum[jj - MLO_LRN_KERNEL_SZ1 + 1][0] -= partial_sum_xy[jj - MLO_LRN_KERNEL_SZ1][0];
#endif
        accum[jj - MLO_LRN_KERNEL_SZ1 + 1][0] += accum[jj - MLO_LRN_KERNEL_SZ1][0];
    }

    // normalization
    FLOAT prv_scale[MLO_LRN_N_VERT_OUT_PIX][MLO_LRN_N_HORIZ_OUT_PIX];
    FLOAT adj_alphaoverarea = alphaoverarea;
    for(int k = 0; k < MLO_LRN_N_VERT_OUT_PIX; k++)
    {

        //			int hstart = y + lcl_id1 * MLO_LRN_N_VERT_OUT_PIX  + k -
        // MLO_LRN_PAD1;
        //			int hend = min(hstart + MLO_LRN_KERNEL_SZ, MLO_LRN_BOT_HEIGHT +
        // MLO_LRN_PAD1);

        for(int l = 0; l < MLO_LRN_N_HORIZ_OUT_PIX; l++)
        {

            //				int wstart = x + lcl_id0 * MLO_LRN_N_HORIZ_OUT_PIX + l -
            // MLO_LRN_PAD0;
            //				int wend = min(wstart + MLO_LRN_KERNEL_SZ, MLO_LRN_BOT_WIDTH
            //+
            // MLO_LRN_PAD0);
            //				int adj_area_size = (hend - hstart) * (wend - wstart);
            //				adj_alphaoverarea = alpha / adj_area_size;

            prv_scale[k][l] = K + accum[k][l] * adj_alphaoverarea;
        }
    }

    int top_off = b * MLO_LRN_TOP_BATCH_STRIDE + o * MLO_LRN_TOP_CHANNEL_STRIDE +
                  top_y * MLO_LRN_TOP_STRIDE + top_x;
#if MLO_LRN_DO_SCALE
    int scale_off = b * MLO_LRN_SCALE_BATCH_STRIDE + o * MLO_LRN_SCALE_CHANNEL_STRIDE +
                    top_y * MLO_LRN_SCALE_STRIDE + top_x;
#endif

    // final output

    for(int k = 0; k < MLO_LRN_N_VERT_OUT_PIX
#if MLO_OUT_VERT_ALIGNED == 0
                   && (top_y + k < MLO_LRN_TOP_HEIGHT)
#endif
            ;
        k++)
    {
        for(int l = 0; l < MLO_LRN_N_HORIZ_OUT_PIX
#if MLO_OUT_HORIZ_ALIGNED == 0
                       && (top_x + l < MLO_LRN_TOP_WIDTH)
#endif
                ;
            l++)
        {
            FLOAT s;
            s = expf(static_cast<float>(static_cast<FLOAT>(-beta) * static_cast<FLOAT>(logf(static_cast<float>(prv_scale[k][l])))));
            //					s = pow(prv_scale[k][l], -beta);
            FLOAT bot_val = bot_data[lcl_off + (k + MLO_LRN_PRE_PAD1) *
                                                      (int)MLO_LRN_LCL_DATA_WIDTH +
                                                      (l + MLO_LRN_PRE_PAD0)];
#if MLO_LRN_DO_SCALE
            scale[scale_off + k * MLO_LRN_SCALE_STRIDE + l] = prv_scale[k][l];
#endif
            top[top_off + k * MLO_LRN_TOP_STRIDE + l] = bot_val * s;
        }
    }
}

#if(MLO_LRN_N_INPUTS < MLO_LRN_KERNEL_SZ)
#define MLO_LOW_CHNL_COUNT 1
#else
#define MLO_LOW_CHNL_COUNT 0
#endif
extern "C" __global__ __launch_bounds__(MLO_LRN_GROUP_SZ0 * MLO_LRN_GROUP_SZ1 * MLO_LRN_GROUP_SZ2) void
MIOpenLRNAcrossChannels4(const FLOAT* bottom,
                         FLOAT* top,
#if MLO_LRN_DO_SCALE
                         FLOAT* scale,
#endif
                         FLOAT alphaoverarea,
                         [[maybe_unused]] FLOAT alpha,
                         FLOAT beta,
                         FLOAT K)
{
    int pix_id          = blockIdx.x * blockDim.x + threadIdx.x; //
    int b               = blockIdx.z; // batch
    MLO_READ_TYPE accum = MLO_MAKE_VEC(0.0f);
    MLO_READ_TYPE bot_in2[MLO_LRN_KERNEL_SZ];
    MLO_READ_TYPE bot_in[MLO_LRN_KERNEL_SZ];
    int c_i = 0, c_o = 0;
    for(int i = 0; i < MLO_LRN_KERNEL_SZ; ++i)
    {
        bot_in2[i] = MLO_MAKE_VEC(0.0f);
        bot_in[i]  = MLO_MAKE_VEC(0.0f);
    }

    int top_off = 0;
#if MLO_LRN_DO_SCALE
    int scale_off;
#endif

    for(c_i = 0; c_i < MLO_LRN_PAD; c_i++)
    {
        MLO_READ_TYPE prv_in;
        prv_in = MLO_MAKE_VEC(0.0f);

#if MLO_LOW_CHNL_COUNT == 1
        if(c_i < MLO_LRN_N_INPUTS)
#endif
        {
#if MLO_C1x1_PIXLEFT > 0
            // if the last one
            if(pix_id == MLO_MAP_SZ4 - 1)
            {

                for(int j = 0; j < MLO_C1x1_PIXLEFT; ++j)
                {
                    ((FLOAT*)&prv_in)[j] =
                        bottom[MLO_LRN_BOT_BATCH_STRIDE * b + MLO_LRN_BOT_CHANNEL_STRIDE * c_i +
                               (pix_id * MLO_READ_UNIT) + j];
                }
            }
            else
#endif
            {
                prv_in = *(const MLO_READ_TYPE*)&bottom[MLO_LRN_BOT_BATCH_STRIDE * b +
                                                        MLO_LRN_BOT_CHANNEL_STRIDE * c_i +
                                                        (pix_id * MLO_READ_UNIT)];
            }
        }

        bot_in2[c_i] = prv_in * prv_in;
        bot_in[c_i]  = prv_in;
        accum        = accum + bot_in2[c_i];
        //				fma(bot_in2[c_i + MLO_LRN_PAD], bot_in2[c_i + MLO_LRN_PAD],
        // accum);
    }

    for(; c_i < MLO_LRN_KERNEL_SZ; c_i++, c_o++)
    {
        MLO_READ_TYPE prv_in;
        prv_in = MLO_MAKE_VEC(0.0f);

#if MLO_LOW_CHNL_COUNT == 1
        if(c_i < MLO_LRN_N_INPUTS)
#endif
        {

#if MLO_C1x1_PIXLEFT > 0
            // if the last one
            if(pix_id == MLO_MAP_SZ4 - 1)
            {

                for(int j = 0; j < MLO_C1x1_PIXLEFT; ++j)
                {
                    ((FLOAT*)&prv_in)[j] =
                        bottom[MLO_LRN_BOT_BATCH_STRIDE * b + MLO_LRN_BOT_CHANNEL_STRIDE * c_i +
                               (pix_id * MLO_READ_UNIT) + j];
                }
            }
            else
#endif
            {
                prv_in = *(const MLO_READ_TYPE*)&bottom[MLO_LRN_BOT_BATCH_STRIDE * b +
                                                        MLO_LRN_BOT_CHANNEL_STRIDE * c_i +
                                                        (pix_id * MLO_READ_UNIT)];
            }
        }

        bot_in2[c_i] = prv_in * prv_in;
        bot_in[c_i]  = prv_in;
        accum        = accum + bot_in2[c_i];

        top_off = b * MLO_LRN_TOP_BATCH_STRIDE + c_o * MLO_LRN_TOP_CHANNEL_STRIDE +
                  (pix_id * MLO_READ_UNIT);
#if MLO_LRN_DO_SCALE
        scale_off = b * MLO_LRN_SCALE_BATCH_STRIDE + c_o * MLO_LRN_SCALE_CHANNEL_STRIDE +
                    (pix_id * MLO_READ_UNIT);
#endif
        MLO_READ_TYPE prv_scale = (MLO_MAKE_VEC(K) + accum * MLO_MAKE_VEC(alphaoverarea));
        //				fma(accum,alphaoverarea, (FLOAT)1.f);

        MLO_READ_TYPE exp_scale = MLO_VEC_EXP(MLO_MAKE_VEC(-beta) * MLO_VEC_LOG(prv_scale));
        //				pow(prv_scale,-beta);
        // bug
        //	MLO_READ_TYPE prv_out = sqrt(bot_in2[c_o]);
        MLO_READ_TYPE prv_out = bot_in[c_o];
        MLO_READ_TYPE out_val = prv_out * exp_scale;
#if MLO_LOW_CHNL_COUNT == 1
        if(c_o < MLO_LRN_N_OUTPUTS)
#endif
        {

#if MLO_C1x1_PIXLEFT > 0

            // if the last one
            if(pix_id == MLO_MAP_SZ4 - 1)
            {
                for(int j = 0; j < MLO_C1x1_PIXLEFT; ++j)
                {
                    top[top_off + j] = ((FLOAT*)&out_val)[j];
#if DBG_OUT
                    printf("K:o0: %d %f %f %f %f %f\n",
                           top_off + j,
                           (float)top[top_off + j],
                           (float)((FLOAT*)&prv_out)[j],
                           (float)((FLOAT*)&exp_scale)[j],
                           (float)((FLOAT*)&prv_scale)[j],
                           (float)((FLOAT*)&accum)[j]);
#endif

#if MLO_LRN_DO_SCALE
                    scale[scale_off + j] = ((FLOAT*)&prv_scale)[j];
#endif
                }
            }
            else
#endif
            {

                *((MLO_READ_TYPE*)&top[top_off]) = out_val;
#if MLO_LRN_DO_SCALE
                *((MLO_READ_TYPE*)&scale[scale_off]) = prv_scale;
#endif
            }
        }
    }

    for(; c_i < MLO_LRN_N_INPUTS; c_i++, c_o++)
    {

        MLO_READ_TYPE prv_in;
        prv_in = MLO_MAKE_VEC(0.0f);

#if MLO_C1x1_PIXLEFT > 0
        // if the last one
        if(pix_id == MLO_MAP_SZ4 - 1)
        {

            for(int j = 0; j < MLO_C1x1_PIXLEFT; ++j)
            {
                ((FLOAT*)&prv_in)[j] =
                    bottom[MLO_LRN_BOT_BATCH_STRIDE * b + MLO_LRN_BOT_CHANNEL_STRIDE * c_i +
                           (pix_id * MLO_READ_UNIT) + j];
            }
        }
        else
#endif
        {
            prv_in = *(const MLO_READ_TYPE*)&bottom[MLO_LRN_BOT_BATCH_STRIDE * b +
                                                     MLO_LRN_BOT_CHANNEL_STRIDE * c_i +
                                                     (pix_id * MLO_READ_UNIT)];
        }

        MLO_READ_TYPE prv_bot_in2 = prv_in * prv_in;
        accum                     = accum + prv_bot_in2;

        accum = accum - bot_in2[0];
        //				fma(-bot_in2[0], bot_in2[0], accum);

        for(int i = 0; i < MLO_LRN_KERNEL_SZ - 1; i++)
        {
            bot_in2[i] = bot_in2[i + 1];
            bot_in[i]  = bot_in[i + 1];
        }

        bot_in2[MLO_LRN_KERNEL_SZ - 1] = prv_bot_in2;
        bot_in[MLO_LRN_KERNEL_SZ - 1]  = prv_in;

        top_off = b * MLO_LRN_TOP_BATCH_STRIDE + c_o * MLO_LRN_TOP_CHANNEL_STRIDE +
                  (pix_id * MLO_READ_UNIT);
#if MLO_LRN_DO_SCALE
        scale_off = b * MLO_LRN_SCALE_BATCH_STRIDE + c_o * MLO_LRN_SCALE_CHANNEL_STRIDE +
                    (pix_id * MLO_READ_UNIT);
#endif
        MLO_READ_TYPE prv_scale = (MLO_MAKE_VEC(K) + accum * MLO_MAKE_VEC(alphaoverarea));
        //				fma(accum,alphaoverarea, (FLOAT)1.f);

        MLO_READ_TYPE exp_scale = MLO_VEC_EXP(MLO_MAKE_VEC(-beta) * MLO_VEC_LOG(prv_scale));
        //				pow(prv_scale,-beta);
        // bug
        //			MLO_READ_TYPE prv_out = sqrt(bot_in2[MLO_LRN_PRE_PAD]);
        MLO_READ_TYPE prv_out = bot_in[MLO_LRN_PRE_PAD];
        MLO_READ_TYPE out_val = prv_out * exp_scale;

#if MLO_LOW_CHNL_COUNT == 1
        if(c_o < MLO_LRN_N_OUTPUTS)
#endif
        {

#if MLO_C1x1_PIXLEFT > 0

            // if the last one
            if(pix_id == MLO_MAP_SZ4 - 1)
            {
                for(int j = 0; j < MLO_C1x1_PIXLEFT; ++j)
                {
                    top[top_off + j] = ((FLOAT*)&out_val)[j];
#if DBG_OUT
                    printf("K:o1: %d %f %f %f\n",
                           top_off + j,
                           (float)top[top_off + j],
                           (float)((FLOAT*)&prv_out)[j],
                           (float)((FLOAT*)&exp_scale)[j]);
#endif

#if MLO_LRN_DO_SCALE
                    scale[scale_off + j] = ((FLOAT*)&prv_scale)[j];
#endif
                }
            }
            else
#endif
            {

                *((MLO_READ_TYPE*)&top[top_off]) = out_val;
#if MLO_LRN_DO_SCALE
                *((MLO_READ_TYPE*)&scale[scale_off]) = prv_scale;
#endif
            }
        }
    }

    for(; c_i < MLO_LRN_N_INPUTS + MLO_LRN_PAD; c_i++, c_o++)
    {

        accum = accum - bot_in2[0];
        //				fma(-bot_in2[0], bot_in2[0], accum);

        for(int i = 0; i < MLO_LRN_KERNEL_SZ - 1; i++)
        {
            bot_in2[i] = bot_in2[i + 1];
            bot_in[i]  = bot_in[i + 1];
        }

        top_off = b * MLO_LRN_TOP_BATCH_STRIDE + c_o * MLO_LRN_TOP_CHANNEL_STRIDE +
                  (pix_id * MLO_READ_UNIT);
#if MLO_LRN_DO_SCALE
        scale_off = b * MLO_LRN_SCALE_BATCH_STRIDE + c_o * MLO_LRN_SCALE_CHANNEL_STRIDE +
                    (pix_id * MLO_READ_UNIT);
#endif
        MLO_READ_TYPE prv_scale = (MLO_MAKE_VEC(K) + accum * MLO_MAKE_VEC(alphaoverarea));
        //				fma(accum,alphaoverarea, (FLOAT)1.f);

        MLO_READ_TYPE exp_scale = MLO_VEC_EXP(MLO_MAKE_VEC(-beta) * MLO_VEC_LOG(prv_scale));
        //				pow(prv_scale,-beta);
        // bug
        //			MLO_READ_TYPE prv_out = sqrt(bot_in2[MLO_LRN_PRE_PAD]);
        MLO_READ_TYPE prv_out = bot_in[MLO_LRN_PRE_PAD];

        MLO_READ_TYPE out_val = prv_out * exp_scale;
#if MLO_LOW_CHNL_COUNT == 1
        if(c_o < MLO_LRN_N_OUTPUTS)
#endif
        {

#if MLO_C1x1_PIXLEFT > 0

            // if the last one
            if(pix_id == MLO_MAP_SZ4 - 1)
            {
                for(int j = 0; j < MLO_C1x1_PIXLEFT; ++j)
                {
                    top[top_off + j] = ((FLOAT*)&out_val)[j];
#if DBG_OUT
                    printf("K:o2: %d %f %f %f\n",
                           top_off + j,
                           (float)top[top_off + j],
                           (float)((FLOAT*)&prv_out)[j],
                           (float)((FLOAT*)&exp_scale)[j]);
#endif

#if MLO_LRN_DO_SCALE
                    scale[scale_off + j] = ((FLOAT*)&prv_scale)[j];
#endif
                }
            }
            else
#endif
            {

                *((MLO_READ_TYPE*)&top[top_off]) = out_val;
#if MLO_LRN_DO_SCALE
                *((MLO_READ_TYPE*)&scale[scale_off]) = prv_scale;
#endif
            }
        }
    }
}
