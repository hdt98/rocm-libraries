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

#define DBG_RANGE 0

#define MLO_LRN_GROUP_SZ2 1
#define MLO_LRN_STRIDE 1

#define MLO_LRN_LCL_DATA_WIDTH (MLO_LRN_GROUP_SZ0 * MLO_LRN_N_HORIZ_OUT_PIX + MLO_LRN_KERNEL_SZ - 1)
#define MLO_LRN_LCL_DATA_HEIGHT (MLO_LRN_GROUP_SZ1 * MLO_LRN_N_VERT_OUT_PIX + MLO_LRN_KERNEL_SZ - 1)
#define MLO_LRN_GROUP_SZ (MLO_LRN_GROUP_SZ2 * MLO_LRN_GROUP_SZ1 * MLO_LRN_GROUP_SZ0)

struct LRNForwardParam
{
    FLOAT alphaoverarea;
    FLOAT alpha;
    FLOAT beta;
    FLOAT K;
};

struct LRNBackwardParam
{
    FLOAT ratio;
    FLOAT alpha;
    FLOAT beta;
};

/*

This is a naive implementation.
The "sliding window" -based implementation is in MIOpenLRNFwd.cpp file

*/

extern "C" __global__ __launch_bounds__(MLO_LRN_GROUP_SZ0 * MLO_LRN_GROUP_SZ1 * MLO_LRN_GROUP_SZ2) void
MIOpenLRNWithinChannelBwd(const FLOAT* top,
                          const FLOAT* bot,
                          const FLOAT* top_df,
                          const FLOAT* scale,
                          FLOAT* bot_df,
                          [[maybe_unused]] FLOAT ratio,
                          FLOAT alpha,
                          FLOAT beta)
{
    __shared__ FLOAT top_df_data[MLO_LRN_LCL_DATA_WIDTH * MLO_LRN_LCL_DATA_HEIGHT];
    __shared__ FLOAT ratio_data[MLO_LRN_LCL_DATA_WIDTH * MLO_LRN_LCL_DATA_HEIGHT];
    int x          = blockIdx.x * MLO_LRN_GROUP_SZ0 * MLO_LRN_N_HORIZ_OUT_PIX;
    int y          = blockIdx.y * MLO_LRN_GROUP_SZ1 * MLO_LRN_N_VERT_OUT_PIX;
    int lcl_id0    = threadIdx.x;
    int lcl_id1    = threadIdx.y;
    int ob         = blockIdx.z; // output * batch_sz
    int o          = ob / MLO_LRN_BATCH_SZ;
    int b          = ob - o * MLO_LRN_BATCH_SZ;
    int top_x      = x;
    int top_y      = y;
    int top_df_off = b * MLO_LRN_TOPDF_BATCH_STRIDE + o * MLO_LRN_TOPDF_CHANNEL_STRIDE;
    int scale_off  = b * MLO_LRN_SCALE_BATCH_STRIDE + o * MLO_LRN_SCALE_CHANNEL_STRIDE;
    int bot_x      = x + lcl_id0 * MLO_LRN_N_HORIZ_OUT_PIX;
    int bot_y      = y + lcl_id1 * MLO_LRN_N_VERT_OUT_PIX;

    FLOAT prv_exp_scale[MLO_LRN_N_VERT_OUT_PIX][MLO_LRN_N_HORIZ_OUT_PIX];
    //		FLOAT prv_top_df[MLO_LRN_N_VERT_OUT_PIX][MLO_LRN_N_HORIZ_OUT_PIX];

    // load top_diff and scale tiles
    for(int b_j = lcl_id1; b_j < MLO_LRN_LCL_DATA_HEIGHT; b_j += MLO_LRN_GROUP_SZ1)
    {
        int top_y_act = top_y + b_j - MLO_LRN_PAD;

        bool invisibleY = (top_y_act < 0) || (top_y_act >= MLO_LRN_TOP_HEIGHT);

        top_y_act = (invisibleY) ? 0 : top_y_act;

        int top_df_y_off = top_y_act * MLO_LRN_TOPDF_STRIDE;
        int scale_y_off  = top_y_act * MLO_LRN_SCALE_STRIDE;

        int lcl_off_v = b_j * MLO_LRN_LCL_DATA_WIDTH;

        for(int b_i = lcl_id0; b_i < MLO_LRN_LCL_DATA_WIDTH; b_i += MLO_LRN_GROUP_SZ0)
        {

            int top_x_act = top_x + b_i - MLO_LRN_PAD;

            bool invisibleX = (top_x_act < 0) || (top_x_act >= MLO_LRN_TOP_WIDTH);

            top_x_act = (invisibleX) ? 0 : top_x_act;
#if DBG_RANGE
            if(top_df_off + top_df_y_off + top_x_act >=
               MLO_LRN_BATCH_SZ * MLO_LRN_TOPDF_BATCH_STRIDE)
            {
                printf("K:err:topdf-off_range\n");
            }
#endif
            FLOAT top_df_val = top_df[top_df_off + top_df_y_off + top_x_act];
            FLOAT scale_val  = scale[scale_off + scale_y_off + top_x_act];

            top_df_val = (invisibleX || invisibleY) ? static_cast<FLOAT>(0) : top_df_val;
            scale_val  = (invisibleX || invisibleY) ? static_cast<FLOAT>(1.0f) : scale_val;

            top_df_data[lcl_off_v + b_i] = top_df_val;
            ratio_data[lcl_off_v + b_i]  = scale_val;
        }
    }

    __syncthreads();

    // actual top_diffs and scales
    for(int j = 0; j < MLO_LRN_N_VERT_OUT_PIX; ++j)
    {
        int lcl_off_v =
            (lcl_id1 * MLO_LRN_N_VERT_OUT_PIX + MLO_LRN_PAD + j) * MLO_LRN_LCL_DATA_WIDTH;
        for(int i = 0; i < MLO_LRN_N_HORIZ_OUT_PIX; i++)
        {
            FLOAT scale_ratio =
                ratio_data[lcl_off_v + lcl_id0 * MLO_LRN_N_HORIZ_OUT_PIX + MLO_LRN_PAD + i];
            prv_exp_scale[j][i] = expf(static_cast<float>(-beta * static_cast<FLOAT>(logf(static_cast<float>(scale_ratio)))));
        }
    }

    __syncthreads();
    // read top and load ratio tile
    int top_off = b * MLO_LRN_TOP_BATCH_STRIDE + o * MLO_LRN_TOP_CHANNEL_STRIDE;
    for(int b_j = lcl_id1; b_j < MLO_LRN_LCL_DATA_HEIGHT; b_j += MLO_LRN_GROUP_SZ1)
    {
        int top_y_act = top_y + b_j - MLO_LRN_PAD;

        bool invisibleY = (top_y_act < 0) || (top_y_act >= MLO_LRN_TOP_HEIGHT);

        top_y_act = (invisibleY) ? 0 : top_y_act;

        int top_y_off = top_y_act * MLO_LRN_TOP_STRIDE;

        int lcl_off_v = b_j * MLO_LRN_LCL_DATA_WIDTH;

        for(int b_i = lcl_id0; b_i < MLO_LRN_LCL_DATA_WIDTH; b_i += MLO_LRN_GROUP_SZ0)
        {

            int top_x_act = top_x + b_i - MLO_LRN_PAD;

            bool invisibleX = (top_x_act < 0) || (top_x_act >= MLO_LRN_TOP_WIDTH);

            top_x_act = (invisibleX) ? 0 : top_x_act;
#if DBG_RANGE

            if(top_off + top_y_off + top_x_act >= MLO_LRN_BATCH_SZ * MLO_LRN_TOP_BATCH_STRIDE)
            {
                printf("K:err:top-off_range\n");
            }
#endif

            FLOAT top_val = top[top_off + top_y_off + top_x_act];

            top_val = (invisibleX || invisibleY) ? static_cast<FLOAT>(0) : top_val;

            FLOAT top_df_val = top_df_data[lcl_off_v + b_i];

            FLOAT scale_val = ratio_data[lcl_off_v + b_i];

            // scale val is not 0
            FLOAT ratio_dta = (top_df_val * top_val) / scale_val;
            // replacing scale with ratio
            ratio_data[lcl_off_v + b_i] = ratio_dta;
        }
    }

    __syncthreads();

    // caculate bot diff
    FLOAT prv_bot_diff[MLO_LRN_N_VERT_OUT_PIX][MLO_LRN_N_HORIZ_OUT_PIX];

    for(int j = 0; j < MLO_LRN_N_VERT_OUT_PIX; ++j)
    {
        int v_off_v = (lcl_id1 * MLO_LRN_N_VERT_OUT_PIX + j);
        int hstart  = y + v_off_v - MLO_LRN_PAD;
        int hend    = min(hstart + MLO_LRN_KERNEL_SZ, MLO_LRN_TOP_HEIGHT + MLO_LRN_PRE_PAD);

        // accum offset, vertical
        //			int lcl_a_off_v = v_off_v *  MLO_LRN_LCL_DATA_WIDTH;
        // value offset, vertical
        int lcl_v_off_v = (v_off_v + MLO_LRN_PAD) * MLO_LRN_LCL_DATA_WIDTH;
        for(int i = 0; i < MLO_LRN_N_HORIZ_OUT_PIX; i++)
        {
            FLOAT prv_ratio_accum = static_cast<FLOAT>(0);
            int v_off_h            = lcl_id0 * MLO_LRN_N_HORIZ_OUT_PIX + i;

            int wstart = x + v_off_h - MLO_LRN_PAD;
            int wend   = min(wstart + MLO_LRN_KERNEL_SZ, MLO_LRN_TOP_WIDTH + MLO_LRN_PRE_PAD);

            int adj_area_size = (hend - hstart) * (wend - wstart);

            // accum offset, horiz
            int lcl_a_off_h = v_off_h;
            //	value offset, horiz
            int lcl_v_off_h = lcl_a_off_h + MLO_LRN_PAD;

            for(int k = 0; k < MLO_LRN_KERNEL_SZ; k++)
            {
                for(int l = 0; l < MLO_LRN_KERNEL_SZ; l++)
                {
                    prv_ratio_accum +=
                        ratio_data[(v_off_v + k) * MLO_LRN_LCL_DATA_WIDTH + lcl_a_off_h + l];
                }
            }

            FLOAT top_df_val = top_df_data[lcl_v_off_v + lcl_v_off_h];

            unsigned int bot_off0 = MLO_LRN_BOT_BATCH_STRIDE * b + MLO_LRN_BOT_CHANNEL_STRIDE * o +
                            MLO_LRN_BOT_STRIDE * (y + v_off_v) + x + v_off_h;

            unsigned int bot_off = (y + v_off_v < MLO_LRN_BOT_HEIGHT && x + v_off_h < MLO_LRN_BOT_WIDTH &&
                            b < MLO_LRN_BATCH_SZ && o < MLO_LRN_N_OUTPUTS)
                               ? bot_off0
                               : MLO_LRN_BATCH_SZ * MLO_LRN_BOT_BATCH_STRIDE - 1;
#if DBG_RANGE

            if(bot_off >= (unsigned int)(MLO_LRN_BATCH_SZ * MLO_LRN_BOT_BATCH_STRIDE))
            {
                printf("K:err:bot-off_range\n");
            }
#endif
            FLOAT bot_dta = bot[bot_off];

            bot_dta = (y + v_off_v < MLO_LRN_BOT_HEIGHT && x + v_off_h < MLO_LRN_BOT_WIDTH &&
                       b < MLO_LRN_BATCH_SZ && o < MLO_LRN_N_OUTPUTS)
                          ? bot_dta
                          : static_cast<FLOAT>(0);

            FLOAT adj_ratio       = static_cast<FLOAT>(2.0f) * alpha * beta / adj_area_size;
            FLOAT prv_accum_ratio = adj_ratio * bot_dta * prv_ratio_accum;
            prv_bot_diff[j][i]     = prv_exp_scale[j][i] * top_df_val - prv_accum_ratio;
        }
    }

    for(int j = 0; j < MLO_LRN_N_VERT_OUT_PIX; j++)
    {
        for(int i = 0; i < MLO_LRN_N_HORIZ_OUT_PIX; i++)
        {
            if(bot_y + j < MLO_LRN_BOT_HEIGHT && bot_x + i < MLO_LRN_BOT_WIDTH &&
               b < MLO_LRN_BATCH_SZ && o < MLO_LRN_N_OUTPUTS)
            {
#if DBG_RANGE

                if(MLO_LRN_BOTDF_BATCH_STRIDE * b + MLO_LRN_BOTDF_CHANNEL_STRIDE * o +
                       MLO_LRN_BOTDF_STRIDE * (bot_y + j) + bot_x + i >=
                   MLO_LRN_BATCH_SZ * MLO_LRN_BOTDF_BATCH_STRIDE)
                {
                    printf("K:err:botdf-off_range\n");
                }
#endif
                bot_df[MLO_LRN_BOTDF_BATCH_STRIDE * b + MLO_LRN_BOTDF_CHANNEL_STRIDE * o +
                       MLO_LRN_BOTDF_STRIDE * (bot_y + j) + bot_x + i] = prv_bot_diff[j][i];
            }
        }
    }
}

#if(MLO_LRN_N_INPUTS < MLO_LRN_KERNEL_SZ)
#define MLO_LOW_CHNL_COUNT 1
#else
#define MLO_LOW_CHNL_COUNT 0
#endif

extern "C" __global__ __launch_bounds__(MLO_LRN_GROUP_SZ0 * MLO_LRN_GROUP_SZ1 * MLO_LRN_GROUP_SZ2) void
MIOpenLRNAcrossChannelsBwd1(const FLOAT* top,
                            const FLOAT* bot,
                            const FLOAT* top_df,
                            const FLOAT* scale,
                            FLOAT* bot_df,
                            FLOAT ratio,
                            [[maybe_unused]] FLOAT alpha,
                            FLOAT beta)
{
    int x              = blockIdx.x * blockDim.x + threadIdx.x; // channel x
    int y              = blockIdx.y * blockDim.y + threadIdx.y; // channel y
    int b              = blockIdx.z; // batch
    FLOAT accum_ratio = static_cast<FLOAT>(0);
    FLOAT top_df_in[MLO_LRN_KERNEL_SZ];
    FLOAT scale_in[MLO_LRN_KERNEL_SZ];
    FLOAT ratio_dta[MLO_LRN_KERNEL_SZ];
    int c_i = 0, c_o = 0;
    int bot_df_off = 0;

    for(c_i = 0; c_i < MLO_LRN_PRE_PAD; c_i++)
    {

        top_df_in[c_i] = top_df[MLO_LRN_TOPDF_BATCH_STRIDE * b +
                                MLO_LRN_TOPDF_CHANNEL_STRIDE * c_i + MLO_LRN_TOPDF_STRIDE * y + x];
        scale_in[c_i]  = scale[MLO_LRN_SCALE_BATCH_STRIDE * b + MLO_LRN_SCALE_CHANNEL_STRIDE * c_i +
                              MLO_LRN_SCALE_STRIDE * y + x];
        FLOAT top_dta = top[MLO_LRN_TOP_BATCH_STRIDE * b + MLO_LRN_TOP_CHANNEL_STRIDE * c_i +
                             MLO_LRN_TOP_STRIDE * y + x];

        ratio_dta[c_i] = (top_df_in[c_i] * top_dta) / scale_in[c_i];

#if MLO_LOW_CHNL_COUNT == 1
        ratio_dta[c_i] = (c_i < MLO_LRN_N_OUTPUTS) ? ratio_dta[c_i] : static_cast<FLOAT>(0);
#endif

        accum_ratio = accum_ratio + ratio_dta[c_i];
    }

    for(; c_i < MLO_LRN_KERNEL_SZ; c_i++, c_o++)
    {
        top_df_in[c_i] = top_df[MLO_LRN_TOPDF_BATCH_STRIDE * b +
                                MLO_LRN_TOPDF_CHANNEL_STRIDE * c_i + MLO_LRN_TOPDF_STRIDE * y + x];
        scale_in[c_i]  = scale[MLO_LRN_SCALE_BATCH_STRIDE * b + MLO_LRN_SCALE_CHANNEL_STRIDE * c_i +
                              MLO_LRN_SCALE_STRIDE * y + x];
        FLOAT top_dta = top[MLO_LRN_TOP_BATCH_STRIDE * b + MLO_LRN_TOP_CHANNEL_STRIDE * c_i +
                             MLO_LRN_TOP_STRIDE * y + x];
        ratio_dta[c_i] = (top_df_in[c_i] * top_dta) / scale_in[c_i];
#if MLO_LOW_CHNL_COUNT == 1
        ratio_dta[c_i] = (c_i < MLO_LRN_N_OUTPUTS) ? ratio_dta[c_i] : static_cast<FLOAT>(0);
#endif

        accum_ratio = accum_ratio + ratio_dta[c_i];
#if MLO_LOW_CHNL_COUNT == 1
        if(c_o < MLO_LRN_N_INPUTS)
#endif
        {
            FLOAT bot_dta = bot[MLO_LRN_BOT_BATCH_STRIDE * b + MLO_LRN_BOT_CHANNEL_STRIDE * c_o +
                                 MLO_LRN_BOT_STRIDE * y + x];

            FLOAT prv_scale = scale_in[c_o];

            FLOAT exp_scale = expf(static_cast<float>(-beta * static_cast<FLOAT>(logf(static_cast<float>(prv_scale)))));
            //					pow(prv_scale, -beta);

            FLOAT prv_accum_ratio = ratio * bot_dta * accum_ratio;

            FLOAT out_val = top_df_in[c_o] * exp_scale - prv_accum_ratio;

            bot_df_off = MLO_LRN_BOTDF_BATCH_STRIDE * b + MLO_LRN_BOTDF_CHANNEL_STRIDE * c_o +
                         MLO_LRN_BOTDF_STRIDE * y + x;

            bot_df[bot_df_off] = out_val;
        }
    }

    for(; c_i < MLO_LRN_N_INPUTS; c_i++, c_o++)
    {

        FLOAT prv_top_df_in =
            top_df[MLO_LRN_TOPDF_BATCH_STRIDE * b + MLO_LRN_TOPDF_CHANNEL_STRIDE * c_i +
                   MLO_LRN_TOPDF_STRIDE * y + x];
        FLOAT prv_scale_in =
            scale[MLO_LRN_SCALE_BATCH_STRIDE * b + MLO_LRN_SCALE_CHANNEL_STRIDE * c_i +
                  MLO_LRN_SCALE_STRIDE * y + x];
        FLOAT top_dta       = top[MLO_LRN_TOP_BATCH_STRIDE * b + MLO_LRN_TOP_CHANNEL_STRIDE * c_i +
                             MLO_LRN_TOP_STRIDE * y + x];
        FLOAT prv_ratio_dta = prv_top_df_in * top_dta / prv_scale_in;
#if MLO_LOW_CHNL_COUNT == 1
        prv_ratio_dta = (c_i < MLO_LRN_N_OUTPUTS) ? prv_ratio_dta : static_cast<FLOAT>(0);
#endif

        accum_ratio = accum_ratio + prv_ratio_dta;

        accum_ratio = accum_ratio - ratio_dta[0];

        for(int i = 0; i < MLO_LRN_KERNEL_SZ - 1; i++)
        {
            top_df_in[i] = top_df_in[i + 1];
            scale_in[i]  = scale_in[i + 1];
            ratio_dta[i] = ratio_dta[i + 1];
        }

        top_df_in[MLO_LRN_KERNEL_SZ - 1] = prv_top_df_in;
        scale_in[MLO_LRN_KERNEL_SZ - 1]  = prv_scale_in;
        ratio_dta[MLO_LRN_KERNEL_SZ - 1] = prv_ratio_dta;

#if MLO_LOW_CHNL_COUNT == 1
        if(c_o < MLO_LRN_N_INPUTS)
#endif
        {
            FLOAT bot_dta = bot[MLO_LRN_BOT_BATCH_STRIDE * b + MLO_LRN_BOT_CHANNEL_STRIDE * c_o +
                                 MLO_LRN_BOT_STRIDE * y + x];

            FLOAT prv_scale = scale_in[MLO_LRN_PAD];

            FLOAT exp_scale = expf(static_cast<float>(-beta * static_cast<FLOAT>(logf(static_cast<float>(prv_scale)))));
            //				pow(prv_scale,-beta);

            FLOAT prv_accum_ratio = ratio * bot_dta * accum_ratio;

            FLOAT out_val = top_df_in[MLO_LRN_PAD] * exp_scale - prv_accum_ratio;

            bot_df_off = MLO_LRN_BOTDF_BATCH_STRIDE * b + MLO_LRN_BOTDF_CHANNEL_STRIDE * c_o +
                         MLO_LRN_BOTDF_STRIDE * y + x;

            bot_df[bot_df_off] = out_val;
        }
    }

    for(; c_i < MLO_LRN_N_INPUTS + MLO_LRN_PRE_PAD; c_i++, c_o++)
    {

        accum_ratio = accum_ratio - ratio_dta[0];

        for(int i = 0; i < MLO_LRN_KERNEL_SZ - 1; i++)
        {
            top_df_in[i] = top_df_in[i + 1];
            scale_in[i]  = scale_in[i + 1];
            ratio_dta[i] = ratio_dta[i + 1];
        }

#if MLO_LOW_CHNL_COUNT == 1
        if(c_o < MLO_LRN_N_INPUTS)
#endif
        {
            FLOAT bot_dta = bot[MLO_LRN_BOT_BATCH_STRIDE * b + MLO_LRN_BOT_CHANNEL_STRIDE * c_o +
                                 MLO_LRN_BOT_STRIDE * y + x];

            FLOAT prv_scale = scale_in[MLO_LRN_PAD];

            FLOAT exp_scale = expf(static_cast<float>(-beta * static_cast<FLOAT>(logf(static_cast<float>(prv_scale)))));
            //				pow(prv_scale,-beta);

            FLOAT prv_accum_ratio = ratio * bot_dta * accum_ratio;

            FLOAT out_val = top_df_in[MLO_LRN_PAD] * exp_scale - prv_accum_ratio;

            bot_df_off = MLO_LRN_BOTDF_BATCH_STRIDE * b + MLO_LRN_BOTDF_CHANNEL_STRIDE * c_o +
                         MLO_LRN_BOTDF_STRIDE * y + x;

            bot_df[bot_df_off] = out_val;
        }
    }
}
