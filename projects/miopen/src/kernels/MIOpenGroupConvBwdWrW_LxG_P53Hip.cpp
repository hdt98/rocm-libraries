/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"
#include "hip_math_ops.hpp"

#define UNUSED __attribute__((__unused__))

#define MLO_N_OUT_HORIZ_READS ((MLO_OUT_WIDTH + MLO_IN_TILE0 - 1) / MLO_IN_TILE0)
#define MLO_N_SPANS_PER_SCAN (MLO_N_OUT_HORIZ_READS)
#define MLO_N_OUT_HORIZ_PIX_READS (MLO_N_OUT_HORIZ_READS * MLO_IN_TILE0)
#define MLO_OUT_N_PIXS_OFF (MLO_OUT_WIDTH - ((MLO_OUT_WIDTH / MLO_IN_TILE0) * MLO_IN_TILE0))
#define MLO_N_OUT_VERTICAL_READS (MLO_FILTER_SIZE1)
// won't run non-border blocks if  MLO_IN_N_VERT_LOOPS < 2
//

#define MLO_IN_VERT_READS MLO_IN_EXTENT1

#if MLO_IN_N_VERT_LOOPS >= 2
#define MLO_N_GENERIC_LOOPS (MLO_IN_N_VERT_LOOPS - 2)
#else
#define MLO_N_GENERIC_LOOPS 0
#endif

// there is an assumption that the scanline fits into LDS
#define MLO_N_IN_HORIZ_PIX_READS (MLO_IN_WIDTH)
#define MLO_N_IN_HORIZ_READS ((MLO_N_IN_HORIZ_PIX_READS + MLO_READ_UNIT - 1) / MLO_READ_UNIT)
#define MLO_IN_N_PIXS_OFF \
    (MLO_N_IN_HORIZ_PIX_READS - (MLO_N_IN_HORIZ_PIX_READS / MLO_READ_UNIT) * MLO_READ_UNIT)
#define MLO_IN_LCL_WIDTH (MLO_N_IN_HORIZ_READS * MLO_READ_UNIT + 2 * MLO_FILTER_PAD0)
#define MLO_IN_LCL_HEIGHT MLO_IN_VERT_READS
#define MLO_IN_LCL_SZ (MLO_IN_LCL_WIDTH * MLO_IN_LCL_HEIGHT)
#define MLO_TOTAL_IN_LCL_SZ (MLO_N_LCL_BATCHS * MLO_N_LCL_IN_MAPS * MLO_IN_LCL_SZ)

#define MLO_WEI_LCL_SZ (MLO_GRP_SZ * MLO_FILTER_SIZE0)
#if MLO_TOTAL_IN_LCL_SZ > MLO_WEI_LCL_SZ
#define MLO_LCL_SZ (MLO_TOTAL_IN_LCL_SZ)
#else
#define MLO_LCL_SZ (MLO_WEI_LCL_SZ)
#endif

// if to read all of the number of MLO_N_LCL_IN_MAPS input channel or not
#define MLO_READ_PARTIAL_N_LCL_IN_MAPS (MLO_N_INPUTS % MLO_N_LCL_IN_MAPS != 0)

/*
        group cooperative read
        read by MLO_READ_UNIT
        handle out of range both horizontally and vertically (by fixed number of veryical reads)

        no guard against number of inputs
*/
__device__ static inline void readInput(unsigned int lcl_id,
                                        unsigned int gbl_in_scan_off,
#if !MLO_READ_PARTIAL_N_LCL_IN_MAPS
                                        UNUSED
#endif
                                        unsigned int n_in_map_reads,
                                        unsigned int n_v_reads,
                                        const FLOAT* __restrict__ bot,
                                        FLOAT* __restrict__ lcl_bot)
{
    for(unsigned int p4 = lcl_id; p4 < MLO_N_LCL_IN_MAPS * MLO_N_IN_HORIZ_READS * n_v_reads;
        p4 += MLO_GRP_SZ)
    {
        unsigned int c    = 0;
        unsigned int t_p4 = p4;
#if MLO_N_LCL_IN_MAPS > 1
        c    = p4 / (MLO_N_IN_HORIZ_READS * n_v_reads);
        t_p4 = iMod(p4, c, (MLO_N_IN_HORIZ_READS * n_v_reads));
#endif

        unsigned int c_scan = t_p4 / (MLO_N_IN_HORIZ_READS);

#if MLO_N_IN_HORIZ_READS & (MLO_N_IN_HORIZ_READS - 1)
        unsigned int c_pix4 = iMod(t_p4, c_scan, (MLO_N_IN_HORIZ_READS));
#else
        unsigned int c_pix4 = t_p4 & (MLO_N_IN_HORIZ_READS - 1);
#endif

        unsigned int bot_off = gbl_in_scan_off + c * MLO_IN_CHANNEL_STRIDE +
                               c_scan * MLO_IN_STRIDE + c_pix4 * MLO_READ_UNIT;
        const FLOAT* bot_p = &bot[bot_off];

        FLOAT in_rd_data[MLO_READ_UNIT];

        for(unsigned int i = 0; i < MLO_READ_UNIT; ++i)
        {
            in_rd_data[i] = 0;
        }

#if MLO_READ_PARTIAL_N_LCL_IN_MAPS
        if(c < n_in_map_reads)
#endif
        {
#if MLO_IN_N_PIXS_OFF > 0
            if(c_pix4 == MLO_N_IN_HORIZ_READS - 1)
            {
                for(unsigned int i = 0; i < MLO_IN_N_PIXS_OFF; ++i)
                {
                    in_rd_data[i] = bot_p[i];
                }
            }
            else
#endif
            {

                for(unsigned int i = 0; i < MLO_READ_UNIT; ++i)
                {
                    in_rd_data[i] = bot_p[i];
                }
            }
        }

        // MLO_N_LCL_IN_MAPS inputs
        for(unsigned int i = 0; i < MLO_READ_UNIT; ++i)
        {
            int lcl_in_off = c * MLO_IN_LCL_SZ + c_scan * MLO_IN_LCL_WIDTH + MLO_FILTER_PAD0 +
                             c_pix4 * MLO_READ_UNIT + i;
            lcl_bot[lcl_in_off] = in_rd_data[i];
        }

    } // for (int p4 = lcl_id; p4 < MLO_N_LCL_IN_MAPS * MLO_N_IN_HORIZ_READS * MLO_IN_VERT_READS;

    __syncthreads();
}

/*
        core processing loop
        bot - input, from local (1 span)
        top - output diff, from global (array of spans, filters vertical size)

        loop over filter vertical size
*/
__device__ static inline void
Processing(UNUSED unsigned int sc,
           unsigned int sc_lcl_off,
           unsigned int top_lim,
           int bot_lim, // bot_lim could be negative at lower boundary padding
           FLOAT_ACCUM* __restrict__ pvt_accum,
           FLOAT* __restrict__ lcl_bot,
           FLOAT* __restrict__ top_dat)
{
    for(int l = top_lim; l >= bot_lim; --l)
    {
        for(unsigned int m = 0; m < MLO_IN_TILE0; ++m)
        {
            for(unsigned int n = 0; n < MLO_FILTER_SIZE0; ++n)
            {
                for(unsigned int c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
                {
                    unsigned int bot_off = sc_lcl_off + c * MLO_IN_LCL_SZ + n + m;

                    FLOAT bot_val = lcl_bot[bot_off];

                    for(unsigned int k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
                    {
                        unsigned int pvt_top_off =
                            k * MLO_IN_TILE0 * MLO_FILTER_SIZE1 + (top_lim - l) * MLO_IN_TILE0 + m;
                        unsigned int pvt_accum_off =
                            (k * MLO_N_LCL_IN_MAPS + c) * MLO_FILTER_SIZE1 * MLO_FILTER_SIZE0 +
                            l * MLO_FILTER_SIZE0 + n;

                        FLOAT top_val = top_dat[pvt_top_off];

                        pvt_accum[pvt_accum_off]
                            // each wk-it process an input
                            += CVT_FLOAT2ACCUM(bot_val) * CVT_FLOAT2ACCUM(top_val);
                    }
                }
            }
        }
    }
}

__device__ static inline void moveOutputUp(FLOAT* __restrict__ top_dat)
{
    // move up output to reduce overfetch
    for(unsigned int k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
    {
        for(unsigned int j = 0; j < MLO_FILTER_SIZE1 - 1; ++j)
        {
            for(unsigned int i = 0; i < MLO_IN_TILE0; ++i)
            {
                unsigned int pvt_off_n = k * MLO_IN_TILE0 * MLO_FILTER_SIZE1 + j * MLO_IN_TILE0 + i;
                unsigned int pvt_off_o =
                    k * MLO_IN_TILE0 * MLO_FILTER_SIZE1 + (j + 1) * MLO_IN_TILE0 + i;
                top_dat[pvt_off_n] = top_dat[pvt_off_o];
            }
        }
    }
}

__device__ static inline void spanReadingOutput(int spn,
                                                int k,
                                                int j,
                                                int top_df_off,
                                                FLOAT mask,
                                                FLOAT* __restrict__ top_dat,
                                                const FLOAT* __restrict__ top_df)
{
    int pvt_off           = k * MLO_IN_TILE0 * MLO_FILTER_SIZE1 + j * MLO_IN_TILE0;
    const FLOAT* top_df_p = &top_df[top_df_off];
#if MLO_OUT_N_PIXS_OFF > 0
    if(spn == MLO_N_SPANS_PER_SCAN - 1)
    {
        unsigned int i = 0;
        for(; i < MLO_OUT_N_PIXS_OFF; ++i)
        {
            top_dat[pvt_off + i] = top_df_p[i] * mask;
        }
        for(; i < MLO_IN_TILE0; ++i)
        {
            top_dat[pvt_off + i] = 0;
        }
    }
    else
#else
    (void)spn;
#endif
    {
        for(unsigned int i = 0; i < MLO_IN_TILE0; ++i)
        {
            top_dat[pvt_off + i] = top_df_p[i] * mask;
        }
    }
}

/*********************************************************************************************************
// wrw algorithm for large filters
// idea:
// split output scan-line on number of spans by the  MLO_IN_TILE0 (2 for example)
// 1 scan-line has ((MLO_OUT_WIDTH + MLO_IN_TILE0 - 1/MLO_IN_TILE0) spans
// group will process MLO_GRP_SZ/((MLO_OUT_WIDTH + MLO_IN_TILE0 - 1/MLO_IN_TILE0) output maps

// alg
// load a block of input map (or full map) into LDS
// loop
// read MLO_FILTER_SIZE1 number of spans from output map into VGPRs (for example 5 *2 = 10)
// read 1 input line for  maps into LDS
// accumulate

// accumulate all spans at the end
// start new loop for the next batch (if defined)
// write out

// kerenl handles 5x5, 3x3 with padding
// small images in 1 short- MLO_N_GENERIC_LOOPS == 0
// big images  in 2 blocks - MLO_IN_N_VERT_LOOPS == 2 or multiple blocks - MLO_IN_N_VERT_LOOPS > 2
// there are prolog and apilog that deal with top/bottom padding.
// left/right padding handles as a LDS border pixels zeroed at the beginning.

**********************************************************************************************************/

extern "C" __global__ void __launch_bounds__((MLO_GRP_SZ0) * (MLO_GRP_SZ1) * (MLO_GRP_SZ2))
    MIOpenCvBwdWrW(const FLOAT* __restrict__ top_df,
                   const FLOAT* __restrict__ bot,
                   FLOAT* __restrict__ weights_df,
#if MLO_CONV_BIAS
                   FLOAT* __restrict__ bias_df,
#endif
                   UNUSED FLOAT padding_val)
{

    // input/output tiles + reduce buffer

    __shared__ FLOAT lcl[(MLO_LCL_SZ) + 1];
    FLOAT* lcl_bot = lcl;

    unsigned int lcl_id = threadIdx.x;

    unsigned int c_idx_base = blockIdx.x; // input map index base

    unsigned int o_idx_base = blockIdx.y; // output map index base

    unsigned int ib_base = blockIdx.z;

    unsigned int ib = ib_base * (MLO_N_BATCH_LOOPS * MLO_N_LCL_BATCHS);

    unsigned int o_idx = o_idx_base * (MLO_N_LCL_OUT_MAPS * MLO_OUT_STACKS); // output map index

    unsigned int channel_group_idx = o_idx / MLO_N_OUTPUTS_PER_GROUP;

    unsigned int c_idx = c_idx_base * MLO_N_LCL_IN_MAPS +
                         channel_group_idx * MLO_N_INPUTS_PER_GROUP; // input map index

    unsigned int wc_idx = c_idx_base * MLO_N_LCL_IN_MAPS;

#if MLO_READ_PARTIAL_N_LCL_IN_MAPS
    unsigned int n_in_map_reads = MLO_N_INPUTS >= c_idx + MLO_N_LCL_IN_MAPS
                                      ? MLO_N_LCL_IN_MAPS
                                      : (MLO_N_INPUTS >= c_idx ? MLO_N_INPUTS - c_idx : 0);
#else
    unsigned int n_in_map_reads = MLO_N_LCL_IN_MAPS;
#endif

    unsigned int gbl_in_off  = c_idx * MLO_IN_CHANNEL_STRIDE + ib * MLO_IN_BATCH_STRIDE;
    unsigned int gbl_out_off = o_idx * MLO_OUT_CHANNEL_STRIDE + ib * MLO_OUT_BATCH_STRIDE;
    // 1 span per wk_item, total scanline with MLO_N_SPANS_PER_SCAN spans
    // TODO: more than 1 input
    unsigned int o = lcl_id / MLO_N_SPANS_PER_SCAN;
#if MLO_N_SPANS_PER_SCAN & (MLO_N_SPANS_PER_SCAN - 1)
    unsigned int spn = iMod(lcl_id, o, MLO_N_SPANS_PER_SCAN);
#else
    unsigned int spn = lcl_id & (MLO_N_SPANS_PER_SCAN - 1);
#endif
    //	bool scan_lead = (o*MLO_N_SPANS_PER_SCAN == lcl_id);

    unsigned int lcl_bot_off     = spn * MLO_IN_TILE0;
    unsigned int out_wk_item_off = o * MLO_OUT_CHANNEL_STRIDE + lcl_bot_off;
    gbl_out_off += out_wk_item_off;
    // no output out of range
    gbl_out_off = (o_idx + o < MLO_N_OUTPUTS && o < MLO_OUT_STACKS) ? gbl_out_off : 0;

#define MLO_TOP_DAT_SZ (MLO_N_LCL_OUT_MAPS * MLO_IN_TILE0 * MLO_FILTER_SIZE1)

    FLOAT top_dat[MLO_TOP_DAT_SZ];

    for(unsigned int i = 0; i < MLO_TOP_DAT_SZ; ++i)
    {
        top_dat[i] = 0;
    }

#define MLO_ACCUM_SZ (MLO_N_LCL_OUT_MAPS * MLO_N_LCL_IN_MAPS * MLO_FILTER_SIZE1 * MLO_FILTER_SIZE0)

    FLOAT_ACCUM pvt_accum[MLO_ACCUM_SZ];

    for(unsigned int i = 0; i < MLO_ACCUM_SZ; ++i)
    {
        pvt_accum[i] = 0;
    }

    // zero out LDS
    for(unsigned int i = lcl_id; i < (MLO_LCL_SZ); i += MLO_GRP_SZ)
    {
        lcl[i] = 0;
    }

    // over all batches
    unsigned int bend = ib + MLO_N_BATCH_LOOPS * MLO_N_LCL_BATCHS;
    bend              = bend > MLO_BATCH_SZ ? MLO_BATCH_SZ : bend;

    for(unsigned int b = ib; b < bend; ++b,
                     gbl_in_off += MLO_N_LCL_BATCHS * MLO_IN_BATCH_STRIDE,
                     gbl_out_off += MLO_N_LCL_BATCHS * MLO_OUT_BATCH_STRIDE)
    {
        __syncthreads();

        // top border input block
        unsigned int gbl_in_scan_off  = gbl_in_off;
        unsigned int gbl_out_scan_off = gbl_out_off;

        // read input map
        readInput(lcl_id, gbl_in_scan_off, n_in_map_reads, MLO_IN_VERT_READS, bot, lcl_bot);

        // move input pointer
        gbl_in_scan_off += MLO_IN_STRIDE * MLO_IN_EXTENT1;

        for(unsigned int i = 0; i < MLO_TOP_DAT_SZ; ++i)
        {
            top_dat[i] = 0;
        }

        // prefetch output
        unsigned int gbl_out_scan_off1 = gbl_out_scan_off;
        for(unsigned int k = 0; k < MLO_N_LCL_OUT_MAPS;
            ++k, gbl_out_scan_off1 += MLO_OUT_STACKS * MLO_OUT_CHANNEL_STRIDE)
        {
            for(unsigned int j = 0; j < MLO_FILTER_SIZE1 - 1; ++j)
            {
                // loop around all output maps
                unsigned int top_df_off = gbl_out_scan_off1 + j * MLO_OUT_STRIDE;
                FLOAT mask              = 1;
#if MLO_IN_HEIGHT != MLO_OUT_HEIGHT || MLO_FILTER_SIZE1 - 1 > MLO_OUT_HEIGHT
                top_df_off = (j < MLO_OUT_HEIGHT) ? top_df_off : 0;
                mask       = (j < MLO_OUT_HEIGHT) ? 1 : 0;
#endif

                spanReadingOutput(spn, k, j, top_df_off, mask, top_dat, top_df);
            }
        }

        gbl_out_scan_off += (MLO_FILTER_SIZE1 - 1) * MLO_OUT_STRIDE;

        unsigned int sc         = 0;
        unsigned int sc_lcl_off = lcl_bot_off;

        // prolog
        // handling padding

        // top padding
        for(; sc < MLO_FILTER_SIZE1 - MLO_FILTER_PAD1 - 1; ++sc, sc_lcl_off += MLO_IN_LCL_WIDTH)
        {
            Processing(sc, sc_lcl_off, sc + MLO_FILTER_PAD1, 0, pvt_accum, lcl_bot, top_dat);
        }

#ifdef __AMDGCN__
#pragma unroll 2
#endif

#if MLO_IN_N_VERT_LOOPS == 1
        for(; sc < MLO_IN_HEIGHT + MLO_FILTER_PAD1 - MLO_FILTER_SIZE1 + 1;
#else
        for(; sc < MLO_IN_EXTENT1;
#endif
            ++sc, gbl_out_scan_off += MLO_OUT_STRIDE, sc_lcl_off += MLO_IN_LCL_WIDTH)
        {

            for(unsigned int k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
            {
                unsigned int top_df_off =
                    gbl_out_scan_off + k * MLO_OUT_STACKS * MLO_OUT_CHANNEL_STRIDE;
                FLOAT mask = 1;

#if MLO_IN_HEIGHT != MLO_OUT_HEIGHT || MLO_FILTER_SIZE1 - 1 > MLO_OUT_HEIGHT
                top_df_off = ((sc + MLO_FILTER_PAD1) < MLO_OUT_HEIGHT) ? top_df_off : 0;
                mask       = ((sc + MLO_FILTER_PAD1) < MLO_OUT_HEIGHT) ? 1 : 0;
#endif

                spanReadingOutput(
                    spn, k, (MLO_FILTER_SIZE1 - 1), top_df_off, mask, top_dat, top_df);
            }

            // processing
            Processing(sc, sc_lcl_off, MLO_FILTER_SIZE1 - 1, 0, pvt_accum, lcl_bot, top_dat);

            // move up output to reduce overfetch
            moveOutputUp(top_dat);
        }

        // non-border input blocks
        for(unsigned int i_loop = 0; i_loop < MLO_N_GENERIC_LOOPS;
            ++i_loop, gbl_in_scan_off += MLO_IN_STRIDE * MLO_IN_EXTENT1)
        {
            __syncthreads();

            readInput(lcl_id, gbl_in_scan_off, n_in_map_reads, MLO_IN_VERT_READS, bot, lcl_bot);

            // point to the start of the local buffer

            sc_lcl_off = lcl_bot_off;

            for(; sc < (i_loop + 2) * MLO_IN_EXTENT1;
                ++sc, gbl_out_scan_off += MLO_OUT_STRIDE, sc_lcl_off += MLO_IN_LCL_WIDTH)
            {

                for(unsigned int k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
                {
                    unsigned int top_df_off =
                        gbl_out_scan_off + k * MLO_OUT_STACKS * MLO_OUT_CHANNEL_STRIDE;
                    FLOAT mask = 1;

#if MLO_IN_HEIGHT != MLO_OUT_HEIGHT
                    top_df_off = ((sc + MLO_FILTER_PAD1) < MLO_OUT_HEIGHT) ? top_df_off : 0;
                    mask       = ((sc + MLO_FILTER_PAD1) < MLO_OUT_HEIGHT) ? 1 : 0;
#endif

                    spanReadingOutput(
                        spn, k, (MLO_FILTER_SIZE1 - 1), top_df_off, mask, top_dat, top_df);
                }

                // processing
                Processing(sc, sc_lcl_off, MLO_FILTER_SIZE1 - 1, 0, pvt_accum, lcl_bot, top_dat);

                // move up output to reduce overfetch
                moveOutputUp(top_dat);
            }
        }

        // bottom border block

        for(int i_loop = 0; i_loop < (MLO_IN_N_VERT_LOOPS - MLO_N_GENERIC_LOOPS - 1);
            ++i_loop, gbl_in_scan_off += MLO_IN_STRIDE * MLO_IN_EXTENT1)
        {
            __syncthreads();

            // read 1 scan line less
            // padding processing takes care of the bottom border.

#define MLO_LAST_VERT_READS (MLO_IN_HEIGHT - MLO_IN_EXTENT1 * (MLO_IN_N_VERT_LOOPS - 1))

            readInput(lcl_id, gbl_in_scan_off, n_in_map_reads, MLO_LAST_VERT_READS, bot, lcl_bot);

            // point to the start of the local buffer
            sc_lcl_off = lcl_bot_off;

#ifndef MLO_DISABLE_PRAGMA_UNROLL_COMPILER_SWDEV_200074_WORKAROUND
#pragma unroll 3
#endif
            for(; sc < MLO_IN_HEIGHT + MLO_FILTER_PAD1 - MLO_FILTER_SIZE1 + 1;
                ++sc, gbl_out_scan_off += MLO_OUT_STRIDE, sc_lcl_off += MLO_IN_LCL_WIDTH)
            {

                for(unsigned int k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
                {
                    unsigned int top_df_off =
                        gbl_out_scan_off + k * MLO_OUT_STACKS * MLO_OUT_CHANNEL_STRIDE;
                    FLOAT mask = 1;

                    spanReadingOutput(
                        spn, k, (MLO_FILTER_SIZE1 - 1), top_df_off, mask, top_dat, top_df);
                }

                // processing
                Processing(sc, sc_lcl_off, MLO_FILTER_SIZE1 - 1, 0, pvt_accum, lcl_bot, top_dat);

                // move up output to reduce overfetch
                moveOutputUp(top_dat);
            }
        }

        // epilog
        // handling padding

        for(; sc < MLO_IN_HEIGHT; ++sc, sc_lcl_off += MLO_IN_LCL_WIDTH)
        {

            // processing
            Processing(sc,
                       sc_lcl_off,
                       MLO_FILTER_SIZE1 - 1,
                       MLO_FILTER_SIZE1 - (MLO_IN_HEIGHT + MLO_FILTER_PAD1 - sc),
                       pvt_accum,
                       lcl_bot,
                       top_dat);

            // move up output to reduce overfetch
            moveOutputUp(top_dat);

        } // for (; sc < MLO_OUT_HEIGHT - MLO_FILTER_PAD1 + 2; ++sc, gbl_out_scan_off +=
          // MLO_OUT_CHANNEL_STRIDE, gbl_in_scan_off += MLO_IN_CHANNEL_STRIDE)
    } //     for (int b = 0;

    // final summation over all output maps and each filter row
    // this coudl be done with log but it negligeble anyway
    for(unsigned int k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
    {
        for(unsigned int c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
        {

            for(unsigned int l = 0; l < MLO_FILTER_SIZE1; ++l)
            {
                __syncthreads();

                for(unsigned int n = 0; n < MLO_FILTER_SIZE0; ++n)
                {
                    unsigned int pvt_off =
                        (k * MLO_N_LCL_IN_MAPS + c) * MLO_FILTER_SIZE1 * MLO_FILTER_SIZE0 +
                        l * MLO_FILTER_SIZE0 + n;

                    lcl[lcl_id * MLO_FILTER_SIZE0 + n] = CVT_ACCUM2FLOAT(pvt_accum[pvt_off]);
                }

                __syncthreads();

                if(spn == 0)
                {
                    for(unsigned int s = 0; s < MLO_N_SPANS_PER_SCAN - 1; ++s)
                    {

                        for(unsigned int n = 0; n < MLO_FILTER_SIZE0; ++n)
                        {
                            unsigned int pvt_off =
                                (k * MLO_N_LCL_IN_MAPS + c) * MLO_FILTER_SIZE1 * MLO_FILTER_SIZE0 +
                                l * MLO_FILTER_SIZE0 + n;
                            pvt_accum[pvt_off] +=
                                CVT_FLOAT2ACCUM(lcl[(lcl_id + s + 1) * MLO_FILTER_SIZE0 + n]);
                        }
                    }
                }
            }
        }
    }

    // output
    // inputs are outputs
    // TODO : for more than 1 input

    unsigned int wei_df_off = (((ib / MLO_N_BATCH_LOOPS) * MLO_N_OUTPUTS + o_idx + o) *
                               (unsigned int)MLO_WEI_BATCH_STRIDE)
                              // this input channel
                              + (wc_idx * (unsigned int)MLO_WEI_CHANNEL_STRIDE);

    for(unsigned int k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
    {
        for(unsigned int c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
        {
            if(spn == 0 && c < n_in_map_reads && o_idx + o + k * MLO_OUT_STACKS < MLO_N_OUTPUTS &&
               o < MLO_OUT_STACKS)
            {
                for(unsigned int i = 0; i < (MLO_FILTER_SIZE1 * MLO_FILTER_SIZE0); ++i)
                {
                    weights_df[wei_df_off + k * MLO_OUT_STACKS * MLO_WEI_BATCH_STRIDE +
                               c * MLO_WEI_CHANNEL_STRIDE + i] =
                        CVT_ACCUM2FLOAT(pvt_accum[(k * MLO_N_LCL_IN_MAPS + c) * MLO_FILTER_SIZE1 *
                                                      MLO_FILTER_SIZE0 +
                                                  i]);
                }
            }
        }
    }
}

// final reduction kernel
// add filters over batches
extern "C" __global__ void __launch_bounds__(MLO_UT_GRP_SZ0)
    MIOpenCvBwdWrW_rdc(const FLOAT* __restrict__ weight_df_tmp, FLOAT* __restrict__ weights_df)
{
    unsigned int gbl_id   = (blockIdx.x * blockDim.x + threadIdx.x);
    unsigned int wei_idx0 = gbl_id * MLO_UT_READ_UNIT;

#if MLO_WEI_CHANNEL_STRIDE & (MLO_WEI_CHANNEL_STRIDE - 1)
    unsigned int wei_blk_idx = iDiv(wei_idx0, MLO_WEI_CHANNEL_STRIDE);
    unsigned int wei_idx     = iMod(wei_idx0, wei_blk_idx, MLO_WEI_CHANNEL_STRIDE);
#else
    unsigned int wei_blk_idx = wei_idx0 / MLO_WEI_CHANNEL_STRIDE;
    unsigned int wei_idx     = wei_idx0 & (MLO_WEI_CHANNEL_STRIDE - 1);
#endif

    FLOAT_ACCUM pvt_accum_wei[MLO_UT_READ_UNIT] = {0};

    int batch_loop = (MLO_BATCH_SZ + (MLO_N_BATCH_LOOPS * MLO_N_LCL_BATCHS) - 1) /
                     (MLO_N_BATCH_LOOPS * MLO_N_LCL_BATCHS);

    for(unsigned int i = 0; i < batch_loop; ++i)
    {
        for(unsigned int j = 0; j < MLO_UT_READ_UNIT; ++j)
        {
            pvt_accum_wei[j] +=
                CVT_FLOAT2ACCUM(weight_df_tmp[(wei_blk_idx * MLO_WEI_CHANNEL_STRIDE +
                                               i * MLO_N_OUTPUTS * MLO_WEI_BATCH_STRIDE) +
                                              wei_idx + j]);
        }
    }

    for(unsigned int j = 0; j < MLO_UT_READ_UNIT; ++j)
    {
        weights_df[wei_idx0 + j] = CVT_ACCUM2FLOAT(pvt_accum_wei[j]);
    }
}
