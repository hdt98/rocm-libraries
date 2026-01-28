/* **************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#include "ideal_sizes.hpp"

/*
ideal_sizes.hpp and ideal_sizes.cpp gather all constants that can be tuned for performance.
We define variables that are used on the host, and not used as compile time constants, as const int in ideal_sizes.cpp.
We define variables that are used on the device, or used as compile time constants, as macros in ideal_sizes.hpp.
*/

/******************************* larf ****************************************
*******************************************************************************/

#ifndef LARF_SSKER_THREADS
const int LARF_SSKER_THREADS = 256; // must be 64, 128, 256, 512, or 1024
#endif
#ifndef LARF_SSKER_BLOCKS
const int LARF_SSKER_BLOCKS = 64;
#endif
#ifndef LARF_SSKER_MIN_DIM
const int LARF_SSKER_MIN_DIM = 64; // should be >= LARF_SSKER_BLOCKS
#endif

/******************************* larfg ****************************************
*******************************************************************************/

#ifndef LARFG_SSKER_THREADS
const int LARFG_SSKER_THREADS = 256; // must be 64, 128, 256, 512, or 1024
#endif
#ifndef LARFG_SSKER_MAX_N
const int LARFG_SSKER_MAX_N = 2048;
#endif

/******************************* larft ****************************************
*******************************************************************************/

#ifndef LARFT_SWITCHSIZE
const int LARFT_SWITCHSIZE = 64;
#endif

/***************** geqr2/geqrf and geql2/geqlf ********************************
*******************************************************************************/

#ifndef GEQxF_BLOCKSIZE
const int GEQxF_BLOCKSIZE = 64;
#endif
#ifndef GEQxF_GEQx2_SWITCHSIZE
const int GEQxF_GEQx2_SWITCHSIZE = 128;
#endif

/***************** gerq2/gerqf and gelq2/gelqf ********************************
*******************************************************************************/

#ifndef GExQF_BLOCKSIZE
const int GExQF_BLOCKSIZE = 64;
#endif
#ifndef GExQF_GExQ2_SWITCHSIZE
const int GExQF_GExQ2_SWITCHSIZE = 128;
#endif

/******** org2r/orgqr, org2l/orgql, ung2r/ungqr and ung2l/ungql ***************
*******************************************************************************/

#ifndef xxGQx_BLOCKSIZE
const int xxGQx_BLOCKSIZE = 64;
#endif
#ifndef xxGQx_xxGQx2_SWITCHSIZE
const int xxGQx_xxGQx2_SWITCHSIZE = 128;
#endif

/******** orgr2/orgrq, orgl2/orglq, ungr2/ungrq and ungl2/unglq **************
*******************************************************************************/

#ifndef xxGxQ_BLOCKSIZE
const int xxGxQ_BLOCKSIZE = 64;
#endif
#ifndef xxGxQ_xxGxQ2_SWITCHSIZE
const int xxGxQ_xxGxQ2_SWITCHSIZE = 128;
#endif

/********* orm2r/ormqr, orm2l/ormql, unm2r/unmqr and unm2l/unmql **************
*******************************************************************************/

#ifndef xxMQx_BLOCKSIZE
const int xxMQx_BLOCKSIZE = 64;
#endif

/********* ormr2/ormrq, orml2/ormlq, unmr2/unmrq and unml2/unmlq ***************
*******************************************************************************/

#ifndef xxMxQ_BLOCKSIZE
const int xxMxQ_BLOCKSIZE = 64;
#endif

/**************************** gebd2/gebrd *************************************
*******************************************************************************/

#ifndef GEBRD_BLOCKSIZE
const int GEBRD_BLOCKSIZE = 32;
#endif
#ifndef GEBRD_GEBD2_SWITCHSIZE
const int GEBRD_GEBD2_SWITCHSIZE = 64;
#endif

/******************************* bdsqr ****************************************
*******************************************************************************/

#ifndef BDSQR_SWITCH_SIZE
const int BDSQR_SWITCH_SIZE = 512;
#endif
#ifndef BDSQR_ITERS_PER_SYNC
const int BDSQR_ITERS_PER_SYNC = 10;
#endif

/******************************* gesvd ****************************************
*******************************************************************************/

#ifndef THIN_SVD_SWITCH
const double THIN_SVD_SWITCH = 1.6;
#endif

/******************* sytd2/sytrd and hetd2/hetrd *******************************
*******************************************************************************/

#ifndef xxTRD_BLOCKSIZE
const int xxTRD_BLOCKSIZE = 64;
#endif
#ifndef xxTRD_xxTD2_SWITCHSIZE
const int xxTRD_xxTD2_SWITCHSIZE = 256;
#endif
#ifndef xxTD2_SSKER_MAX_N
const int xxTD2_SSKER_MAX_N = 192;
#endif

/***************** sygs2/sygst and hegs2/hegst ********************************
*******************************************************************************/

#ifndef xxGST_BLOCKSIZE
const int xxGST_BLOCKSIZE = 64;
#endif

/****************************** stedc ******************************************
*******************************************************************************/

#ifndef STEDC_MIN_DC_SIZE
const int STEDC_MIN_DC_SIZE = 16;
#endif
#ifndef STEDC_NUM_SPLIT_BLKS
const int STEDC_NUM_SPLIT_BLKS = 8;
#endif

/************************** syevj/heevj ***************************************
*******************************************************************************/

#ifndef SYEVJ_BLOCKED_SWITCH
const int SYEVJ_BLOCKED_SWITCH = 58;
#endif

/*************************** sytf2/sytrf **************************************
*******************************************************************************/

#ifndef SYTRF_BLOCKSIZE
const int SYTRF_BLOCKSIZE = 64;
#endif
#ifndef SYTRF_SYTF2_SWITCHSIZE
const int SYTRF_SYTF2_SWITCHSIZE = 128;
#endif

/****************************** syevdj ******************************************
*******************************************************************************/

#ifndef SYEVDJ_MIN_DC_SIZE
const int SYEVDJ_MIN_DC_SIZE = 16;
#endif

/****************************** syevdx ******************************************
*******************************************************************************/

#ifndef SYEVDX_MIN_DC_SIZE
const int SYEVDX_MIN_DC_SIZE = 16;
#endif

/**************************** getf2/getfr *************************************
*******************************************************************************/

#ifndef GETF2_SPKER_MAX_M
const int GETF2_SPKER_MAX_M = 1024;
#endif
#ifndef GETF2_SPKER_MAX_N
const int GETF2_SPKER_MAX_N = 256;
#endif
#ifndef GETRF_NUM_INTERVALS_REAL
const int GETRF_NUM_INTERVALS_REAL = 4;
#endif
#ifndef GETRF_BATCH_NUM_INTERVALS_REAL
const int GETRF_BATCH_NUM_INTERVALS_REAL = 9;
#endif
#ifndef GETRF_NPVT_NUM_INTERVALS_REAL
const int GETRF_NPVT_NUM_INTERVALS_REAL = 2;
#endif
#ifndef GETRF_NPVT_BATCH_NUM_INTERVALS_REAL
const int GETRF_NPVT_BATCH_NUM_INTERVALS_REAL = 6;
#endif
#ifndef GETRF_NUM_INTERVALS_COMPLEX
const int GETRF_NUM_INTERVALS_COMPLEX = 4;
#endif
#ifndef GETRF_BATCH_NUM_INTERVALS_COMPLEX
const int GETRF_BATCH_NUM_INTERVALS_COMPLEX = 10;
#endif
#ifndef GETRF_NPVT_NUM_INTERVALS_COMPLEX
const int GETRF_NPVT_NUM_INTERVALS_COMPLEX = 2;
#endif
#ifndef GETRF_NPVT_BATCH_NUM_INTERVALS_COMPLEX
const int GETRF_NPVT_BATCH_NUM_INTERVALS_COMPLEX = 5;
#endif

/****************************** getri *****************************************
*******************************************************************************/

#ifndef GETRI_MAX_COLS
const int GETRI_MAX_COLS = 64;
#endif
#ifndef GETRI_TINY_SIZE
const int GETRI_TINY_SIZE = 43;
#endif
#ifndef GETRI_NUM_INTERVALS
const int GETRI_NUM_INTERVALS = 1;
#endif
#ifndef GETRI_INTERVALS
const int GETRI_INTERVALS = 1185;
#endif
#ifndef GETRI_BATCH_TINY_SIZE
const int GETRI_BATCH_TINY_SIZE = 35;
#endif
#ifndef GETRI_BATCH_NUM_INTERVALS
const int GETRI_BATCH_NUM_INTERVALS = 2;
#endif

/***************************** trtri ******************************************
*******************************************************************************/

#ifndef TRTRI_NUM_INTERVALS
const int TRTRI_NUM_INTERVALS = 1;
#endif
#ifndef TRTRI_INTERVALS
const int TRTRI_INTERVALS = 0;
#endif
#ifndef TRTRI_BATCH_NUM_INTERVALS
const int TRTRI_BATCH_NUM_INTERVALS = 3;
#endif

/************************** splitlu ***************************************
*******************************************************************************/

#ifndef SPLITLU_SWITCH_SIZE
const int SPLITLU_SWITCH_SIZE = 64;
#endif
