#include "ideal_sizes.hpp"

//// Thread block sizes
// (BS1 and BS2 are now macros, not variables)

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

#ifndef SYEVJ_BLOCKED_SWITCH
const int SYEVJ_BLOCKED_SWITCH = 58;
#endif

#ifndef SYTRF_BLOCKSIZE
const int SYTRF_BLOCKSIZE = 64;
#endif

// larf
#ifndef LARF_SSKER_THREADS
const int LARF_SSKER_THREADS = 256;
#endif
#ifndef LARF_SSKER_BLOCKS
const int LARF_SSKER_BLOCKS = 64;
#endif
#ifndef LARF_SSKER_MIN_DIM
const int LARF_SSKER_MIN_DIM = 64;
#endif

// larfg
#ifndef LARFG_SSKER_THREADS
const int LARFG_SSKER_THREADS = 256;
#endif
#ifndef LARFG_SSKER_MAX_N
const int LARFG_SSKER_MAX_N = 4096;
#endif

// larft
#ifndef LARFT_SWITCHSIZE
const int LARFT_SWITCHSIZE = 128;
#endif

// GEQRF & GEQLF, block sizes
#ifndef GEQxF_BLOCKSIZE
const int GEQxF_BLOCKSIZE = 64;
#endif
#ifndef GEQxF_GEQx2_SWITCHSIZE
const int GEQxF_GEQx2_SWITCHSIZE = 128;
#endif

// GERQF & GELQF, block sizes
#ifndef GExQF_BLOCKSIZE
const int GExQF_BLOCKSIZE = 64;
#endif
#ifndef GExQF_GExQ2_SWITCHSIZE
const int GExQF_GExQ2_SWITCHSIZE = 128;
#endif

// ORGQR/ORGQL/UNGQR/UNGQL
#ifndef xxGQx_BLOCKSIZE
const int xxGQx_BLOCKSIZE = 64;
#endif
#ifndef xxGQx_xxGQx2_SWITCHSIZE
const int xxGQx_xxGQx2_SWITCHSIZE = 128;
#endif

// ORGRQ/ORGLQ/UNGRQ/UNGLQ
#ifndef xxGxQ_BLOCKSIZE
const int xxGxQ_BLOCKSIZE = 64;
#endif
#ifndef xxGxQ_xxGxQ2_SWITCHSIZE
const int xxGxQ_xxGxQ2_SWITCHSIZE = 128;
#endif

// ORMQR/ORMQL/UNMQR/UNMQL
#ifndef xxMQx_BLOCKSIZE
const int xxMQx_BLOCKSIZE = 64;
#endif

// ORMRQ/ORMLQ/UNMRQ/UNMLQ
#ifndef xxMxQ_BLOCKSIZE
const int xxMxQ_BLOCKSIZE = 64;
#endif

// GEBRD, block sizes
#ifndef GEBRD_BLOCKSIZE
const int GEBRD_BLOCKSIZE = 32;
#endif
#ifndef GEBRD_GEBD2_SWITCHSIZE
const int GEBRD_GEBD2_SWITCHSIZE = 64;
#endif

// BDSQR
#ifndef BDSQR_SWITCH_SIZE
const int BDSQR_SWITCH_SIZE = 512;
#endif
#ifndef BDSQR_ITERS_PER_SYNC
const int BDSQR_ITERS_PER_SYNC = 10;
#endif

// SYTRD/HETRD
#ifndef xxTRD_BLOCKSIZE
const int xxTRD_BLOCKSIZE = 64;
#endif
#ifndef xxTRD_xxTD2_SWITCHSIZE
const int xxTRD_xxTD2_SWITCHSIZE = 256;
#endif
#ifndef xxTD2_SSKER_MAX_N
const int xxTD2_SSKER_MAX_N = 192;
#endif

// SYGST/HEGST
#ifndef xxGST_BLOCKSIZE
const int xxGST_BLOCKSIZE = 64;
#endif

// STEDC
#ifndef STEDC_MIN_DC_SIZE
const int STEDC_MIN_DC_SIZE = 16;
#endif
#ifndef STEDC_NUM_SPLIT_BLKS
const int STEDC_NUM_SPLIT_BLKS = 8;
#endif

#ifndef THIN_SVD_SWITCH
const double THIN_SVD_SWITCH = 1.6;
#endif
#ifndef SYTRF_SYTF2_SWITCHSIZE
const int SYTRF_SYTF2_SWITCHSIZE = 128;
#endif
#ifndef SYEVDJ_MIN_DC_SIZE
const int SYEVDJ_MIN_DC_SIZE = 16;
#endif
#ifndef SYEVDX_MIN_DC_SIZE
const int SYEVDX_MIN_DC_SIZE = 16;
#endif
#ifndef GETF2_SPKER_MAX_M
const int GETF2_SPKER_MAX_M = 1024;
#endif
#ifndef GETF2_SPKER_MAX_N
const int GETF2_SPKER_MAX_N = 256;
#endif
#ifndef GETF2_SSKER_MAX_M
const int GETF2_SSKER_MAX_M = 512;
#endif
#ifndef GETF2_SSKER_MAX_N
const int GETF2_SSKER_MAX_N = 64;
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
#ifndef TRTRI_MAX_COLS
const int TRTRI_MAX_COLS = 64;
#endif
#ifndef TRTRI_NUM_INTERVALS
const int TRTRI_NUM_INTERVALS = 1;
#endif
#ifndef TRTRI_INTERVALS
const int TRTRI_INTERVALS = 0;
#endif
#ifndef TRTRI_BATCH_NUM_INTERVALS
const int TRTRI_BATCH_NUM_INTERVALS = 3;
#endif
#ifndef SPLITLU_SWITCH_SIZE
const int SPLITLU_SWITCH_SIZE = 64;
#endif
