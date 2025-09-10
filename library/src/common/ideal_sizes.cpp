#include "ideal_sizes.hpp"

//// Thread block sizes
// (BS1 and BS2 are now macros, not variables)

// larf
#ifndef LARF_SSKER_THREADS
const int LARF_SSKER_THREADS = 256;      // must be 64, 128, 256, 512, or 1024
#endif
#ifndef LARF_SSKER_BLOCKS
const int LARF_SSKER_BLOCKS = 64;
#endif
#ifndef LARF_SSKER_MIN_DIM
const int LARF_SSKER_MIN_DIM = 64;       // should be >= LARF_SSKER_BLOCKS
#endif

// larfg
#ifndef LARFG_SSKER_THREADS
const int LARFG_SSKER_THREADS = 256;     // must be 64, 128, 256, 512, or 1024
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

// GESVD
#ifndef THIN_SVD_SWITCH
const double THIN_SVD_SWITCH = 1.6;
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
