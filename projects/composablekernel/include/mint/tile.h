#pragma once
#include <mint/tile/generic/atomic_add_no_shuffle.h>
#include <mint/tile/generic/elementwise_index_aware_no_shuffle.h>
#include <mint/tile/generic/elementwise_no_shuffle.h>
#include <mint/tile/generic/load_no_shuffle.h>
#include <mint/tile/generic/load_no_shuffle_vectorized.h>
#include <mint/tile/generic/load_no_shuffle_vectorized_bytewise_offset.h>
#include <mint/tile/generic/matmul_no_shuffle.h>
#include <mint/tile/generic/reduce_no_shuffle.h>
#include <mint/tile/generic/store_no_shuffle.h>
#include <mint/tile/generic/store_no_shuffle_vectorized.h>
#include <mint/tile/generic/store_no_shuffle_vectorized_bytewise_offset.h>
#include <mint/tile/simt/block/load.h>
#include <mint/tile/simt/block/store.h>
#include <mint/tile/simt/load.h>
#include <mint/tile/simt/partition.h>
#include <mint/tile/simt/store.h>
#include <mint/tile/simt/warp/atomic_add.h>
#include <mint/tile/simt/warp/elementwise.h>
#include <mint/tile/simt/warp/load.h>
#include <mint/tile/simt/warp/matmul.h>
#include <mint/tile/simt/warp/reduce.h>
#include <mint/tile/simt/warp/reduce_z2.h>
#include <mint/tile/simt/warp/shuffle_z2.h>
#include <mint/tile/simt/warp/store.h>

#if defined(MINT_BACKEND_CUDA)
// include nothing
#elif defined(MINT_BACKEND_ROCM)
#include <mint/tile/rocm/load_no_shuffle_vectorized_async_copy.h>
#include <mint/tile/rocm/warp/matmul_dl.h>
#include <mint/tile/rocm/warp/matmul_xdl.h>
#endif

#if defined(MINT_BACKEND_CUDA)
namespace mint {
namespace tile {
namespace cuda {

using namespace mint::tile::simt;

} // namespace cuda
} // namespace tile
} // namespace mint
#elif defined(MINT_BACKEND_ROCM)
namespace mint {
namespace tile {
namespace rocm {

using namespace mint::tile::simt;

} // namespace rocm
} // namespace tile
} // namespace mint
#endif
