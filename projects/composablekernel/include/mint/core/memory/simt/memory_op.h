#pragma once
#include <mint/config.h>
#include <mint/core/memory/simt/global_memory_view.h>
#include <mint/core/memory/simt/shared_memory_view.h>
#include <mint/core/memory/simt/vgpr_memory_view.h>

namespace mint {

// global load: global-->vgpr
template <class Dst, class Src, index_t kIDst, index_t kLength>
  requires(
      Dst::address_space() == address_space::vgpr &&
      Src::address_space() == address_space::global &&
      is_same_v<typename Dst::value_type, typename Src::value_type>)
MINT_DEVICE void thread_copy_1d(
    Dst& dst,
    index_constant<kIDst> /*iDst*/,
    const Src& src,
    typename Src::size_type iSrc,
    index_constant<kLength>) {
  dst.template as_vectors<kLength>().template at<kIDst / kLength>() =
      src.template as_vectors<kLength>()[iSrc / kLength];
}

// global store: vpgr-->global
template <class Dst, class Src, index_t kISrc, index_t kLength>
  requires(
      Dst::address_space() == address_space::global &&
      Src::address_space() == address_space::vgpr &&
      is_same_v<typename Dst::value_type, typename Src::value_type>)
MINT_DEVICE void thread_copy_1d(
    Dst& dst,
    typename Dst::size_type iDst,
    const Src& src,
    index_constant<kISrc> /*iSrc*/,
    index_constant<kLength>) {
  dst.template as_vectors<kLength>()[iDst / kLength] =
      src.template as_vectors<kLength>().template at<kISrc / kLength>();
}

} // namespace mint
