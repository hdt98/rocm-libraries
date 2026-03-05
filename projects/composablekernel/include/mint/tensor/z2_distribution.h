#pragma once
#include <mint/core.h>
#include <mint/poly.h>

namespace mint::tensor {

// {r, m, n} <--> {p, e} bidirectional mapping
template <class Z2LinearMorpher>
  requires(is_z2_linear_morpher<Z2LinearMorpher>::value)
struct z2_distribution {
  Z2LinearMorpher morpher_;

  MINT_HOST_DEVICE constexpr z2_distribution(const Z2LinearMorpher& morpher)
      : morpher_{morpher} {}
};

} // namespace mint::tensor
