#pragma once

#if 0
#include <bitset>

namespace mint {

using std::bitset;

} //namespace mint
#else
#include "mint/core/custom_bitset.h"

namespace mint {

template <index_t kN>
using bitset = custom_bitset<kN>;

} // namespace mint
#endif
