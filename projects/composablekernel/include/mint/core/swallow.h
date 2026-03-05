#pragma once
#include <mint/config.h>

namespace mint {

template <class... Ts>
MINT_HOST_DEVICE constexpr void swallow(Ts&&...) {}

} // namespace mint
