#pragma once
#include <mint/core/integral_constant.h>
#include <mint/core/sequence.h>
#include <mint/core/static_string.h>

namespace mint {

// Used throughout MINT for compile-time string identifiers
using alias_t = static_string<16>;

// Template wrapper for variadic aliases
// Usage: aliases<"M", "N", "K">{}
template <alias_t... kAliases>
using aliases = sequence<alias_t, kAliases...>;

// Template wrapper for single alias
// Usage: alias<"Offset">{}
template <alias_t kAlias>
using alias = integral_constant<alias_t, kAlias>;

} // namespace mint
