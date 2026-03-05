#pragma once
#include <type_traits>

namespace mint {

// https://stackoverflow.com/questions/51523965/template-dependent-false
// https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1830r1.pdf
template <typename>
struct dependent_false : std::false_type {};

} // namespace mint
