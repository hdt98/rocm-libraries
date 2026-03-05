#pragma once
#include <mint/config.h>
#include <mint/core/vector_type/custom_vector_type.h>

namespace mint {

// nvcc doesn't support clang native vector type
template <class T>
struct has_clang_native_vector_type : std::false_type {};

template <class S, index_t kNS>
using vector_type = custom_vector_type<S, kNS>;

} // namespace mint
