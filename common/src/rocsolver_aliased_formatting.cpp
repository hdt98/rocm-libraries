// #include <string>
// #include <string_view>
// #include "rocsolver_utility.hpp"

// #ifdef USE_LIB_FMT
//     #include <fmt/base.h>
//     #include <fmt/format.h>
// 	#include <fmt/ranges.h>
// 	#define LIB_NAMESPACE fmt
// #else
//     #include <print>
//     #include <format>
// 	#include <ranges>
// 	#define LIB_NAMESPACE std
// #endif

// concaternate the two arguments, evaluating them first if they are macros
#define ROCSOLVER_CONCAT2_HELPER(a, b) a##b
#define ROCSOLVER_CONCAT2(a, b) ROCSOLVER_CONCAT2_HELPER(a, b)

#define ROCSOLVER_CONCAT4_HELPER(a, b, c, d) a##b##c##d
#define ROCSOLVER_CONCAT4(a, b, c, d) ROCSOLVER_CONCAT4_HELPER(a, b, c, d)

#if ROCSOLVER_VERSION_MINOR < 10
#define ROCSOLVER_VERSION_MINOR_PADDED ROCSOLVER_CONCAT2(0, ROCSOLVER_VERSION_MINOR)
#else
#define ROCSOLVER_VERSION_MINOR_PADDED ROCSOLVER_VERSION_MINOR
#endif

#if ROCSOLVER_VERSION_PATCH < 10
#define ROCSOLVER_VERSION_PATCH_PADDED ROCSOLVER_CONCAT2(0, ROCSOLVER_VERSION_PATCH)
#else
#define ROCSOLVER_VERSION_PATCH_PADDED ROCSOLVER_VERSION_PATCH
#endif

#ifndef ROCSOLVER_BEGIN_NAMESPACE
#define ROCSOLVER_BEGIN_NAMESPACE                                      \
    namespace rocsolver                                                \
    {                                                                  \
    inline namespace ROCSOLVER_CONCAT4(v,                              \
                                       ROCSOLVER_VERSION_MAJOR,        \
                                       ROCSOLVER_VERSION_MINOR_PADDED, \
                                       ROCSOLVER_VERSION_PATCH_PADDED) \
    {
#define ROCSOLVER_END_NAMESPACE \
    }                           \
    }
#endif

// ROCSOLVER_BEGIN_NAMESPACE

// template<class... Args>
// void print(LIB_NAMESPACE::format_string<Args...> fmt, Args&&... args){
// 	LIB_NAMESPACE::print(fmt, std::forward<Args>(args)...);
// }

// template<class... Args>
// void print(FILE* stream, LIB_NAMESPACE::format_string<Args...> fmt, Args&&... args){
// 	LIB_NAMESPACE::print(stream, fmt, std::forward<Args>(args)...);
// }

// template<class... Args>
// std::string format(LIB_NAMESPACE::format_string<Args...> fmt, Args&&... args){
// 	return LIB_NAMESPACE::format(fmt, std::forward<Args>(args)...);
// }

// template <typename Range>
// std::string join(Range& r, std::string_view sep){
//     #ifdef USE_FMT_LIB
// 	return lib::join(r, sep);
// 	#else
// 	joinable_range range{r, sep};
// 	return lib::format("{}", range);
// 	#endif
// }

// ROCSOLVER_END_NAMESPACE
