#pragma once

#include <string>
#include <string_view>

// concaternate the two arguments, evaluating them first if they are macros
#define ROCSOLVER_CONCAT2_HELPER(a, b) a##b
#define ROCSOLVER_CONCAT2(a, b) ROCSOLVER_CONCAT2_HELPER(a, b)

#define ROCSOLVER_CONCAT4_HELPER(a, b, c, d) a##b##c##d
#define ROCSOLVER_CONCAT4(a, b, c, d) ROCSOLVER_CONCAT4_HELPER(a, b, c, d)

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

#ifdef USE_LIB_FMT
    #include <fmt/base.h>
    #include <fmt/format.h>
    #include <fmt/ostream.h>
	#include <fmt/ranges.h>
	#define LIB_NAMESPACE fmt
#else
    #include <format>
    #include <print>
    #include <ostream>
    
	#include <ranges>
	#define LIB_NAMESPACE std
#endif

#ifdef USE_LIB_FMT
#define LIB_NAMESPACE fmt
#else
#define LIB_NAMESPACE std
#endif

// Aliased printing
#ifndef USE_FMT_LIB
    template<typename Range>
    struct joinable_range{
	const Range& range;
	const std::string_view sep;
    };

    template<typename Range, typename CharT>
    struct LIB_NAMESPACE::formatter<joinable_range<Range>, CharT>{
	    constexpr auto parse(std::format_parse_context& ctx){
		auto it = ctx.begin();
		auto end = ctx.end();

		if (it != ctx.end() && *it != '}')
			throw std::format_error("Invalid format args provided for rocsolver::join. The format string '{}' must be used. \n \
					The joinable_range type should be privately used by rocsolver::join.");

		return it;
	    }

	    template<typename FormatContext>
	    auto format(const joinable_range<Range>& jr, FormatContext& ctx) const {
		auto it = std::begin(jr.range);
		auto end = std::end(jr.range);

		auto out = ctx.out();
		if (it != end){
			out = std::format_to(out, "{}", *it);
			++it;
		}

		for (;it != end; ++it){
			out = std::format_to(out, "{}", jr.sep);
			out = std::format_to(out, "{}", *it);
		}

		return out;
	    }

    };
#endif

ROCSOLVER_BEGIN_NAMESPACE

template<class... Args>
void print(LIB_NAMESPACE::format_string<Args...> fmt, Args&&... args){
	LIB_NAMESPACE::print(fmt, std::forward<Args>(args)...);
}

template<class... Args>
void print(FILE* stream, LIB_NAMESPACE::format_string<Args...> fmt, Args&&... args){
	LIB_NAMESPACE::print(stream, fmt, std::forward<Args>(args)...);
}

template<class... Args>
void print(std::ostream& stream, LIB_NAMESPACE::format_string<Args...> fmt, Args&&... args){
	LIB_NAMESPACE::print(stream, fmt, std::forward<Args>(args)...);
}

template<class... Args>
std::string format(LIB_NAMESPACE::format_string<Args...> fmt, Args&&... args){
	return LIB_NAMESPACE::format(fmt, std::forward<Args>(args)...);
}

template <typename Range>
std::string join(Range&& r, std::string_view sep){
    #ifdef USE_FMT_LIB
	return LIB_NAMESPACE::join(r, sep);
	#else
	joinable_range range{r, sep};
	return LIB_NAMESPACE::format("{}", range);
	#endif
}
/*
template<typename T>
concept TupleType = std::is_tuple_v<T>;

template <TupleType T>
std::string join(T&& tup, std::string_view sep){
	#ifdef USE_FMT_LIB
	return LIB_NAMESAPCE::join(tup, sep);
	#else
	LIB_NAMESPACE::formatter<T> formatter;
	formatter.set_separator(sep);
	return LIB_NAMESPACE::format("{}", tup);
	#endif
}
*/

template<class... Args>
std::string join(std::tuple<Args...>&& tup, std::string_view sep){
	#ifdef USE_FMT_LIB
	return LIB_NAMESAPCE::join(tup, sep);
	#else
	LIB_NAMESPACE::formatter<std::tuple<Args...>> formatter;
	formatter.set_separator(sep);
	return LIB_NAMESPACE::format("{}", tup);
	#endif
}
ROCSOLVER_END_NAMESPACE
