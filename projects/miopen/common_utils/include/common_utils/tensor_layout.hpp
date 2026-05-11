// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_COMMON_UTILS_TENSOR_LAYOUT_HPP
#define GUARD_COMMON_UTILS_TENSOR_LAYOUT_HPP

#include <common_utils/errors.hpp>
#include <algorithm>
#include <iterator>
#include <map>
#include <numeric>
#include <string>
#include <vector>

namespace miopen {

template <typename T>
void tensor_layout_to_strides(const std::vector<T>& len,
                              const std::string& len_layout,
                              const std::string& layout,
                              std::vector<T>& strides)
{
    std::map<char, T> dim_to_len;
    std::transform(len.begin(),
                   len.end(),
                   len_layout.begin(),
                   std::inserter(dim_to_len, dim_to_len.end()),
                   [](T l, char dim) { return std::make_pair(dim, l); });

    std::transform(len_layout.begin(),
                   len_layout.end(),
                   std::back_inserter(strides),
                   [&layout, &dim_to_len](char cur_layout_char) {
                       auto pos = layout.find(cur_layout_char);
                       if(pos == std::string::npos)
                       {
                           COMMON_THROW(std::string("mismatched layout string - ").append(layout));
                       }
                       return std::accumulate(layout.begin() + pos + 1,
                                              layout.end(),
                                              static_cast<T>(1),
                                              [&dim_to_len](T accumulator, char l) {
                                                  return accumulator * dim_to_len[l];
                                              });
                   });
}

inline std::string tensor_layout_get_default(unsigned size)
{
    if(size == 4)
        return "NCHW";
    if(size == 5)
        return "NCDHW";
    return "";
}

} // namespace miopen

#endif // GUARD_COMMON_UTILS_TENSOR_LAYOUT_HPP
