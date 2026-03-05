#pragma once
#include <mint/core/index_t.h>
#include <mint/core/integer_sequence.h>

namespace mint {

// pack_param create an equivalent class template "packed_templ" from class
// template "unpacked_templ".
/*
 // "unpacked_templ" class template with non-typed parameter pack "kXs..."
 template <auto... kXs>
 struct unpacked_templ {};

 // "packed_templ" class template with non-typed containter parameter "kXs"
 template <auto kXs>
 using packed_templ = pack_param<upacked_templ, kXs>::type;

 // unpacked_templ and packed_templ are eqivalent, which can be instantiated
 // into the same type
 // For example:
 using type0 = unpacked_templ<1, 2.f, 's'>;
 using type1 = packed_templ<make_tuple(1, 2.f, 's')>;
 static_assert(is_same_v<type0, type1>);
 // For example:
 using type0 = unpacked_templ<1, 2, 3>;
 using type1 = packed_templ<array<index_t, 3>{1, 2, 3}>;
 static_assert(is_same_v<type0, type1>);
*/
template <template <auto...> class Templ, auto kXs>
struct pack_param {
 private:
  template <class Seq>
  struct impl;

  template <index_t... kSeq>
  struct impl<index_sequence<kSeq...>> {
    using type = Templ<kXs.template at<kSeq>()...>;
  };

 public:
  using type = impl<make_index_sequence<kXs.size()>>::type;
};

} // namespace mint
