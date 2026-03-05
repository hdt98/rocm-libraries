#pragma once
#include <type_traits>

namespace mint {

using std::remove_cv;
using std::remove_cv_t;

using std::remove_reference;
using std::remove_reference_t;

using std::remove_cvref;
using std::remove_cvref_t;

using std::is_same;
using std::is_same_v;

using std::is_const;
using std::is_const_v;

using std::is_fundamental;
using std::is_fundamental_v;

using std::is_default_constructible;
using std::is_default_constructible_v;

using std::is_empty;
using std::is_empty_v;

using std::unwrap_ref_decay_t;

using std::is_integral;
using std::is_integral_v;

using std::is_floating_point;
using std::is_floating_point_v;

} // namespace mint
