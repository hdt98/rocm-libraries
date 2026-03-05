#pragma once

namespace mint {

// elementwise_nd_containers_inout
// "loop" is functor that takes functor "f" and container "ts...", use multi-dim
// iterator "iter..." and call f(ts.at(iter...))
template <class Loop, class F, class... Ts>
constexpr void
elementwise_nd_containers_inout(const Loop& loop, F&& f, Ts&&... ts) {
  loop([&f, &ts...](auto... iter) { f(ts.at(iter...)...); });
}

// similar to elementwise_nd_containers_inout, but also return a new container
template <class Ret, class Loop, class F, class... Ts>
constexpr Ret
elementwise_nd_containers_in(const Loop& loop, F&& f, Ts&&... ts) {
  Ret ret{};
  loop([&f, &ret, &ts...](auto... iter) {
    ret.at(iter...) = f(ts.at(iter...)...);
  });
  return ret;
}

// reduce_containers
// "acc" is accumulator
// "loop" is functor that takes functor "f" and container "ts...", use multi-dim
// iterator "iter..." and call f(acc, ts.at(iter...))
template <class Acc, class Loop, class F, class... Ts>
constexpr auto
reduce_nd_containers(const Loop& loop, F&& f, const Acc& init, Ts&&... ts) {
  Acc acc{init};
  loop([&acc, &f, &ts...](auto... iter) { f(acc, ts.at(iter...)...); });
  return acc;
}

template <class Loop, class Predicate, class... Ts>
constexpr index_t
count_true_nd_containers(const Loop& loop, Predicate&& f, Ts&&... ts) {
  index_t cnt = 0;
  loop([&cnt, &f, &ts...](auto... iter) { cnt += f(ts.at(iter...)...); });
  return cnt;
}

} // namespace mint
