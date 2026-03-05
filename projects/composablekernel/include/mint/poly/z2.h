#pragma once
#include <mint/core.h>

namespace mint::poly::z2 {

// examples:
//  z2_matrix<4, 5>
//        n = 0 1 2 3 4
//  m = 0    |1 0 0 0 0|
//  m = 1    |0 1 0 0 0|
//  m = 2    |0 0 1 0 0|
//  m = 3    |0 0 0 1 1|
template <index_t kM_, index_t kN_>
struct z2_matrix {
  static constexpr index_t kM = kM_;
  static constexpr index_t kN = kN_;

  // add a dumpy element at the end,
  //   otherwise nvcc would complain when array is empty
  bool data_[kM * kN + 1] = {};

  constexpr z2_matrix() = default;

  MINT_HOST_DEVICE constexpr z2_matrix(
      std::initializer_list<std::initializer_list<bool>> init) {
    index_t m = 0;
    for (auto row = init.begin(); row != init.end() && m < kM; row++, m++)
      std::copy(row->begin(), row->end(), &data_[m * kN]);
  }

  MINT_HOST_DEVICE constexpr bool operator==(const z2_matrix& a) const {
    for (index_t m = 0; m < kM; m++)
      for (index_t n = 0; n < kN; n++)
        if (data_[m * kN + n] != a(m, n))
          return false;
    return true;
  }

  MINT_HOST_DEVICE constexpr const bool& operator()(index_t m, index_t n)
      const {
    return data_[m * kN + n];
  }

  MINT_HOST_DEVICE constexpr bool& operator()(index_t m, index_t n) {
    return data_[m * kN + n];
  }

  MINT_HOST_DEVICE constexpr bool is_empty() const {
    return kM == 0 or kN == 0;
  }

  MINT_HOST_DEVICE constexpr void fill(bool v) {
    for (index_t m = 0; m < kM; m++)
      for (index_t n = 0; n < kN; n++)
        data_[m * kN + n] = v;
  }

  MINT_HOST_DEVICE void print() const {
    printf("z2_matrix, {size: %d %d\n", kM, kN);
    for (index_t m = 0; m < kM; m++) {
      if (m != 0)
        printf(" {");
      else
        printf("\n{{");
      for (index_t n = 0; n < kN; n++)
        printf("%d", (*this)(m, n));
      if (m != kM - 1)
        printf("}, \n");
      else
        printf("}}\n");
    }
    printf("}\n");
  }
};

// z2 matrix inversion: returns true if invertible, false otherwise.
template <index_t kN>
MINT_HOST_DEVICE constexpr mint::tuple<z2_matrix<kN, kN>, bool>
z2_try_matrix_invert(const z2_matrix<kN, kN>& input) {
  z2_matrix<kN, kN> output;

  // Copy input to a working matrix
  z2_matrix tmp = input;

  // Initialize output as identity
  for (index_t i = 0; i < kN; ++i)
    for (index_t j = 0; j < kN; ++j)
      output(i, j) = (i == j) ? 1 : 0;

  for (index_t col = 0; col < kN; ++col) {
    // Find tmp row with a 1 in the current column
    index_t pivot = col;
    while (pivot < kN && tmp(pivot, col) == 0)
      ++pivot;

    if (pivot == kN)
      return {output, false}; // Singular matrix

    // Swap rows if needed
    if (pivot != col) {
      for (index_t j = 0; j < kN; ++j) {
        std::swap(tmp(col, j), tmp(pivot, j));
        std::swap(output(col, j), output(pivot, j));
      }
    }

    // Eliminate all other 1s in this column
    for (index_t row = 0; row < kN; ++row) {
      if (row != col && tmp(row, col)) {
        for (index_t j = 0; j < kN; ++j) {
          tmp(row, j) ^= tmp(col, j);
          output(row, j) ^= output(col, j);
        }
      }
    }
  }

  return {output, true};
}

// z2 matrix inversion: returns true if invertible, false otherwise.
template <index_t kM, index_t kN>
MINT_HOST_DEVICE constexpr bool is_z2_matrix_invertible(
    const z2_matrix<kM, kN>& input) {
  if constexpr (kM != kN) {
    return false;
  } else {
    const auto [output, success] = z2_try_matrix_invert(input);
    (void)output;
    return success;
  }
}

// z2 matrix inversion
template <index_t kN>
MINT_HOST_DEVICE constexpr z2_matrix<kN, kN> z2_matrix_invert(
    const z2_matrix<kN, kN>& input) {
  const auto [output, success] = z2_try_matrix_invert(input);

  if (success)
    return output;
  else
    return z2_matrix<kN, kN>{}; // invert fail, return 0 matrix
}

// z2 matrix-matrix multiply
template <index_t kM, index_t kN, index_t kK>
MINT_HOST_DEVICE constexpr auto z2_matmul(
    const z2_matrix<kM, kK>& a,
    const z2_matrix<kK, kN>& b) {
  z2_matrix<kM, kN> c;

  for (index_t m = 0; m < kM; m++) {
    for (index_t n = 0; n < kN; n++) {
      c(m, n) = 0;
      for (index_t k = 0; k < kK; k++) {
        c(m, n) ^= a(m, k) && b(k, n);
      }
    }
  }

  return c;
}

// z2 matrix transpose
template <index_t kM, index_t kN>
MINT_HOST_DEVICE constexpr auto z2_matrix_transpose(
    const z2_matrix<kM, kN>& a) {
  z2_matrix<kN, kM> b;

  for (index_t m = 0; m < kM; m++)
    for (index_t n = 0; n < kN; n++)
      b(n, m) = a(m, n);

  return b;
}

// z2 matrix extract rows
template <index_t kM, index_t kN, class Rows>
MINT_HOST_DEVICE constexpr auto z2_matrix_extract_rows(
    const z2_matrix<kM, kN>& a,
    const Rows& rows) {
  z2_matrix<Rows::size(), kN> b;

  for (index_t i = 0; i < Rows::size(); i++) {
    index_t m = rows[i];
    for (index_t n = 0; n < kN; n++) {
      b(i, n) = a(m, n);
    }
  }

  return b;
}

// z2 matrix extract columns
template <index_t kM, index_t kN, class Columns>
MINT_HOST_DEVICE constexpr auto z2_matrix_extract_columns(
    const z2_matrix<kM, kN>& a,
    const Columns& cols) {
  z2_matrix<kM, Columns::size()> b;

  for (index_t i = 0; i < Columns::size(); i++) {
    index_t n = cols[i];
    for (index_t m = 0; m < kM; m++) {
      b(m, i) = a(m, n);
    }
  }

  return b;
}

// z2 matrix extract sub_matrix
template <index_t kM, index_t kN, class Rows, class Columns>
MINT_HOST_DEVICE constexpr auto z2_matrix_extract_sub_matrix(
    const z2_matrix<kM, kN>& a,
    const Rows& rows,
    const Columns& cols) {
  z2_matrix<Rows::size(), Columns::size()> b;
  for (index_t i = 0; i < Rows::size(); i++) {
    index_t m = rows[i];
    for (index_t j = 0; j < Columns::size(); j++) {
      index_t n = cols[j];
      b(i, j) = a(m, n);
    }
  }

  return b;
}

template <index_t kM, index_t kN, index_t kSubM, index_t kSubN>
MINT_HOST_DEVICE constexpr void z2_matrix_set_sub_matrix(
    z2_matrix<kM, kN>& dst,
    const z2_matrix<kSubM, kSubN>& src,
    index_t m_begin,
    index_t n_begin) {
  for (index_t m = 0; m < kSubM; m++)
    for (index_t n = 0; n < kSubN; n++)
      dst(m + m_begin, n + n_begin) = src(m, n);
}

#if 0
// z2_matrix-bitset dot-product
//   run-time z2_matrix version
// example:
//  (y: bitset<4>) = (a: z2_matrix<4, 3>) * (x: bitset<3>)
//
//      | x0  x1  x2|
//       -----------
// |y0| |a00 a01 a02|
// |y1| |a10 a11 a12|
// |y2| |a20 a21 a22|
// |y3| |a30 a31 a32|
//       -----------
//
// |y0|         |a00|          |a01|          |a02|
// |y1| = (x0 & |a10|) ^ (x1 & |a11|) ^ (x2 & |a12|)
// |y2|         |a20|          |a21|          |a22|
// |y3|         |a30|          |a31|          |a32|
template <index_t kM, index_t kN>
MINT_HOST_DEVICE constexpr auto z2_matrix_dot_bitset(
    const z2_matrix<kM, kN>& a,
    const bitset<kN>& x) {
  bitset<kM> y;
  bitset<kM> tmp;

  for (index_t n = 0; n < kN; n++) {
    for (index_t m = 0; m < kM; m++) {
      tmp[m] = x[n] && a(m, n);
    }
    y ^= tmp;
  }

  return y;
}
#endif

// z2_matrix-bitset dot-product
//   compile-time constant z2_matrix version
// example:
//  (y: bitset<4>) = (a: z2_matrix<4, 3>) * (x: bitset<3>)
//
//      | y0  y1  y2|
//       -----------
// |x0| |a00 a01 a02|
// |x1| |a10 a11 a12|
// |x2| |a20 a21 a22|
// |x3| |a30 a31 a32|
//       -----------
//
// |x0|         |a00|          |a01|          |a02|
// |x1| = (y0 & |a10|) ^ (y1 & |a11|) ^ (y2 & |a12|)
// |x2|         |a20|          |a21|          |a22|
// |x3|         |a30|          |a31|          |a32|
template <index_t kM, index_t kN, z2_matrix<kM, kN> kA>
MINT_HOST_DEVICE constexpr auto z2_matrix_dot_bitset(
    integral_constant<z2_matrix<kM, kN>, kA>,
    const bitset<kN>& x) {
  bitset<kM> y = {};

  for (index_t n = 0; n < kN; n++) {
    if (x[n]) {
      bitset<kM> vec = {};
      for (index_t m = 0; m < kM; m++) {
        vec[m] = kA(m, n);
      }
      y = y ^ vec;
    }
  }

  return y;
}

template <index_t kM, index_t kN>
MINT_HOST_DEVICE constexpr bool z2_matrix_allzero(const z2_matrix<kM, kN>& a) {
  for (index_t m = 0; m < kM; m++) {
    for (index_t n = 0; n < kN; n++) {
      if (a(m, n)) {
        return false;
      }
    }
  }
  return true;
}

template <index_t kM, index_t kNa, index_t kNb>
MINT_HOST_DEVICE constexpr bool z2_matrix_no_overlap_nonzero_rows(
    const z2_matrix<kM, kNa>& a,
    const z2_matrix<kM, kNb>& b) {
  for (index_t m = 0; m < kM; m++) {
    bool is_nonzero_a = false;
    bool is_nonzero_b = false;

    for (index_t n = 0; n < kNa; n++) {
      if (a(m, n)) {
        is_nonzero_a = true;
        continue;
      }
    }

    for (index_t n = 0; n < kNb; n++) {
      if (b(m, n)) {
        is_nonzero_b = true;
        continue;
      }
    }

    if (is_nonzero_a && is_nonzero_b)
      return false;
  }

  return true;
}

template <index_t kM, index_t kN, index_t kCol>
MINT_HOST_DEVICE constexpr auto z2_matrix_orthogonal_columns(
    const z2_matrix<kM, kN>& a,
    const nd_index<kCol>& cols) {
  // find all non-zero rows
  array<bool, kM> is_non_zero_rows;
  is_non_zero_rows.fill(false);

  for (index_t i = 0; i < kCol; i++) {
    index_t n = cols[i];
    for (index_t m = 0; m < kM; m++)
      if (a(m, n) != 0)
        is_non_zero_rows[m] = true;
  }

  array<bool, kN> is_orth_cols;
  is_orth_cols.fill(true);

  for (index_t i = 0; i < kCol; i++)
    is_orth_cols[cols[i]] = false;

  for (index_t n = 0; n < kN; n++)
    if (is_orth_cols[n])
      for (index_t m = 0; m < kM; m++)
        if (is_non_zero_rows[m] && a(m, n))
          is_orth_cols[n] = false;

  nd_index<kN> orth_cols;
  orth_cols.fill(-1);
  index_t cnt = 0;

  for (index_t n = 0; n < kN; n++)
    if (is_orth_cols[n])
      orth_cols[cnt++] = n;

  return mint::make_tuple(orth_cols, cnt);
}

template <
    index_t kM,
    index_t kN,
    index_t kCol,
    z2_matrix<kM, kN> kA,
    nd_index<kCol> kCols>
MINT_HOST_DEVICE constexpr auto z2_matrix_orthogonal_columns(
    integral_constant<z2_matrix<kM, kN>, kA>,
    integral_constant<nd_index<kCol>, kCols>) {
  constexpr auto b_cols = z2_matrix_orthogonal_columns(kA, kCols);

  constexpr auto b_tmp = b_cols[0_ic];
  constexpr auto cnt = b_cols[1_ic];

  nd_index<cnt> ret;

  for (index_t i = 0; i < cnt; i++)
    ret[i] = b_tmp[i];

  return ret;
}

template <index_t kN>
MINT_HOST_DEVICE constexpr auto make_z2_unity_matrix() {
  z2_matrix<kN, kN> ret;
  for (index_t i = 0; i < kN; i++) {
    ret(i, i) = true;
  }
  return ret;
}

// return z2 matrix "A", for bitsets y and x, y = A * x:
//   y[k] =
//     1) x[k] ^ x[l], (k, l) = (kIs[i], kJs[i]), i = 0, 1 ...
//     2) x[k], otherwise
template <index_t kN, auto kIs, auto kJs>
  requires(kIs.size() == kJs.size() && kIs.size() <= kN)
MINT_HOST_DEVICE constexpr auto make_z2_bitxor_matrix() {
  auto ret = make_z2_unity_matrix<kN>();
  for (index_t cnt = 0; cnt < kIs.size(); cnt++) {
    index_t i = kIs[cnt];
    index_t j = kJs[cnt];
    ret(j, i) ^= ret(j, j);
  }
  return ret;
}

template <
    index_t kMold,
    index_t kN,
    index_t kMnew,
    nd_index<kMnew> kNewToOldRows>
MINT_HOST_DEVICE constexpr auto z2_matrix_reorder_rows(
    const z2_matrix<kMold, kN>& a,
    integral_constant<nd_index<kMnew>, kNewToOldRows>) {
  z2_matrix<kMnew, kN> b;
  for (index_t m = 0; m < kMnew; m++)
    for (index_t n = 0; n < kN; n++)
      b(m, n) = a(kNewToOldRows[m], n);
  return b;
}

template <
    index_t kM,
    index_t kNold,
    index_t kNnew,
    nd_index<kNnew> kNewToOldCols>
MINT_HOST_DEVICE constexpr auto z2_matrix_reorder_columns(
    const z2_matrix<kM, kNold>& a,
    integral_constant<nd_index<kNnew>, kNewToOldCols>) {
  z2_matrix<kM, kNnew> b;
  for (index_t m = 0; m < kM; m++)
    for (index_t n = 0; n < kNnew; n++)
      b(m, n) = a(m, kNewToOldCols[n]);
  return b;
}

} // namespace mint::poly::z2
