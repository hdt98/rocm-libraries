#include "tolerance.h"

#include <stdexcept>

namespace hipconv
{

float get_unit_roundoff(DataType dtype)
{
    switch(dtype)
    {
    case DataType::fp16:
        return 0x1p-11f; // 2^{-(10+1)}, 10 mantissa bits
    case DataType::bf16:
        return 0x1p-8f; // 2^{-(7+1)},  7 mantissa bits
    case DataType::fp32:
        return 0x1p-24f; // 2^{-(23+1)}, 23 mantissa bits
    default:
        throw std::logic_error("Data-type not supported");
    }
}

size_t get_accumulation_depth(const Conv2dParams& par)
{
    if(par.direction == Direction::Wgrad)
        return static_cast<size_t>(par.n) * par.p * par.q;
    if(par.direction == Direction::Fprop)
        return static_cast<size_t>(par.channels_per_group()) * par.kh * par.kw;
    if(par.direction == Direction::Dgrad)
        return static_cast<size_t>(par.filters_per_group()) * par.kh * par.kw;
    throw std::runtime_error("Unexpected direction.");
}

// Error bound for mixed-precision matrix multiply (low-precision inputs, fp32 accumulation).
//
// From Blanchard, Higham, Lopez, Mary, Pranesh, "Mixed Precision Block Fused Multiply-Add:
// Error Analysis and Application to GPU Tensor Cores", SIAM J. Sci. Comput., 2020.
//
// By Theorem 3.1 / Table 3.3 (TC32, A,B in u16), the accumulation error satisfies:
//
//   |C_hat - C|_ij <= n * u_high * (|A| * |B|)_ij
//
// where n is the inner dimension (accumulation depth), u_high = 2^{-24}, and
// |A|*|B| is the product of the componentwise absolute values.
//
// Both the reference and kernel outputs are independently rounded to the output
// precision (u_low), so comparing them introduces up to 2 * u_low relative error
// (triangle inequality over two independent roundings). The total bound is:
//
//   rtol = 2 * u_low + n * u_high
//
// For wgrad, the output is fp32 so u_low = u_high and this term is negligible.
void get_grouped_tolerance(const Conv2dParams& par, float& atol, float& rtol)
{
    auto n      = get_accumulation_depth(par);
    auto u_low  = get_unit_roundoff(par.out_type);
    auto u_high = get_unit_roundoff(DataType::fp32);
    rtol        = 2 * u_low + n * u_high;
    atol        = u_high;
}

} // namespace hipconv
