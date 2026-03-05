#pragma once

#ifndef __HIP_DEVICE_COMPILE__ // for host code
#define AMD_BUFFER_RESOURCE_3RD_DWORD -1
#elif defined(__gfx942__) || defined(__gfx950__)
#define AMD_BUFFER_RESOURCE_3RD_DWORD 0x00020000
#endif

// memory coherency bit for buffer store/load instruction
// check ISA manual for each GFX target
// e.g. for
// https://www.amd.com/system/files/TechDocs/instinct-mi200-cdna2-instruction-set-architecture.pdf,
// page 67~68
enum struct AmdBufferCoherenceEnum {
  DefaultCoherence = 0, // default value
  GLC = 1,
  SLC = 2,
  GLC_SLC = 3,
  // gfx94: bit 0 = sc0, bit 1 = nt, bit 3 = swz, bit 4 = sc1
  // SC[1:0] System Cache level: 0=wave, 1=group, 2=device, 3=system
  // NT Non-Temporal: 0=expect temporal reuse; 1=do not expect temporal reuse
  WAVE_NT0 = 0,
  WAVE_NT1 = 2,
  GROUP_NT0 = 1,
  GROUP_NT1 = 3,
  DEVICE_NT0 = 8,
  DEVICE_NT1 = 10,
  SYSTEM_NT0 = 9,
  SYSTEM_NT1 = 11,
};

/*using int8_t = char;*/
using int32_t = int;

typedef int8_t int8x2_t __attribute__((ext_vector_type(2)));
typedef int8_t int8x4_t __attribute__((ext_vector_type(4)));
typedef int8_t int8x8_t __attribute__((ext_vector_type(8)));
typedef int8_t int8x16_t __attribute__((ext_vector_type(16)));

typedef int32_t int32x2_t __attribute__((ext_vector_type(2)));
typedef int32_t int32x4_t __attribute__((ext_vector_type(4)));

namespace mint {

// buffer load i8
__device__ int8_t llvm_amdgcn_raw_buffer_load_i8(
    int32x4_t srsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i8");

// buffer load i16
__device__ int16_t llvm_amdgcn_raw_buffer_load_i16(
    int32x4_t srsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i16");

// buffer load i32
__device__ int32_t llvm_amdgcn_raw_buffer_load_i32(
    int32x4_t srsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i32");

__device__ int32x2_t llvm_amdgcn_raw_buffer_load_i32x2(
    int32x4_t srsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2i32");

__device__ int32x4_t llvm_amdgcn_raw_buffer_load_i32x4(
    int32x4_t srsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

using as3_uint32_ptr = uint32_t __attribute__((address_space(3)))*;

#define AMD_LDS_ADDR __attribute__((address_space(3)))

template <typename T>
union BufferResource {
  __device__ constexpr BufferResource() : content{} {}

  // 128 bit SGPRs to supply buffer resource in buffer instructions
  // https://rocm-documentation.readthedocs.io/en/latest/GCN_ISA_Manuals/testdocbook.html#vector-memory-buffer-instructions
  int32x4_t content;
  custom_vector_type<T*, 2> address;
  custom_vector_type<int32_t, 4> range;
  custom_vector_type<int32_t, 4> config;
};

template <typename T>
__device__ int32x4_t
make_wave_buffer_resource(T* p_wave, index_t element_space_size) {
  BufferResource<T> wave_buffer_resource;

  // wavewise base address (64 bit)
  wave_buffer_resource.address[0_ic] = const_cast<remove_cv_t<T>*>(p_wave);
  // wavewise range (32 bit)
  wave_buffer_resource.range[2_ic] = element_space_size * sizeof(T);
  // wavewise setting (32 bit)
  wave_buffer_resource.config[3_ic] = AMD_BUFFER_RESOURCE_3RD_DWORD;

  return wave_buffer_resource.content;
}

template <
    typename T,
    index_t N,
    AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ vector_type<T, N> amd_buffer_load_impl_raw(
    int32x4_t src_wave_buffer_resource,
    index_t src_thread_addr_offset,
    index_t src_wave_addr_offset) {
  using r_t = vector_type<T, N>;

  constexpr index_t NBYTES = sizeof(T) * N;
  static_assert(
      NBYTES == 1 || NBYTES == 2 || NBYTES == 4 || NBYTES == 8 || NBYTES == 16,
      "wrong! not implemented");

  if constexpr (NBYTES == 1) {
    return llvm_amdgcn_raw_buffer_load_i8(
        src_wave_buffer_resource,
        src_thread_addr_offset,
        src_wave_addr_offset,
        static_cast<index_t>(coherence));

  } else if constexpr (NBYTES == 2) {
    int16_t tmp = llvm_amdgcn_raw_buffer_load_i16(
        src_wave_buffer_resource,
        src_thread_addr_offset,
        src_wave_addr_offset,
        static_cast<index_t>(coherence));

    return bit_cast<r_t>(tmp);
  } else if constexpr (NBYTES == 4) {
    int32_t tmp = llvm_amdgcn_raw_buffer_load_i32(
        src_wave_buffer_resource,
        src_thread_addr_offset,
        src_wave_addr_offset,
        static_cast<index_t>(coherence));

    return bit_cast<r_t>(tmp);
  } else if constexpr (NBYTES == 8) {
    int32x2_t tmp = llvm_amdgcn_raw_buffer_load_i32x2(
        src_wave_buffer_resource,
        src_thread_addr_offset,
        src_wave_addr_offset,
        static_cast<index_t>(coherence));

    return bit_cast<r_t>(tmp);
  } else if constexpr (NBYTES == 16) {
    int32x4_t tmp = llvm_amdgcn_raw_buffer_load_i32x4(
        src_wave_buffer_resource,
        src_thread_addr_offset,
        src_wave_addr_offset,
        static_cast<index_t>(coherence));
    return bit_cast<r_t>(tmp);
  }
}

template <
    typename T,
    index_t N,
    AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ vector_type<T, N> amd_buffer_load_impl(
    int32x4_t src_wave_buffer_resource,
    index_t src_thread_addr_offset,
    index_t src_wave_addr_offset) {
  static_assert(
      (is_same<T, float>::value && (N == 1 || N == 2 || N == 4)) ||
          (is_same<T, fp16_t>::value && (N == 1 || N == 2 || N == 4 || N == 8)),
      "wrong! not implemented");

  using r_t = vector_type<T, N>;
  auto raw_data = amd_buffer_load_impl_raw<T, N, coherence>(
      src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset);
  return bit_cast<r_t>(raw_data);
}

// buffer_load requires:
//   1) p_src_wave must point to global memory space
//   2) p_src_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <
    typename T,
    index_t N,
    AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ vector_type<T, N> amd_buffer_load_invalid_element_return_zero(
    const T* p_src_wave,
    index_t src_thread_element_offset,
    index_t src_element_space_size,
    bool src_thread_element_valid = true) {
  const int32x4_t src_wave_buffer_resource =
      make_wave_buffer_resource(p_src_wave, src_element_space_size);

  index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);

  return src_thread_element_valid
      ? amd_buffer_load_impl<T, N, coherence>(
            src_wave_buffer_resource, src_thread_addr_offset, 0)
      : vector_type<T, N>{};
}

template <bool pre_nop = false>
__device__ void async_buffer_load_dword(
    void* smem,
    int32x4_t rsrc,
    index_t voffset,
    index_t /*soffset*/,
    index_t ioffset /*max 0xFFF*/,
    index_t /*flag*/ = 0,
    bool_constant<pre_nop> = {}) {
  if constexpr (pre_nop)
    asm volatile(
        "s_nop 4\n"
        "buffer_load_dword %1, %2, 0 offen offset:%3 lds"
        : "=r"(smem) /*dummy dependency for smem*/
        : "v"(voffset), "s"(rsrc), "n"(ioffset)
        : "memory");
  else
    asm volatile(
        "\n"
        "buffer_load_dword %1, %2, 0 offen offset:%3 lds"
        : "=r"(smem) /*dummy dependency for smem*/
        : "v"(voffset), "s"(rsrc), "n"(ioffset)
        : "memory");
}

template <bool pre_nop = false>
__device__ void async_buffer_load_dwordx4(
    void* smem,
    int32x4_t rsrc,
    index_t voffset,
    index_t /*soffset*/,
    index_t ioffset /*max 0xFFF*/,
    index_t /*flag*/ = 0,
    bool_constant<pre_nop> = {}) {
  if constexpr (pre_nop)
    asm volatile(
        "s_nop 4\n"
        "buffer_load_dwordx4 %1, %2, 0 offen offset:%3 lds"
        : "=r"(smem) /*dummy dependency for smem*/
        : "v"(voffset), "s"(rsrc), "n"(ioffset)
        : "memory");
  else
    asm volatile(
        "\n"
        "buffer_load_dwordx4 %1, %2, 0 offen offset:%3 lds"
        : "=r"(smem) /*dummy dependency for smem*/
        : "v"(voffset), "s"(rsrc), "n"(ioffset)
        : "memory");
}

__device__ void llvm_amdgcn_raw_buffer_load_lds(
    int32x4_t rsrc,
    __attribute__((address_space(3))) uint32_t* lds_ptr,
    index_t element_size,
    index_t voffset,
    index_t soffset,
    index_t glc,
    index_t slc) __asm("llvm.amdgcn.raw.buffer.load.lds");

__device__ void m0_set_with_memory(index_t v) {
  asm volatile("s_mov_b32 m0, %0" : : "s"(v) : "memory");
}

template <typename T, index_t N, bool pre_nop = false>
__device__ void amd_async_buffer_load_impl(
    T* smem,
    int32x4_t src_wave_buffer_resource,
    index_t src_thread_addr_offset,
    index_t src_wave_addr_offset,
    index_t src_immediate_addr_offset = 0,
    bool_constant<pre_nop> = {}) {
  if constexpr (sizeof(T) * N == 16)
    async_buffer_load_dwordx4(
        smem,
        src_wave_buffer_resource,
        src_thread_addr_offset * sizeof(T),
        src_wave_addr_offset,
        src_immediate_addr_offset,
        0,
        bool_constant<pre_nop>{});
  else if constexpr (sizeof(T) * N == 4)
    async_buffer_load_dword(
        smem,
        src_wave_buffer_resource,
        src_thread_addr_offset * sizeof(T),
        src_wave_addr_offset,
        src_immediate_addr_offset,
        0,
        bool_constant<pre_nop>{});
  else
    static_assert(0);
}

// i is offset of T, not X. i should be aligned to X
template <typename T, index_t N>
__device__ constexpr auto amd_buffer_load_async_copy(
    const T* p_src_wave,
    T* smem,
    index_t src_thread_offset,
    index_t src_wave_offset,
    index_t dst_wave_offset,
    index_t src_element_space_size) {
  const int32x4_t src_wave_buffer_resource =
      make_wave_buffer_resource(p_src_wave, src_element_space_size);

  index_t offset = reinterpret_cast<int64_t>(smem) - 0x1000000000000;

  m0_set_with_memory((dst_wave_offset) * sizeof(T) + offset);

  amd_async_buffer_load_impl<remove_cvref_t<T>, N>(
      smem, src_wave_buffer_resource, src_thread_offset, src_wave_offset);
}

} // namespace mint
