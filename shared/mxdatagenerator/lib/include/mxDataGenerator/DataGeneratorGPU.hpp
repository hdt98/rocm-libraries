// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Header-only HIP/GPU backend mirroring DataGenerator<DTYPE> (CPU). The body
// is HIP-guarded so non-HIP TUs can include the rest of mxDataGenerator
// without dragging in HIP. Gated by CMake option MXDATAGENERATOR_ENABLE_GPU.

#include "DataGenerator.hpp"
#include "PreSwizzle.hpp"
#include "dataTypeInfo.hpp"

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace DGen
{
    /**
     * @brief HIP/GPU backend for MX data generation.
     *
     * Mirrors the public interface of DGen::DataGenerator but uses a separate
     * device PRNG: bytes are deterministic per-seed within this backend but
     * will not match the CPU backend bit-for-bit (statistics match).
     *
     * Supported data types: ocp_e2m1_mxfp4 (+ e4m3/e5m3 scale variants),
     * ocp_e2m3_mxfp6, ocp_e3m2_mxfp6, ocp_e4m3_mxfp8, ocp_e5m2_mxfp8.
     * Scale formats: E8M0, E4M3, E5M3. Init modes: Bounded,
     * BoundedAlternatingSign, Unbounded, Identity, Ones, Zeros, Sequential,
     * RowIndex, ColIndex, Checkerboard, ScaledDiagonal,
     * TrigonometricFromFloat, NormalFromFloat.
     */
    template <typename DTYPE>
    class DataGeneratorGPU
    {
    public:
        DataGeneratorGPU() = default;
        ~DataGeneratorGPU();

        DataGeneratorGPU(DataGeneratorGPU const&)            = delete;
        DataGeneratorGPU& operator=(DataGeneratorGPU const&) = delete;
        DataGeneratorGPU(DataGeneratorGPU&&)                 = delete;
        DataGeneratorGPU& operator=(DataGeneratorGPU&&)      = delete;

        /// Set the seed used by the device PRNG.
        void setSeed(uint32_t seed)
        {
            m_seed = seed;
        }

        /**
         * @brief Allocate device buffers and generate MX data.
         *
         * `sizes` must be 2D (the only shape used by current callers). The
         * fastest-varying dimension (sizes[0]) must be a multiple of
         * `options.blockScaling`.
         */
        DataGeneratorGPU& generate(std::vector<index_t>        sizes,
                                   std::vector<index_t>        strides,
                                   DataGeneratorOptions const& options,
                                   hipStream_t                 stream = nullptr);

        /**
         * @brief Generate MX data into caller-owned device buffers.
         *
         * `devData` must point to at least `getDataBufferBytes(sizes,
         * options)` bytes of device memory; `devScale` must point to at least
         * `getScaleBufferBytes(sizes, options)` bytes of device memory (or
         * may be `nullptr` for unscaled DTYPEs).
         */
        void generateInto(void*                       devData,
                          void*                       devScale,
                          std::vector<index_t>        sizes,
                          std::vector<index_t>        strides,
                          DataGeneratorOptions const& options,
                          hipStream_t                 stream = nullptr);

        /// Device pointer to the packed data buffer (valid after `generate`).
        uint8_t const* getDataBytesDevice() const
        {
            return m_dataDevice;
        }

        /// Device pointer to the packed scale buffer (valid after `generate`).
        uint8_t const* getScaleBytesDevice() const
        {
            return m_scaleDevice;
        }

        /// Copy the packed data buffer back to host memory.
        std::vector<uint8_t> getDataBytes() const;

        /// Copy the packed scale buffer back to host memory.
        std::vector<uint8_t> getScaleBytes() const;

        /// Materialise the reference float vector on the host (for validation).
        std::vector<float> getReferenceFloat() const;

        /**
         * @brief Apply `preSwizzleScalesGFX950` to the scale buffer in-place.
         *
         * `scaleSizes` is `{numScaleRows, numScaleCols}` (the same arguments
         * as the host helper of the same name). After this call,
         * `getScaleBytesDevice()` refers to the swizzled layout.
         */
        void preSwizzleScalesGFX950Device(std::vector<size_t> const& scaleSizes,
                                          hipStream_t                stream = nullptr);

        /// Compute the byte size of the packed data buffer for given sizes/options.
        static size_t getDataBufferBytes(std::vector<index_t> const& sizes,
                                         DataGeneratorOptions const& options);

        /// Compute the byte size of the packed scale buffer for given sizes/options.
        static size_t getScaleBufferBytes(std::vector<index_t> const& sizes,
                                          DataGeneratorOptions const& options);

    private:
        uint32_t m_seed = 1713573848;

        DataGeneratorOptions m_options;
        std::vector<index_t> m_sizes;

        size_t m_dataBufferBytes  = 0;
        size_t m_scaleBufferBytes = 0;

        // Owned device buffers (populated by `generate`, not by `generateInto`).
        uint8_t* m_dataDevice  = nullptr;
        uint8_t* m_scaleDevice = nullptr;
        bool     m_ownsBuffers = false;
    };

} // namespace DGen

#include "DataGeneratorGPU_impl.hpp"

#endif // HIP guard
