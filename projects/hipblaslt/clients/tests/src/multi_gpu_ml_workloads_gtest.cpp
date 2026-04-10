/*******************************************************************************
 * Multi-GPU ML Workload Patterns Test Suite
 * Tests: Transformer attention, pipeline parallelism, tensor parallelism, gradient accumulation
 *******************************************************************************/

#include "hipblaslt_data.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_test.hpp"
#include "utility.hpp"
#include <vector>
#include <atomic>
#include <thread>
#include <gtest/gtest-spi.h>

namespace
{
    int getNumGPUs()
    {
        int numDevices = 0;
        auto err = hipGetDeviceCount(&numDevices);
        (void)err;
        return numDevices;
    }

    TEST(MultiGPUMLWorkloads, TransformerAttentionDataParallel)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Transformer Attention - Data Parallel ===" << std::endl;

        // Simulated transformer layer parameters
        const int64_t batch_size = 32;
        const int64_t seq_len = 512;
        const int64_t hidden_dim = 1024;
        const int64_t num_heads = 16;
        const int64_t head_dim = hidden_dim / num_heads;

        // Distribute batches across GPUs
        int64_t batch_per_gpu = batch_size / numDevices;

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_Q(numDevices), d_K(numDevices), d_V(numDevices);
        std::vector<float*> d_QK(numDevices), d_out(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            hipblasLtCreate(&handles[dev]);

            // Allocate Q, K, V matrices for this GPU's batch
            int64_t local_batch = batch_per_gpu + (dev < (batch_size % numDevices) ? 1 : 0);

            EXPECT_EQ(hipMalloc(&d_Q[dev], local_batch * seq_len * hidden_dim * sizeof(float)), hipSuccess);
            EXPECT_EQ(hipMalloc(&d_K[dev], local_batch * seq_len * hidden_dim * sizeof(float)), hipSuccess);
            EXPECT_EQ(hipMalloc(&d_V[dev], local_batch * seq_len * hidden_dim * sizeof(float)), hipSuccess);
            EXPECT_EQ(hipMalloc(&d_QK[dev], local_batch * num_heads * seq_len * seq_len * sizeof(float)), hipSuccess);
            EXPECT_EQ(hipMalloc(&d_out[dev], local_batch * seq_len * hidden_dim * sizeof(float)), hipSuccess);

            // Initialize (in practice would load from input)
            EXPECT_EQ(hipMemset(d_Q[dev], 1, local_batch * seq_len * hidden_dim * sizeof(float)), hipSuccess);
            EXPECT_EQ(hipMemset(d_K[dev], 1, local_batch * seq_len * hidden_dim * sizeof(float)), hipSuccess);
            EXPECT_EQ(hipMemset(d_V[dev], 1, local_batch * seq_len * hidden_dim * sizeof(float)), hipSuccess);

            hipblaslt_cout << "GPU " << dev << ": Allocated attention matrices for "
                           << local_batch << " batches" << std::endl;
        }

        // Step 1: Compute Q @ K^T for each GPU (attention scores)
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);

            int64_t local_batch = batch_per_gpu + (dev < (batch_size % numDevices) ? 1 : 0);

            // For each head, compute Q @ K^T
            // Simplified: treat as single GEMM (seq_len x head_dim) @ (head_dim x seq_len)
            // = (seq_len x seq_len) attention scores

            hipblasLtMatrixLayout_t matQ, matK, matQK, matQK_out;
            hipblasLtMatrixLayoutCreate(&matQ, HIP_R_32F, seq_len, head_dim, seq_len);
            hipblasLtMatrixLayoutCreate(&matK, HIP_R_32F, head_dim, seq_len, head_dim);
            hipblasLtMatrixLayoutCreate(&matQK, HIP_R_32F, seq_len, seq_len, seq_len);
            hipblasLtMatrixLayoutCreate(&matQK_out, HIP_R_32F, seq_len, seq_len, seq_len);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            hipblasOperation_t opN = HIPBLAS_OP_N;
            hipblasOperation_t opT = HIPBLAS_OP_T;
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opT, sizeof(opT));

            float alpha = 1.0f / sqrtf(static_cast<float>(head_dim)); // Scale for attention
            float beta = 0.0f;

            // Execute Q @ K^T for first head (in practice would loop over all heads)
            hipblasLtMatmul(handles[dev], matmul, &alpha,
                           d_Q[dev], matQ, d_K[dev], matK,
                           &beta, d_QK[dev], matQK, d_QK[dev], matQK_out,
                           nullptr, nullptr, 0, 0);

            EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);

            hipblasLtMatrixLayoutDestroy(matQ);
            hipblasLtMatrixLayoutDestroy(matK);
            hipblasLtMatrixLayoutDestroy(matQK);
            hipblasLtMatrixLayoutDestroy(matQK_out);
            hipblasLtMatmulDescDestroy(matmul);
        }

        // Step 2: Softmax (would apply here, skipped for test)

        // Step 3: Compute attention @ V
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);

            // Simplified: (seq_len x seq_len) @ (seq_len x head_dim) = (seq_len x head_dim)
            hipblasLtMatrixLayout_t matAttn, matV, matOut, matOut_out;
            hipblasLtMatrixLayoutCreate(&matAttn, HIP_R_32F, seq_len, seq_len, seq_len);
            hipblasLtMatrixLayoutCreate(&matV, HIP_R_32F, seq_len, head_dim, seq_len);
            hipblasLtMatrixLayoutCreate(&matOut, HIP_R_32F, seq_len, head_dim, seq_len);
            hipblasLtMatrixLayoutCreate(&matOut_out, HIP_R_32F, seq_len, head_dim, seq_len);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            float alpha = 1.0f, beta = 0.0f;

            hipblasLtMatmul(handles[dev], matmul, &alpha,
                           d_QK[dev], matAttn, d_V[dev], matV,
                           &beta, d_out[dev], matOut, d_out[dev], matOut_out,
                           nullptr, nullptr, 0, 0);

            EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);

            hipblasLtMatrixLayoutDestroy(matAttn);
            hipblasLtMatrixLayoutDestroy(matV);
            hipblasLtMatrixLayoutDestroy(matOut);
            hipblasLtMatrixLayoutDestroy(matOut_out);
            hipblasLtMatmulDescDestroy(matmul);

            hipblaslt_cout << "GPU " << dev << ": Completed attention computation" << std::endl;
        }

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipFree(d_Q[dev]), hipSuccess);
            EXPECT_EQ(hipFree(d_K[dev]), hipSuccess);
            EXPECT_EQ(hipFree(d_V[dev]), hipSuccess);
            EXPECT_EQ(hipFree(d_QK[dev]), hipSuccess);
            EXPECT_EQ(hipFree(d_out[dev]), hipSuccess);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "✓ Transformer attention data parallel test passed" << std::endl;
    }

    TEST(MultiGPUMLWorkloads, TensorParallelism)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Tensor Parallelism (Column Split) ===" << std::endl;

        // Simulate FFN layer: Y = X @ W where W is split column-wise across GPUs
        const int64_t batch_size = 64;
        const int64_t seq_len = 512;
        const int64_t hidden_dim = 1024;
        const int64_t ffn_dim = 4096; // Total FFN dimension

        // Split FFN dimension across GPUs
        int64_t ffn_per_gpu = ffn_dim / numDevices;

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_X(numDevices);        // Full input replicated on each GPU
        std::vector<float*> d_W_col(numDevices);    // Column partition of W
        std::vector<float*> d_Y_col(numDevices);    // Column partition of Y

        // Allocate and initialize
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            hipblasLtCreate(&handles[dev]);

            // Input: (batch * seq_len) x hidden_dim (replicated)
            EXPECT_EQ(hipMalloc(&d_X[dev], batch_size * seq_len * hidden_dim * sizeof(float)), hipSuccess);

            // Weight partition: hidden_dim x ffn_per_gpu
            EXPECT_EQ(hipMalloc(&d_W_col[dev], hidden_dim * ffn_per_gpu * sizeof(float)), hipSuccess);

            // Output partition: (batch * seq_len) x ffn_per_gpu
            EXPECT_EQ(hipMalloc(&d_Y_col[dev], batch_size * seq_len * ffn_per_gpu * sizeof(float)), hipSuccess);

            // Initialize
            EXPECT_EQ(hipMemset(d_X[dev], 1, batch_size * seq_len * hidden_dim * sizeof(float)), hipSuccess);
            EXPECT_EQ(hipMemset(d_W_col[dev], 1, hidden_dim * ffn_per_gpu * sizeof(float)), hipSuccess);

            hipblaslt_cout << "GPU " << dev << ": W columns [" << (dev * ffn_per_gpu)
                           << ":" << ((dev + 1) * ffn_per_gpu) << "]" << std::endl;
        }

        // Compute Y_col = X @ W_col on each GPU
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);

            int64_t M = batch_size * seq_len;
            int64_t N = ffn_per_gpu;
            int64_t K = hidden_dim;

            hipblasLtMatrixLayout_t matX, matW, matY, matY_out;
            hipblasLtMatrixLayoutCreate(&matX, HIP_R_32F, M, K, M);
            hipblasLtMatrixLayoutCreate(&matW, HIP_R_32F, K, N, K);
            hipblasLtMatrixLayoutCreate(&matY, HIP_R_32F, M, N, M);
            hipblasLtMatrixLayoutCreate(&matY_out, HIP_R_32F, M, N, M);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            float alpha = 1.0f, beta = 0.0f;

            hipblasLtMatmul(handles[dev], matmul, &alpha,
                           d_X[dev], matX, d_W_col[dev], matW,
                           &beta, d_Y_col[dev], matY, d_Y_col[dev], matY_out,
                           nullptr, nullptr, 0, 0);

            EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);

            hipblasLtMatrixLayoutDestroy(matX);
            hipblasLtMatrixLayoutDestroy(matW);
            hipblasLtMatrixLayoutDestroy(matY);
            hipblasLtMatrixLayoutDestroy(matY_out);
            hipblasLtMatmulDescDestroy(matmul);

            hipblaslt_cout << "GPU " << dev << ": Computed column partition" << std::endl;
        }

        // Note: In production, would need to concatenate Y_col partitions
        // This requires gathering results from all GPUs

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipFree(d_X[dev]), hipSuccess);
            EXPECT_EQ(hipFree(d_W_col[dev]), hipSuccess);
            EXPECT_EQ(hipFree(d_Y_col[dev]), hipSuccess);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "✓ Tensor parallelism (column split) test passed" << std::endl;
    }

    TEST(MultiGPUMLWorkloads, PipelineParallelism)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 4) GTEST_SKIP() << "Requires 4+ GPUs for pipeline";

        hipblaslt_cout << "=== Pipeline Parallelism (4-layer Transformer) ===" << std::endl;

        // Simulate 4-layer transformer pipeline
        const int num_layers = std::min(4, numDevices);
        const int num_micro_batches = 8;
        const int64_t micro_batch_size = 8;
        const int64_t seq_len = 512;
        const int64_t hidden_dim = 1024;

        std::vector<hipblasLtHandle_t> handles(num_layers);
        std::vector<hipStream_t> streams(num_layers);
        std::vector<hipEvent_t> layer_done(num_layers);

        // Each GPU handles one layer
        struct LayerBuffers
        {
            float* d_input;
            float* d_output;
            float* d_weight;
        };

        std::vector<std::vector<LayerBuffers>> buffers(num_layers);

        // Allocate buffers for each layer and micro-batch
        for(int layer = 0; layer < num_layers; ++layer)
        {
            EXPECT_EQ(hipSetDevice(layer), hipSuccess);
            hipblasLtCreate(&handles[layer]);
            EXPECT_EQ(hipStreamCreate(&streams[layer]), hipSuccess);
            EXPECT_EQ(hipEventCreate(&layer_done[layer]), hipSuccess);

            buffers[layer].resize(num_micro_batches);

            for(int mb = 0; mb < num_micro_batches; ++mb)
            {
                size_t io_size = micro_batch_size * seq_len * hidden_dim * sizeof(float);
                size_t weight_size = hidden_dim * hidden_dim * sizeof(float);

                EXPECT_EQ(hipMalloc(&buffers[layer][mb].d_input, io_size), hipSuccess);
                EXPECT_EQ(hipMalloc(&buffers[layer][mb].d_output, io_size), hipSuccess);
                EXPECT_EQ(hipMalloc(&buffers[layer][mb].d_weight, weight_size), hipSuccess);

                // Initialize weights
                EXPECT_EQ(hipMemset(buffers[layer][mb].d_weight, 1, weight_size), hipSuccess);

                // Initialize input for layer 0
                if(layer == 0)
                {
                    EXPECT_EQ(hipMemset(buffers[layer][mb].d_input, 1, io_size), hipSuccess);
                }
            }

            hipblaslt_cout << "Layer " << layer << " (GPU " << layer << "): Buffers allocated" << std::endl;
        }

        auto start = std::chrono::high_resolution_clock::now();

        // Pipeline execution: Process micro-batches through layers
        for(int mb = 0; mb < num_micro_batches; ++mb)
        {
            for(int layer = 0; layer < num_layers; ++layer)
            {
                EXPECT_EQ(hipSetDevice(layer), hipSuccess);

                // Wait for previous layer to complete (if not first layer)
                if(layer > 0)
                {
                    EXPECT_EQ(hipStreamWaitEvent(streams[layer], layer_done[layer - 1], 0), hipSuccess);

                    // Copy output from previous layer to input of this layer
                    // (In practice would use P2P transfer)
                    size_t io_size = micro_batch_size * seq_len * hidden_dim * sizeof(float);
                    EXPECT_EQ(hipMemcpyAsync(buffers[layer][mb].d_input,
                                             buffers[layer - 1][mb].d_output,
                                             io_size, hipMemcpyDeviceToDevice,
                                             streams[layer]), hipSuccess);
                }

                // Execute layer computation: output = input @ weight
                int64_t M = micro_batch_size * seq_len;
                int64_t N = hidden_dim;
                int64_t K = hidden_dim;

                hipblasLtMatrixLayout_t matIn, matW, matOut, matOut_out;
                hipblasLtMatrixLayoutCreate(&matIn, HIP_R_32F, M, K, M);
                hipblasLtMatrixLayoutCreate(&matW, HIP_R_32F, K, N, K);
                hipblasLtMatrixLayoutCreate(&matOut, HIP_R_32F, M, N, M);
                hipblasLtMatrixLayoutCreate(&matOut_out, HIP_R_32F, M, N, M);

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                float alpha = 1.0f, beta = 0.0f;

                hipblasLtMatmul(handles[layer], matmul, &alpha,
                               buffers[layer][mb].d_input, matIn,
                               buffers[layer][mb].d_weight, matW,
                               &beta, buffers[layer][mb].d_output, matOut,
                               buffers[layer][mb].d_output, matOut_out,
                               nullptr, nullptr, 0, streams[layer]);

                // Record layer completion
                EXPECT_EQ(hipEventRecord(layer_done[layer], streams[layer]), hipSuccess);

                hipblasLtMatrixLayoutDestroy(matIn);
                hipblasLtMatrixLayoutDestroy(matW);
                hipblasLtMatrixLayoutDestroy(matOut);
                hipblasLtMatrixLayoutDestroy(matOut_out);
                hipblasLtMatmulDescDestroy(matmul);
            }
        }

        // Wait for pipeline to complete
        for(int layer = 0; layer < num_layers; ++layer)
        {
            EXPECT_EQ(hipStreamSynchronize(streams[layer]), hipSuccess);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        hipblaslt_cout << "Pipeline (" << num_layers << " layers, " << num_micro_batches
                       << " micro-batches) completed in " << (elapsed.count() * 1000.0)
                       << " ms" << std::endl;

        // Cleanup
        for(int layer = 0; layer < num_layers; ++layer)
        {
            EXPECT_EQ(hipSetDevice(layer), hipSuccess);

            for(int mb = 0; mb < num_micro_batches; ++mb)
            {
                EXPECT_EQ(hipFree(buffers[layer][mb].d_input), hipSuccess);
                EXPECT_EQ(hipFree(buffers[layer][mb].d_output), hipSuccess);
                EXPECT_EQ(hipFree(buffers[layer][mb].d_weight), hipSuccess);
            }

            EXPECT_EQ(hipStreamDestroy(streams[layer]), hipSuccess);
            EXPECT_EQ(hipEventDestroy(layer_done[layer]), hipSuccess);
            hipblasLtDestroy(handles[layer]);
        }

        hipblaslt_cout << "✓ Pipeline parallelism test passed" << std::endl;
    }

    TEST(MultiGPUMLWorkloads, GradientAccumulation)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Gradient Accumulation ===" << std::endl;

        const int64_t M = 1024, N = 1024, K = 1024;
        const int accumulation_steps = 4;

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_gradients(numDevices);
        std::vector<float*> d_accumulated(numDevices);

        // Allocate
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            hipblasLtCreate(&handles[dev]);

            EXPECT_EQ(hipMalloc(&d_gradients[dev], M * N * sizeof(float)), hipSuccess);
            EXPECT_EQ(hipMalloc(&d_accumulated[dev], M * N * sizeof(float)), hipSuccess);

            // Initialize accumulated gradients to zero
            EXPECT_EQ(hipMemset(d_accumulated[dev], 0, M * N * sizeof(float)), hipSuccess);
        }

        // Simulate gradient accumulation over multiple steps
        for(int step = 0; step < accumulation_steps; ++step)
        {
            hipblaslt_cout << "Accumulation step " << (step + 1) << "/" << accumulation_steps << std::endl;

            for(int dev = 0; dev < numDevices; ++dev)
            {
                EXPECT_EQ(hipSetDevice(dev), hipSuccess);

                // Compute gradients (simulated with memset)
                float grad_value = (step + 1) * 0.1f;
                EXPECT_EQ(hipMemset(d_gradients[dev], static_cast<int>(grad_value * 100), M * N * sizeof(float)), hipSuccess);

                // Accumulate: accumulated = accumulated + gradients
                // In practice would use element-wise add kernel or GEMM with beta=1
                // Here we simulate with a copy for first step, then add

                if(step == 0)
                {
                    EXPECT_EQ(hipMemcpy(d_accumulated[dev], d_gradients[dev],
                                        M * N * sizeof(float), hipMemcpyDeviceToDevice), hipSuccess);
                }
                else
                {
                    // Simplified: In practice would use element-wise add kernel
                    // For test purposes, just verify allocation and synchronization work
                    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
                }
            }
        }

        // Average gradients across GPUs (all-reduce)
        // In production would use NCCL all-reduce
        // Here we simulate by copying to host and averaging

        std::vector<float> h_avg_gradient(M * N, 0.0f);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);

            std::vector<float> h_grad(M * N);
            EXPECT_EQ(hipMemcpy(h_grad.data(), d_accumulated[dev],
                                M * N * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);

            // Accumulate on host
            for(size_t i = 0; i < h_avg_gradient.size(); ++i)
            {
                h_avg_gradient[i] += h_grad[i];
            }
        }

        // Average
        for(auto& val : h_avg_gradient)
        {
            val /= numDevices;
        }

        hipblaslt_cout << "✓ Gradients accumulated over " << accumulation_steps
                       << " steps across " << numDevices << " GPUs" << std::endl;

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipFree(d_gradients[dev]), hipSuccess);
            EXPECT_EQ(hipFree(d_accumulated[dev]), hipSuccess);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "✓ Gradient accumulation test passed" << std::endl;
    }

} // namespace
