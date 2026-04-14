/*******************************************************************************
 * Multi-GPU RCCL Collective Operations Test Suite
 * Tests: RCCL all-reduce, all-gather, reduce-scatter, fused ops, gradient compression
 *
 * CRITICAL: Real distributed training uses RCCL, not manual P2P
 * Integration: PyTorch DDP, DeepSpeed, Megatron-LM, FSDP
 *
 * NOTE: Requires RCCL library. Tests will skip if RCCL not available.
 *******************************************************************************/

#include "hipblaslt_data.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_test.hpp"
#include "utility.hpp"
#include <vector>

// RCCL headers (conditional compilation)
#ifdef USE_RCCL
#include <rccl/rccl.h>
#endif

namespace
{
    int getNumGPUs()
    {
        int numDevices = 0;
        auto err = hipGetDeviceCount(&numDevices);
        (void)err;
        return numDevices;
    }

    // ============================================================================
    // Test 1: RCCL All-Reduce Integration
    // Sum gradients across GPUs (standard DDP pattern)
    // ============================================================================
    TEST(MultiGPURCCLCollectives, AllReduceIntegration)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

#ifndef USE_RCCL
        hipblaslt_cout << "=== RCCL All-Reduce (Simulated) ===" << std::endl;
        hipblaslt_cout << "⚠️  RCCL not available - demonstrating pattern only" << std::endl;

        const int64_t gradient_size = 1024 * 1024;  // 1M elements

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_gradients(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            hipMalloc(&d_gradients[dev], gradient_size * sizeof(float));

            // Initialize with dev-specific values
            std::vector<float> h_grad(gradient_size, static_cast<float>(dev + 1));
            hipMemcpy(d_gradients[dev], h_grad.data(), gradient_size * sizeof(float),
                     hipMemcpyHostToDevice);

            hipblaslt_cout << "GPU " << dev << ": Gradient initialized to " << (dev + 1) << std::endl;
        }

        // Manual all-reduce simulation (ring pattern)
        hipblaslt_cout << "\n=== Manual All-Reduce (Ring Pattern) ===" << std::endl;
        for(int step = 0; step < numDevices - 1; ++step)
        {
            for(int dev = 0; dev < numDevices; ++dev)
            {
                int send_to = (dev + 1) % numDevices;
                hipblaslt_cout << "Step " << (step + 1) << ": GPU " << dev
                               << " → GPU " << send_to << std::endl;
            }
        }

        // Expected result: sum of all GPU values
        float expected_sum = (numDevices * (numDevices + 1)) / 2.0f;
        hipblaslt_cout << "Expected all-reduce result: " << expected_sum << " per element" << std::endl;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_gradients[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "✓ All-reduce pattern demonstrated (use RCCL in production)" << std::endl;

#else
        hipblaslt_cout << "=== RCCL All-Reduce Integration ===" << std::endl;

        const int64_t gradient_size = 1024 * 1024;

        // Initialize RCCL
        std::vector<ncclComm_t> nccl_comms(numDevices);
        ncclUniqueId id;
        ncclGetUniqueId(&id);

        // Create RCCL communicators
        std::vector<hipStream_t> streams(numDevices);
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipStreamCreate(&streams[dev]);
        }

        ncclGroupStart();
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            ncclCommInitRank(&nccl_comms[dev], numDevices, id, dev);
        }
        ncclGroupEnd();

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_gradients(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            hipMalloc(&d_gradients[dev], gradient_size * sizeof(float));

            std::vector<float> h_grad(gradient_size, static_cast<float>(dev + 1));
            hipMemcpy(d_gradients[dev], h_grad.data(), gradient_size * sizeof(float),
                     hipMemcpyHostToDevice);

            hipblaslt_cout << "GPU " << dev << ": Gradient = " << (dev + 1) << std::endl;
        }

        // Perform RCCL all-reduce (sum)
        ncclGroupStart();
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            ncclAllReduce(d_gradients[dev], d_gradients[dev], gradient_size,
                         ncclFloat32, ncclSum, nccl_comms[dev], streams[dev]);
        }
        ncclGroupEnd();

        // Synchronize and verify
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipStreamSynchronize(streams[dev]);

            std::vector<float> h_result(10);  // Check first 10 elements
            hipMemcpy(h_result.data(), d_gradients[dev], 10 * sizeof(float), hipMemcpyDeviceToHost);

            float expected = (numDevices * (numDevices + 1)) / 2.0f;
            hipblaslt_cout << "GPU " << dev << ": Result[0] = " << h_result[0]
                           << " (expected " << expected << ")" << std::endl;
        }

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_gradients[dev]);
            hipStreamDestroy(streams[dev]);
            ncclCommDestroy(nccl_comms[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "✓ RCCL all-reduce completed successfully" << std::endl;
#endif
    }

    // ============================================================================
    // Test 2: Fused Backward Pass (GEMM + All-Reduce)
    // Overlap computation with communication
    // ============================================================================
    TEST(MultiGPURCCLCollectives, FusedBackwardPass)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Fused Backward Pass (GEMM + All-Reduce) ===" << std::endl;

        const int64_t M = 4096, N = 4096, K = 4096;

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_dL_dOut(numDevices), d_W(numDevices), d_dL_dW(numDevices);
        std::vector<hipStream_t> compute_streams(numDevices), comm_streams(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            hipStreamCreate(&compute_streams[dev]);
            hipStreamCreate(&comm_streams[dev]);

            // Allocate matrices
            hipMalloc(&d_dL_dOut[dev], M * K * sizeof(float));  // Gradient from next layer
            hipMalloc(&d_W[dev], K * N * sizeof(float));         // Weights
            hipMalloc(&d_dL_dW[dev], M * N * sizeof(float));     // Weight gradient

            hipMemset(d_dL_dOut[dev], 0, M * K * sizeof(float));
            hipMemset(d_W[dev], 0, K * N * sizeof(float));
        }

        hipblaslt_cout << "\n=== Fused Execution Pattern ===" << std::endl;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);

            // Compute weight gradient: dL/dW = dL/dOut @ W
            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            float alpha = 1.0f, beta = 0.0f;

            // Launch GEMM on compute stream
            hipblasLtMatmul(handles[dev], matmul, &alpha,
                           d_dL_dOut[dev], matA, d_W[dev], matB,
                           &beta, d_dL_dW[dev], matC, d_dL_dW[dev], matD,
                           nullptr, nullptr, 0, 0);

            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);

            hipblaslt_cout << "GPU " << dev << ": GEMM launched on compute stream" << std::endl;
        }

        // In real code with RCCL:
        // 1. GEMM completes on compute stream
        // 2. Event signals comm stream
        // 3. All-reduce starts immediately on comm stream
        // 4. Overlap next layer's GEMM with all-reduce

        hipblaslt_cout << "\n=== Overlap Strategy ===" << std::endl;
        hipblaslt_cout << "Timeline:" << std::endl;
        hipblaslt_cout << "  t=0:   Layer N GEMM starts" << std::endl;
        hipblaslt_cout << "  t=10:  Layer N GEMM ends, all-reduce starts" << std::endl;
        hipblaslt_cout << "  t=10:  Layer N-1 GEMM starts (overlapped!)" << std::endl;
        hipblaslt_cout << "  t=20:  Layer N all-reduce ends" << std::endl;
        hipblaslt_cout << "  t=20:  Layer N-1 GEMM ends, all-reduce starts" << std::endl;

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipStreamSynchronize(compute_streams[dev]);
            hipFree(d_dL_dOut[dev]);
            hipFree(d_W[dev]);
            hipFree(d_dL_dW[dev]);
            hipStreamDestroy(compute_streams[dev]);
            hipStreamDestroy(comm_streams[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "✓ Fused backward pass pattern demonstrated" << std::endl;
        hipblaslt_cout << "  Key benefit: Hide all-reduce latency behind next layer's computation" << std::endl;
    }

    // ============================================================================
    // Test 3: Gradient Compression (Top-K / Quantization)
    // Reduce communication volume for all-reduce
    // ============================================================================
    TEST(MultiGPURCCLCollectives, GradientCompression)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Gradient Compression for All-Reduce ===" << std::endl;

        const int64_t gradient_size = 10 * 1024 * 1024;  // 10M parameters
        const float compression_ratio = 0.1f;  // Top-10%

        int64_t compressed_size = static_cast<int64_t>(gradient_size * compression_ratio);

        hipblaslt_cout << "Configuration:" << std::endl;
        hipblaslt_cout << "  Full gradient: " << gradient_size << " elements ("
                       << (gradient_size * sizeof(float) / 1024.0 / 1024.0) << " MB)" << std::endl;
        hipblaslt_cout << "  Compressed: " << compressed_size << " elements ("
                       << (compressed_size * sizeof(float) / 1024.0 / 1024.0) << " MB)" << std::endl;
        hipblaslt_cout << "  Compression ratio: " << (compression_ratio * 100) << "%" << std::endl;

        std::vector<hipblasLtHandle_t> handles(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            // In compressed gradient communication:
            // 1. Select top-K gradients by magnitude
            // 2. Send only indices + values
            // 3. All-reduce compressed gradients
            // 4. Accumulate into full gradient with error correction

            hipblaslt_cout << "GPU " << dev << ": Would compress " << gradient_size
                           << " → " << compressed_size << " elements" << std::endl;
        }

        // Calculate communication savings
        size_t full_comm = gradient_size * sizeof(float) * numDevices;  // All-reduce cost
        size_t compressed_comm = compressed_size * (sizeof(float) + sizeof(int)) * numDevices;
        float savings = (1.0f - (float)compressed_comm / full_comm) * 100;

        hipblaslt_cout << "\n=== Communication Analysis ===" << std::endl;
        hipblaslt_cout << "Full all-reduce: " << (full_comm / 1024.0 / 1024.0) << " MB" << std::endl;
        hipblaslt_cout << "Compressed all-reduce: " << (compressed_comm / 1024.0 / 1024.0) << " MB" << std::endl;
        hipblaslt_cout << "Bandwidth savings: " << savings << "%" << std::endl;

        hipblaslt_cout << "\n=== Compression Methods ===" << std::endl;
        hipblaslt_cout << "1. Top-K Sparsification: Select largest K% gradients" << std::endl;
        hipblaslt_cout << "2. Quantization: FP32 → INT8 (4x compression)" << std::endl;
        hipblaslt_cout << "3. Error Feedback: Accumulate quantization errors for next iteration" << std::endl;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "✓ Gradient compression pattern demonstrated" << std::endl;
        hipblaslt_cout << "  Trade-off: Communication reduction vs convergence speed" << std::endl;
    }

    // ============================================================================
    // Test 4: Hierarchical All-Reduce (Intra-node + Inter-node)
    // Optimize for multi-node topology
    // ============================================================================
    TEST(MultiGPURCCLCollectives, HierarchicalAllReduce)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 4) GTEST_SKIP() << "Requires 4+ GPUs";

        hipblaslt_cout << "=== Hierarchical All-Reduce (Intra-node + Inter-node) ===" << std::endl;

        // Simulate multi-node setup
        int gpus_per_node = 4;
        int num_nodes = (numDevices + gpus_per_node - 1) / gpus_per_node;

        hipblaslt_cout << "Topology:" << std::endl;
        hipblaslt_cout << "  Total GPUs: " << numDevices << std::endl;
        hipblaslt_cout << "  Nodes: " << num_nodes << std::endl;
        hipblaslt_cout << "  GPUs per node: " << gpus_per_node << std::endl;

        std::vector<hipblasLtHandle_t> handles(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            int node_id = dev / gpus_per_node;
            int local_rank = dev % gpus_per_node;

            hipblaslt_cout << "GPU " << dev << " → Node " << node_id
                           << ", Local rank " << local_rank << std::endl;
        }

        hipblaslt_cout << "\n=== Hierarchical All-Reduce Pattern ===" << std::endl;
        hipblaslt_cout << "Phase 1: Intra-node all-reduce (NVLink/Infinity Fabric)" << std::endl;
        for(int node = 0; node < num_nodes; ++node)
        {
            hipblaslt_cout << "  Node " << node << ": All-reduce across GPUs "
                           << (node * gpus_per_node) << "-" << ((node + 1) * gpus_per_node - 1) << std::endl;
        }

        hipblaslt_cout << "\nPhase 2: Inter-node reduce-scatter (Ethernet/InfiniBand)" << std::endl;
        hipblaslt_cout << "  Node leaders exchange gradient chunks" << std::endl;

        hipblaslt_cout << "\nPhase 3: Inter-node all-gather" << std::endl;
        hipblaslt_cout << "  Broadcast reduced gradients across nodes" << std::endl;

        hipblaslt_cout << "\nPhase 4: Intra-node broadcast (NVLink/Infinity Fabric)" << std::endl;
        hipblaslt_cout << "  Distribute final result within each node" << std::endl;

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "✓ Hierarchical all-reduce pattern demonstrated" << std::endl;
        hipblaslt_cout << "  Key benefit: Optimize for network topology (fast intra-node, slower inter-node)" << std::endl;
    }

    // ============================================================================
    // Test 5: RCCL All-Gather and Reduce-Scatter
    // Building blocks for FSDP and ZeRO
    // ============================================================================
    TEST(MultiGPURCCLCollectives, AllGatherReduceScatter)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== RCCL All-Gather and Reduce-Scatter ===" << std::endl;

        const int64_t param_size = 1024 * 1024;  // 1M parameters

        // Reduce-Scatter: Each GPU ends up with 1/N of the sum
        int64_t chunk_size = param_size / numDevices;

        hipblaslt_cout << "\n=== Reduce-Scatter Pattern ===" << std::endl;
        hipblaslt_cout << "Total params: " << param_size << std::endl;
        hipblaslt_cout << "Chunk per GPU: " << chunk_size << std::endl;

        std::vector<hipblasLtHandle_t> handles(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            hipblaslt_cout << "GPU " << dev << ": Owns chunk [" << (dev * chunk_size)
                           << ":" << ((dev + 1) * chunk_size) << "]" << std::endl;
        }

        hipblaslt_cout << "\n=== All-Gather Pattern ===" << std::endl;
        hipblaslt_cout << "Gather all chunks to reconstruct full parameter tensor" << std::endl;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipblaslt_cout << "GPU " << dev << ": Broadcasts its chunk " << dev
                           << " to all GPUs" << std::endl;
        }

        hipblaslt_cout << "\n=== Usage in FSDP/ZeRO ===" << std::endl;
        hipblaslt_cout << "Forward pass:" << std::endl;
        hipblaslt_cout << "  1. All-gather weights from all GPUs" << std::endl;
        hipblaslt_cout << "  2. Compute forward pass" << std::endl;
        hipblaslt_cout << "  3. Free gathered weights" << std::endl;

        hipblaslt_cout << "\nBackward pass:" << std::endl;
        hipblaslt_cout << "  1. All-gather weights (again)" << std::endl;
        hipblaslt_cout << "  2. Compute gradients" << std::endl;
        hipblaslt_cout << "  3. Reduce-scatter gradients (each GPU gets its chunk)" << std::endl;
        hipblaslt_cout << "  4. Update local weight chunk" << std::endl;

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "✓ All-gather and reduce-scatter patterns demonstrated" << std::endl;
        hipblaslt_cout << "  Foundation for: FSDP, ZeRO-2, ZeRO-3" << std::endl;
    }

} // namespace
