/*******************************************************************************
 * Multi-GPU Sequence Parallelism Test Suite
 * Tests: Sequence dimension partitioning, context parallelism, variable lengths,
 *        all-gather patterns
 *
 * Critical for: Long context models (>32K tokens), memory-efficient training
 * Used in: Megatron-LM, DeepSpeed, FSDP
 *******************************************************************************/

#include "hipblaslt_data.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_test.hpp"
#include "utility.hpp"
#include <vector>
#include <numeric>
#include <algorithm>

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
    // Test 1: Sequence Dimension Partitioning
    // Split sequence length across GPUs (orthogonal to tensor parallelism)
    // ============================================================================
    TEST(MultiGPUSequenceParallel, SequenceDimensionPartition)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Sequence Parallelism - Sequence Dimension Partition ===" << std::endl;

        const int64_t batch_size = 4;
        const int64_t total_seq_len = 16384;  // 16K tokens
        const int64_t hidden_dim = 2048;
        const int64_t ffn_dim = 8192;

        // Split sequence across GPUs
        int64_t seq_per_gpu = total_seq_len / numDevices;

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_input_seq(numDevices), d_ffn_weight(numDevices);
        std::vector<float*> d_output_local(numDevices), d_output_gathered(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            int64_t local_seq_start = dev * seq_per_gpu;
            int64_t local_seq_len = (dev == numDevices - 1) ? (total_seq_len - local_seq_start) : seq_per_gpu;

            // Each GPU has:
            // - Its local sequence chunk: [batch, local_seq_len, hidden_dim]
            // - Shared FFN weights: [hidden_dim, ffn_dim]
            // - Local output: [batch, local_seq_len, ffn_dim]
            hipMalloc(&d_input_seq[dev], batch_size * local_seq_len * hidden_dim * sizeof(float));
            hipMalloc(&d_ffn_weight[dev], hidden_dim * ffn_dim * sizeof(float));
            hipMalloc(&d_output_local[dev], batch_size * local_seq_len * ffn_dim * sizeof(float));

            hipMemset(d_input_seq[dev], 0, batch_size * local_seq_len * hidden_dim * sizeof(float));
            hipMemset(d_ffn_weight[dev], 0, hidden_dim * ffn_dim * sizeof(float));

            hipblaslt_cout << "GPU " << dev << ": Sequence chunk [" << local_seq_start
                           << ":" << (local_seq_start + local_seq_len) << "]" << std::endl;

            // Compute local FFN: input @ weight
            // [batch * local_seq_len, hidden_dim] @ [hidden_dim, ffn_dim]
            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, batch_size * local_seq_len, hidden_dim,
                                       batch_size * local_seq_len);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, hidden_dim, ffn_dim, hidden_dim);
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, batch_size * local_seq_len, ffn_dim,
                                       batch_size * local_seq_len);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, batch_size * local_seq_len, ffn_dim,
                                       batch_size * local_seq_len);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            float alpha = 1.0f, beta = 0.0f;
            hipblasLtMatmul(handles[dev], matmul, &alpha,
                           d_input_seq[dev], matA, d_ffn_weight[dev], matB,
                           &beta, d_output_local[dev], matC, d_output_local[dev], matD,
                           nullptr, nullptr, 0, 0);

            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);

            hipblaslt_cout << "  Computed local FFN for " << local_seq_len << " tokens" << std::endl;
        }

        // In real implementation, would need all-gather to collect full sequence
        hipblaslt_cout << "\n=== Sequence Parallelism Benefits ===" << std::endl;
        hipblaslt_cout << "Memory per GPU: " << (seq_per_gpu * hidden_dim * sizeof(float) / 1024.0 / 1024.0)
                       << " MB (vs " << (total_seq_len * hidden_dim * sizeof(float) / 1024.0 / 1024.0)
                       << " MB for full sequence)" << std::endl;
        hipblaslt_cout << "Reduction: " << (float)numDevices << "x" << std::endl;

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_input_seq[dev]);
            hipFree(d_ffn_weight[dev]);
            hipFree(d_output_local[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "✓ Sequence parallelism computation completed" << std::endl;
    }

    // ============================================================================
    // Test 2: Context Parallelism (Megatron-style)
    // Combine with tensor parallelism for maximum efficiency
    // ============================================================================
    TEST(MultiGPUSequenceParallel, MegatronContextParallelism)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 4) GTEST_SKIP() << "Requires 4+ GPUs for 2D parallelism";

        hipblaslt_cout << "=== Megatron-Style Context Parallelism ===" << std::endl;

        const int64_t batch_size = 2;
        const int64_t total_seq_len = 32768;  // 32K context
        const int64_t hidden_dim = 4096;

        // 2D parallelism: tensor parallel (TP) + sequence parallel (SP)
        // Example with 8 GPUs: 2x TP, 4x SP
        int64_t tp_size = 2;
        int64_t sp_size = numDevices / tp_size;

        int64_t seq_per_gpu = total_seq_len / sp_size;

        hipblaslt_cout << "Configuration:" << std::endl;
        hipblaslt_cout << "  Tensor Parallel size: " << tp_size << std::endl;
        hipblaslt_cout << "  Sequence Parallel size: " << sp_size << std::endl;
        hipblaslt_cout << "  Total GPUs: " << numDevices << " (TP=" << tp_size << " × SP=" << sp_size << ")" << std::endl;

        std::vector<hipblasLtHandle_t> handles(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            int64_t tp_rank = dev % tp_size;
            int64_t sp_rank = dev / tp_size;

            int64_t seq_start = sp_rank * seq_per_gpu;
            int64_t local_seq_len = (sp_rank == sp_size - 1) ? (total_seq_len - seq_start) : seq_per_gpu;

            hipblaslt_cout << "GPU " << dev << " (TP rank " << tp_rank << ", SP rank " << sp_rank
                           << "): Sequence [" << seq_start << ":" << (seq_start + local_seq_len) << "]" << std::endl;

            // This GPU handles:
            // - Sequence chunk: sp_rank determines which tokens
            // - Model chunk: tp_rank determines which weights
        }

        hipblaslt_cout << "\n=== 2D Parallelism Benefits ===" << std::endl;
        hipblaslt_cout << "Activation memory per GPU: " << (seq_per_gpu * hidden_dim * sizeof(float) / 1024.0 / 1024.0)
                       << " MB (" << sp_size << "x reduction from sequence parallel)" << std::endl;
        hipblaslt_cout << "Weight memory per GPU: Reduced by " << tp_size << "x from tensor parallel" << std::endl;
        hipblaslt_cout << "Total memory reduction: " << (sp_size * tp_size) << "x" << std::endl;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "✓ Context parallelism pattern demonstrated" << std::endl;
    }

    // ============================================================================
    // Test 3: Variable Sequence Lengths (Ragged Batching)
    // Different sequences have different lengths - common in real inference
    // ============================================================================
    TEST(MultiGPUSequenceParallel, VariableSequenceLengths)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Variable Sequence Lengths (Ragged Batching) ===" << std::endl;

        const int64_t num_sequences = 16;
        const int64_t hidden_dim = 1024;

        // Variable length sequences (real-world distribution)
        std::vector<int64_t> sequence_lengths = {
            128, 256, 512, 64,   // Batch 0-3
            1024, 2048, 96, 300, // Batch 4-7
            150, 400, 768, 200,  // Batch 8-11
            32, 1536, 512, 256   // Batch 12-15
        };

        // Calculate total tokens
        int64_t total_tokens = std::accumulate(sequence_lengths.begin(), sequence_lengths.end(), 0LL);

        hipblaslt_cout << "Sequence lengths: [";
        for(size_t i = 0; i < sequence_lengths.size(); ++i)
        {
            hipblaslt_cout << sequence_lengths[i];
            if(i < sequence_lengths.size() - 1) hipblaslt_cout << ", ";
        }
        hipblaslt_cout << "]" << std::endl;
        hipblaslt_cout << "Total tokens: " << total_tokens << std::endl;

        // Strategy 1: Pack by total tokens (load balancing)
        std::vector<std::vector<int64_t>> gpu_assignments(numDevices);
        std::vector<int64_t> gpu_token_counts(numDevices, 0);

        // Greedy assignment: assign each sequence to GPU with fewest tokens
        for(size_t i = 0; i < sequence_lengths.size(); ++i)
        {
            int min_gpu = std::min_element(gpu_token_counts.begin(), gpu_token_counts.end())
                          - gpu_token_counts.begin();
            gpu_assignments[min_gpu].push_back(i);
            gpu_token_counts[min_gpu] += sequence_lengths[i];
        }

        hipblaslt_cout << "\n=== Load-Balanced Assignment ===" << std::endl;
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipblaslt_cout << "GPU " << dev << ": " << gpu_assignments[dev].size()
                           << " sequences, " << gpu_token_counts[dev] << " tokens [";
            for(size_t i = 0; i < gpu_assignments[dev].size(); ++i)
            {
                hipblaslt_cout << "seq" << gpu_assignments[dev][i];
                if(i < gpu_assignments[dev].size() - 1) hipblaslt_cout << ", ";
            }
            hipblaslt_cout << "]" << std::endl;
        }

        // Calculate load balance quality
        int64_t max_tokens = *std::max_element(gpu_token_counts.begin(), gpu_token_counts.end());
        int64_t min_tokens = *std::min_element(gpu_token_counts.begin(), gpu_token_counts.end());
        float balance_ratio = (float)min_tokens / max_tokens;

        hipblaslt_cout << "\nLoad balance: " << (balance_ratio * 100) << "%" << std::endl;
        hipblaslt_cout << "  Max GPU tokens: " << max_tokens << std::endl;
        hipblaslt_cout << "  Min GPU tokens: " << min_tokens << std::endl;

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_packed_input(numDevices), d_packed_output(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            // Allocate for this GPU's total tokens
            int64_t local_tokens = gpu_token_counts[dev];
            hipMalloc(&d_packed_input[dev], local_tokens * hidden_dim * sizeof(float));
            hipMalloc(&d_packed_output[dev], local_tokens * hidden_dim * sizeof(float));

            hipMemset(d_packed_input[dev], 0, local_tokens * hidden_dim * sizeof(float));

            hipblaslt_cout << "GPU " << dev << ": Allocated " << local_tokens << " packed tokens" << std::endl;
        }

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_packed_input[dev]);
            hipFree(d_packed_output[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "✓ Variable sequence length handling demonstrated" << std::endl;
        hipblaslt_cout << "  Key insight: Pack by tokens, not sequences, for balanced load" << std::endl;
    }

    // ============================================================================
    // Test 4: All-Gather Pattern for Sequence Parallel
    // Reconstruct full sequence from distributed chunks
    // ============================================================================
    TEST(MultiGPUSequenceParallel, AllGatherPattern)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== All-Gather for Sequence Parallel ===" << std::endl;

        const int64_t batch_size = 4;
        const int64_t total_seq_len = 8192;
        const int64_t hidden_dim = 2048;

        int64_t seq_per_gpu = total_seq_len / numDevices;

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_local_chunk(numDevices);
        std::vector<float*> d_gathered_full(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            int64_t local_seq_start = dev * seq_per_gpu;
            int64_t local_seq_len = (dev == numDevices - 1) ? (total_seq_len - local_seq_start) : seq_per_gpu;

            // Local chunk: [batch, local_seq_len, hidden_dim]
            hipMalloc(&d_local_chunk[dev], batch_size * local_seq_len * hidden_dim * sizeof(float));

            // Buffer for gathered full sequence: [batch, total_seq_len, hidden_dim]
            hipMalloc(&d_gathered_full[dev], batch_size * total_seq_len * hidden_dim * sizeof(float));

            hipMemset(d_local_chunk[dev], 0, batch_size * local_seq_len * hidden_dim * sizeof(float));

            hipblaslt_cout << "GPU " << dev << ": Local chunk size " << local_seq_len
                           << " tokens, will gather to " << total_seq_len << " tokens" << std::endl;
        }

        // Simulate all-gather (in real code, would use RCCL ncclAllGather)
        hipblaslt_cout << "\n=== All-Gather Communication Pattern ===" << std::endl;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);

            // Each GPU needs to:
            // 1. Send its local chunk to all other GPUs
            // 2. Receive chunks from all other GPUs
            // 3. Assemble into full sequence

            for(int source_gpu = 0; source_gpu < numDevices; ++source_gpu)
            {
                int64_t source_seq_start = source_gpu * seq_per_gpu;
                int64_t source_seq_len = (source_gpu == numDevices - 1) ?
                                         (total_seq_len - source_seq_start) : seq_per_gpu;

                if(source_gpu == dev)
                {
                    // Copy own chunk to correct position in gathered buffer
                    hipblaslt_cout << "GPU " << dev << ": Copying own chunk [" << source_seq_start
                                   << ":" << (source_seq_start + source_seq_len) << "]" << std::endl;
                }
                else
                {
                    // Would receive from other GPU via P2P or collective
                    hipblaslt_cout << "GPU " << dev << ": Would receive chunk ["
                                   << source_seq_start << ":" << (source_seq_start + source_seq_len)
                                   << "] from GPU " << source_gpu << std::endl;
                }
            }
        }

        // Calculate communication volume
        size_t chunk_bytes = batch_size * seq_per_gpu * hidden_dim * sizeof(float);
        size_t total_comm_per_gpu = chunk_bytes * (numDevices - 1);  // Receive from all others

        hipblaslt_cout << "\n=== Communication Analysis ===" << std::endl;
        hipblaslt_cout << "Chunk size per GPU: " << (chunk_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
        hipblaslt_cout << "Total communication per GPU: " << (total_comm_per_gpu / 1024.0 / 1024.0) << " MB" << std::endl;
        hipblaslt_cout << "Aggregate bandwidth needed: "
                       << (total_comm_per_gpu * numDevices / 1024.0 / 1024.0 / 1024.0) << " GB" << std::endl;

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_local_chunk[dev]);
            hipFree(d_gathered_full[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "✓ All-gather pattern for sequence parallel demonstrated" << std::endl;
        hipblaslt_cout << "  In production: Use RCCL ncclAllGather for efficient implementation" << std::endl;
    }

    // ============================================================================
    // Test 5: Ulysses-Style Sequence Parallelism
    // Optimized sequence parallelism for attention layers
    // ============================================================================
    TEST(MultiGPUSequenceParallel, UlyssesSequenceParallel)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Ulysses-Style Sequence Parallelism for Attention ===" << std::endl;

        const int64_t batch_size = 2;
        const int64_t total_seq_len = 16384;
        const int64_t num_heads = 32;
        const int64_t head_dim = 128;

        // In Ulysses, each GPU computes attention for a subset of heads
        // But needs full sequence for its heads
        int64_t heads_per_gpu = num_heads / numDevices;

        std::vector<hipblasLtHandle_t> handles(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            int64_t head_start = dev * heads_per_gpu;
            int64_t local_heads = (dev == numDevices - 1) ? (num_heads - head_start) : heads_per_gpu;

            hipblaslt_cout << "GPU " << dev << ": Handles heads [" << head_start
                           << ":" << (head_start + local_heads) << "]" << std::endl;
            hipblaslt_cout << "  Full sequence length: " << total_seq_len << " tokens" << std::endl;
            hipblaslt_cout << "  Memory: " << (batch_size * total_seq_len * local_heads * head_dim * sizeof(float) / 1024.0 / 1024.0)
                           << " MB per matrix (Q/K/V)" << std::endl;
        }

        hipblaslt_cout << "\n=== Ulysses Communication Pattern ===" << std::endl;
        hipblaslt_cout << "Step 1: All-to-All scatter Q, K, V by heads" << std::endl;
        hipblaslt_cout << "  Each GPU sends " << heads_per_gpu << " heads to all GPUs" << std::endl;
        hipblaslt_cout << "  Each GPU receives " << heads_per_gpu << " heads from all GPUs" << std::endl;
        hipblaslt_cout << "Step 2: Local attention computation (full sequence, subset of heads)" << std::endl;
        hipblaslt_cout << "Step 3: All-to-All gather attention outputs" << std::endl;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "✓ Ulysses sequence parallelism pattern demonstrated" << std::endl;
        hipblaslt_cout << "  Benefits: Balanced computation, minimal communication for long sequences" << std::endl;
    }

} // namespace
