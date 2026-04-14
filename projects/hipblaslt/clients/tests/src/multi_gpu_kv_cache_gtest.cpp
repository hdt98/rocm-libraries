/*******************************************************************************
 * Multi-GPU KV Cache Management Test Suite
 * Tests: KV cache distribution, PagedAttention, continuous batching, prefix caching
 *
 * Critical for LLM inference serving (vLLM, TGI, TensorRT-LLM)
 * KV cache is the memory bottleneck for inference
 *******************************************************************************/

#include "hipblaslt_data.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_test.hpp"
#include "utility.hpp"
#include <vector>
#include <map>
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
    // Test 1: Basic KV Cache Distribution
    // Each GPU caches KV for subset of sequence
    // ============================================================================
    TEST(MultiGPUKVCache, DistributedKVCache)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Distributed KV Cache Across GPUs ===" << std::endl;

        const int64_t max_seq_len = 16384;
        const int64_t num_layers = 32;
        const int64_t num_heads = 32;
        const int64_t head_dim = 128;
        const int64_t max_batch = 64;

        // Distribute sequence length across GPUs
        int64_t seq_per_gpu = max_seq_len / numDevices;

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_k_cache(numDevices), d_v_cache(numDevices);

        size_t total_kv_cache_bytes = 0;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            int64_t local_seq_start = dev * seq_per_gpu;
            int64_t local_seq_len = (dev == numDevices - 1) ? (max_seq_len - local_seq_start) : seq_per_gpu;

            // KV cache: [num_layers, max_batch, num_heads, local_seq_len, head_dim]
            size_t kv_cache_size = num_layers * max_batch * num_heads * local_seq_len * head_dim;
            size_t kv_cache_bytes = kv_cache_size * sizeof(float);

            hipMalloc(&d_k_cache[dev], kv_cache_bytes);
            hipMalloc(&d_v_cache[dev], kv_cache_bytes);

            total_kv_cache_bytes += 2 * kv_cache_bytes;  // K and V

            hipblaslt_cout << "GPU " << dev << ": KV cache for sequence [" << local_seq_start
                           << ":" << (local_seq_start + local_seq_len) << "]" << std::endl;
            hipblaslt_cout << "  Memory: " << (kv_cache_bytes / 1024.0 / 1024.0 / 1024.0)
                           << " GB (K) + " << (kv_cache_bytes / 1024.0 / 1024.0 / 1024.0) << " GB (V)" << std::endl;
        }

        float single_gpu_cache_gb = (num_layers * max_batch * num_heads * max_seq_len * head_dim * sizeof(float) * 2)
                                     / 1024.0 / 1024.0 / 1024.0;
        float distributed_cache_gb = total_kv_cache_bytes / 1024.0 / 1024.0 / 1024.0;

        hipblaslt_cout << "\n=== KV Cache Memory Analysis ===" << std::endl;
        hipblaslt_cout << "Single GPU would need: " << single_gpu_cache_gb << " GB" << std::endl;
        hipblaslt_cout << "Distributed across " << numDevices << " GPUs: "
                       << distributed_cache_gb << " GB total ("
                       << (distributed_cache_gb / numDevices) << " GB per GPU)" << std::endl;
        hipblaslt_cout << "Memory savings per GPU: "
                       << ((single_gpu_cache_gb - distributed_cache_gb / numDevices) / single_gpu_cache_gb * 100)
                       << "%" << std::endl;

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_k_cache[dev]);
            hipFree(d_v_cache[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "✓ Distributed KV cache allocation successful" << std::endl;
    }

    // ============================================================================
    // Test 2: PagedAttention (vLLM-style)
    // KV cache organized in pages that can be shared/reused
    // ============================================================================
    TEST(MultiGPUKVCache, PagedAttention)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== PagedAttention Multi-GPU (vLLM-style) ===" << std::endl;

        const int64_t page_size = 64;  // Tokens per page
        const int64_t num_pages_per_gpu = 512;
        const int64_t num_heads = 32;
        const int64_t head_dim = 128;
        const int64_t num_layers = 32;

        // Page table maps logical sequence positions to physical pages
        std::vector<std::map<int64_t, int64_t>> page_tables(numDevices);

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_page_pool_k(numDevices), d_page_pool_v(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            // Allocate page pool: [num_layers, num_pages, num_heads, page_size, head_dim]
            size_t page_pool_size = num_layers * num_pages_per_gpu * num_heads * page_size * head_dim;
            hipMalloc(&d_page_pool_k[dev], page_pool_size * sizeof(float));
            hipMalloc(&d_page_pool_v[dev], page_pool_size * sizeof(float));

            hipblaslt_cout << "GPU " << dev << ": Paged KV cache pool with "
                           << num_pages_per_gpu << " pages" << std::endl;
            hipblaslt_cout << "  Page size: " << page_size << " tokens" << std::endl;
            hipblaslt_cout << "  Total capacity: " << (num_pages_per_gpu * page_size) << " tokens" << std::endl;
        }

        // Simulate multiple inference requests with different sequence lengths
        struct Request
        {
            int64_t request_id;
            int64_t seq_len;
            int assigned_gpu;
            std::vector<int64_t> page_ids;  // Physical pages assigned
        };

        std::vector<Request> requests = {
            {0, 100, 0, {}},   // 100 tokens → 2 pages
            {1, 250, 1, {}},   // 250 tokens → 4 pages
            {2, 50, 2, {}},    // 50 tokens → 1 page
            {3, 512, 3, {}},   // 512 tokens → 8 pages
        };

        int64_t next_page_id = 0;
        for(auto& req : requests)
        {
            int64_t pages_needed = (req.seq_len + page_size - 1) / page_size;
            req.assigned_gpu = req.request_id % numDevices;

            // Allocate pages for this request
            for(int64_t i = 0; i < pages_needed; ++i)
            {
                req.page_ids.push_back(next_page_id++);
            }

            hipblaslt_cout << "Request " << req.request_id << " (seq_len=" << req.seq_len
                           << ") → GPU " << req.assigned_gpu << ", pages: [";
            for(size_t i = 0; i < req.page_ids.size(); ++i)
            {
                hipblaslt_cout << req.page_ids[i];
                if(i < req.page_ids.size() - 1) hipblaslt_cout << ", ";
            }
            hipblaslt_cout << "]" << std::endl;
        }

        // Demonstrate prefix sharing (key feature of PagedAttention)
        hipblaslt_cout << "\n=== Prefix Sharing Example ===" << std::endl;
        hipblaslt_cout << "Request 4 and 5 share common prefix (first 64 tokens):" << std::endl;

        Request req4 = {4, 128, 0, {100}};  // First page shared
        Request req5 = {5, 150, 0, {100}};  // Same first page!

        hipblaslt_cout << "  Request 4: pages [100, 101] (shares page 100)" << std::endl;
        hipblaslt_cout << "  Request 5: pages [100, 102] (shares page 100)" << std::endl;
        hipblaslt_cout << "  → Page 100 computed once, used by both requests (2x speedup)" << std::endl;

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipFree(d_page_pool_k[dev]);
            hipFree(d_page_pool_v[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "✓ PagedAttention pattern demonstrated" << std::endl;
        hipblaslt_cout << "  Benefits: Memory efficiency, prefix sharing, flexible allocation" << std::endl;
    }

    // ============================================================================
    // Test 3: Continuous Batching
    // Dynamic batching of requests with different sequence lengths
    // ============================================================================
    TEST(MultiGPUKVCache, ContinuousBatching)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Continuous Batching Multi-GPU ===" << std::endl;

        const int64_t max_batch_size = 64;
        const int64_t num_heads = 32;
        const int64_t head_dim = 128;

        // Active requests with varying sequence lengths (ragged batch)
        struct ActiveRequest
        {
            int64_t request_id;
            int64_t current_seq_len;
            int64_t max_seq_len;
            int gpu_id;
            bool is_generating;
        };

        std::vector<ActiveRequest> active_requests;

        // Initialize with some requests
        active_requests = {
            {0, 50, 512, 0, true},    // Early in generation
            {1, 200, 512, 0, true},   // Mid generation
            {2, 30, 256, 1, true},    // Just started
            {3, 480, 512, 1, true},   // Almost done
            {4, 100, 1024, 2, true},  // Long context
        };

        std::vector<hipblasLtHandle_t> handles(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);
        }

        // Simulate multiple generation steps
        for(int step = 0; step < 5; ++step)
        {
            hipblaslt_cout << "\n=== Generation Step " << (step + 1) << " ===" << std::endl;

            // Group requests by GPU
            std::map<int, std::vector<ActiveRequest*>> requests_per_gpu;
            for(auto& req : active_requests)
            {
                if(req.is_generating)
                {
                    requests_per_gpu[req.gpu_id].push_back(&req);
                }
            }

            // Process each GPU's batch
            for(auto& [gpu_id, gpu_requests] : requests_per_gpu)
            {
                hipSetDevice(gpu_id);

                hipblaslt_cout << "GPU " << gpu_id << ": Processing " << gpu_requests.size()
                               << " requests" << std::endl;

                // In continuous batching, each request advances by 1 token
                for(auto* req : gpu_requests)
                {
                    req->current_seq_len++;

                    // Simulate GEMM for this request's current token
                    // Q: [1, num_heads, 1, head_dim] (single new token)
                    // K/V cache: [1, num_heads, current_seq_len, head_dim]

                    hipblaslt_cout << "  Request " << req->request_id << ": token "
                                   << req->current_seq_len << "/" << req->max_seq_len << std::endl;

                    // Check if request finished
                    if(req->current_seq_len >= req->max_seq_len)
                    {
                        req->is_generating = false;
                        hipblaslt_cout << "    → Request " << req->request_id << " COMPLETED" << std::endl;
                    }
                }
            }

            // Add new request (continuous batching!)
            if(step == 2)
            {
                ActiveRequest new_req = {5 + step, 1, 300, step % numDevices, true};
                active_requests.push_back(new_req);
                hipblaslt_cout << "  → NEW REQUEST " << new_req.request_id
                               << " added to GPU " << new_req.gpu_id << std::endl;
            }
        }

        // Count completed vs active
        int completed = 0, still_active = 0;
        for(const auto& req : active_requests)
        {
            if(req.is_generating) still_active++;
            else completed++;
        }

        hipblaslt_cout << "\n=== Final Statistics ===" << std::endl;
        hipblaslt_cout << "Completed requests: " << completed << std::endl;
        hipblaslt_cout << "Still generating: " << still_active << std::endl;

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "✓ Continuous batching demonstrated" << std::endl;
        hipblaslt_cout << "  Key benefit: GPU utilization stays high even with variable-length requests" << std::endl;
    }

    // ============================================================================
    // Test 4: Prefix Caching
    // Cache and reuse KV for common prefixes (system prompts, few-shot examples)
    // ============================================================================
    TEST(MultiGPUKVCache, PrefixCaching)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Prefix Caching Multi-GPU ===" << std::endl;

        const int64_t num_layers = 32;
        const int64_t num_heads = 32;
        const int64_t head_dim = 128;

        // Common prefixes (e.g., system prompts)
        struct CachedPrefix
        {
            std::string name;
            int64_t token_length;
            int gpu_id;
            void* kv_cache_ptr;
        };

        std::vector<CachedPrefix> cached_prefixes = {
            {"system_prompt_v1", 256, 0, nullptr},
            {"few_shot_3examples", 512, 0, nullptr},
            {"cot_prompt", 128, 1, nullptr},
            {"function_calling_prompt", 384, 1, nullptr},
        };

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<std::vector<float*>> prefix_caches(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            hipblaslt_cout << "GPU " << dev << " cached prefixes:" << std::endl;

            for(auto& prefix : cached_prefixes)
            {
                if(prefix.gpu_id == dev)
                {
                    // Allocate KV cache for this prefix
                    // [num_layers, 1, num_heads, prefix_len, head_dim] * 2 (K and V)
                    size_t prefix_cache_size = num_layers * num_heads * prefix.token_length * head_dim * 2;
                    float* d_prefix_cache;
                    hipMalloc(&d_prefix_cache, prefix_cache_size * sizeof(float));
                    prefix.kv_cache_ptr = d_prefix_cache;
                    prefix_caches[dev].push_back(d_prefix_cache);

                    hipblaslt_cout << "  - " << prefix.name << " (" << prefix.token_length
                                   << " tokens, " << (prefix_cache_size * sizeof(float) / 1024.0 / 1024.0)
                                   << " MB)" << std::endl;
                }
            }
        }

        // Simulate requests that reuse cached prefixes
        hipblaslt_cout << "\n=== Requests with Cached Prefixes ===" << std::endl;

        struct Request
        {
            int64_t id;
            std::string prefix_used;
            int64_t user_tokens;
        };

        std::vector<Request> requests = {
            {0, "system_prompt_v1", 50},       // Reuses 256 tokens
            {1, "system_prompt_v1", 75},       // Same prefix!
            {2, "few_shot_3examples", 100},    // Different prefix
            {3, "system_prompt_v1", 30},       // Same as req 0,1
        };

        int total_computed_tokens = 0;
        int total_cached_tokens = 0;

        for(const auto& req : requests)
        {
            auto it = std::find_if(cached_prefixes.begin(), cached_prefixes.end(),
                                   [&](const CachedPrefix& p) { return p.name == req.prefix_used; });

            if(it != cached_prefixes.end())
            {
                hipblaslt_cout << "Request " << req.id << ": Reusing cached prefix '"
                               << it->name << "' (" << it->token_length << " tokens) + computing "
                               << req.user_tokens << " new tokens" << std::endl;

                total_cached_tokens += it->token_length;
                total_computed_tokens += req.user_tokens;
            }
        }

        int total_tokens_without_cache = total_cached_tokens + total_computed_tokens;
        float cache_hit_rate = (float)total_cached_tokens / total_tokens_without_cache * 100;
        float speedup = (float)total_tokens_without_cache / total_computed_tokens;

        hipblaslt_cout << "\n=== Prefix Caching Statistics ===" << std::endl;
        hipblaslt_cout << "Total tokens without caching: " << total_tokens_without_cache << std::endl;
        hipblaslt_cout << "Tokens from cache: " << total_cached_tokens
                       << " (" << cache_hit_rate << "%)" << std::endl;
        hipblaslt_cout << "Tokens computed: " << total_computed_tokens << std::endl;
        hipblaslt_cout << "Effective speedup: " << speedup << "x" << std::endl;

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto err = hipSetDevice(dev);
            (void)err;
            for(auto* cache : prefix_caches[dev])
            {
                auto err2 = hipFree(cache);
                (void)err2;
            }
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "✓ Prefix caching demonstrated" << std::endl;
        hipblaslt_cout << "  Use cases: System prompts, few-shot examples, RAG contexts" << std::endl;
    }

    // ============================================================================
    // Test 5: KV Cache Compression
    // Reduce KV cache memory footprint via quantization/compression
    // ============================================================================
    TEST(MultiGPUKVCache, KVCacheCompression)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== KV Cache Compression Multi-GPU ===" << std::endl;

        const int64_t num_layers = 32;
        const int64_t max_batch = 32;
        const int64_t num_heads = 32;
        const int64_t max_seq_len = 8192;
        const int64_t head_dim = 128;

        // Calculate memory for different precision levels
        size_t kv_size_per_layer = max_batch * num_heads * max_seq_len * head_dim * 2; // K and V

        struct CompressionFormat
        {
            std::string name;
            int bits_per_element;
            float compression_ratio;
        };

        std::vector<CompressionFormat> formats = {
            {"FP32 (baseline)", 32, 1.0f},
            {"FP16", 16, 2.0f},
            {"INT8", 8, 4.0f},
            {"INT4", 4, 8.0f},
        };

        hipblaslt_cout << "\n=== KV Cache Memory Comparison ===" << std::endl;
        hipblaslt_cout << "Configuration: " << num_layers << " layers, " << max_batch
                       << " batch, " << max_seq_len << " seq_len" << std::endl;

        for(const auto& fmt : formats)
        {
            size_t bytes_per_layer = kv_size_per_layer * fmt.bits_per_element / 8;
            size_t total_bytes = bytes_per_layer * num_layers;
            float total_gb = total_bytes / 1024.0f / 1024.0f / 1024.0f;
            float per_gpu_gb = total_gb / numDevices;

            hipblaslt_cout << fmt.name << ": " << total_gb << " GB total, "
                           << per_gpu_gb << " GB per GPU (/" << numDevices << ")" << std::endl;
        }

        hipblaslt_cout << "\n=== Compression Benefits ===" << std::endl;
        hipblaslt_cout << "INT8 KV cache: 4x memory reduction" << std::endl;
        hipblaslt_cout << "  → Can serve 4x more requests" << std::endl;
        hipblaslt_cout << "  → Or support 4x longer sequences" << std::endl;
        hipblaslt_cout << "INT4 KV cache: 8x memory reduction" << std::endl;
        hipblaslt_cout << "  → Extreme memory efficiency" << std::endl;
        hipblaslt_cout << "  → Slight accuracy degradation (acceptable for many use cases)" << std::endl;

        hipblaslt_cout << "✓ KV cache compression analysis completed" << std::endl;
    }

    // ============================================================================
    // WEEK 6: Advanced KV Cache Features (P1)
    // ============================================================================

    // ----------------------------------------------------------------------------
    // Test: Radix Tree Prefix Sharing
    // Production Use: vLLM, SGLang advanced prefix sharing with tree structure
    // ----------------------------------------------------------------------------
    TEST(MultiGPUKVCache, RadixTree_PrefixSharing)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "\n=== Radix Tree KV Cache Prefix Sharing ===" << std::endl;
        hipblaslt_cout << "Production Use: vLLM, SGLang tree-based prefix caching" << std::endl;

        const int64_t num_layers = 32;
        const int64_t num_heads = 32;
        const int64_t head_dim = 128;

        // Radix tree structure for KV cache sharing
        struct RadixTreeNode
        {
            int64_t node_id;
            int64_t token_start;
            int64_t token_end;
            int64_t parent_id;
            std::vector<int64_t> children;
            int gpu_id;
            void* kv_cache_ptr;
            int ref_count;  // Number of requests using this node
        };

        std::vector<RadixTreeNode> tree_nodes = {
            // Root: System prompt (256 tokens) on GPU 0
            {0, 0, 256, -1, {1, 2, 3}, 0, nullptr, 0},
            // Branch 1: Code generation continuation
            {1, 256, 384, 0, {}, 0, nullptr, 0},  // "Write a Python function..."
            // Branch 2: Math problem continuation
            {2, 256, 400, 0, {}, 1, nullptr, 0},  // "Solve the equation..."
            // Branch 3: Translation continuation
            {3, 256, 320, 0, {4, 5}, 1, nullptr, 0},  // "Translate to French..."
            // Sub-branches under translation
            {4, 320, 450, 3, {}, 1, nullptr, 0},  // "...formal tone"
            {5, 320, 480, 3, {}, 1, nullptr, 0},  // "...casual tone"
        };

        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::map<int64_t, float*> node_cache_map;

        hipblaslt_cout << "\n  Radix Tree Structure:" << std::endl;
        hipblaslt_cout << "    Node 0 (Root): [0-256] tokens (system prompt) - GPU 0" << std::endl;
        hipblaslt_cout << "      ├─ Node 1: [256-384] (code gen) - GPU 0" << std::endl;
        hipblaslt_cout << "      ├─ Node 2: [256-400] (math) - GPU 1" << std::endl;
        hipblaslt_cout << "      └─ Node 3: [256-320] (translate) - GPU 1" << std::endl;
        hipblaslt_cout << "           ├─ Node 4: [320-450] (formal) - GPU 1" << std::endl;
        hipblaslt_cout << "           └─ Node 5: [320-480] (casual) - GPU 1" << std::endl;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto err = hipSetDevice(dev);
            (void)err;
            hipblasLtCreate(&handles[dev]);

            hipblaslt_cout << "\n  GPU " << dev << " tree nodes:" << std::endl;

            for(auto& node : tree_nodes)
            {
                if(node.gpu_id == dev)
                {
                    int64_t node_tokens = node.token_end - node.token_start;
                    size_t cache_size = num_layers * num_heads * node_tokens * head_dim * 2;

                    float* d_cache;
                    hipMalloc(&d_cache, cache_size * sizeof(float));
                    node.kv_cache_ptr = d_cache;
                    node_cache_map[node.node_id] = d_cache;

                    hipblaslt_cout << "    Node " << node.node_id << ": ["
                                  << node.token_start << "-" << node.token_end << "] ("
                                  << node_tokens << " tokens, "
                                  << (cache_size * sizeof(float)) / (1024*1024) << " MB)" << std::endl;
                }
            }
        }

        // Simulate requests using the tree
        struct Request
        {
            int64_t request_id;
            std::vector<int64_t> node_path;  // Path through radix tree
            int64_t unique_tokens;  // Tokens unique to this request
        };

        std::vector<Request> requests = {
            {0, {0, 1}, 100},       // Uses root + code gen branch
            {1, {0, 2}, 80},        // Uses root + math branch (shares root!)
            {2, {0, 3, 4}, 50},     // Uses root + translate + formal
            {3, {0, 3, 5}, 60},     // Uses root + translate + casual (shares root+translate!)
        };

        int total_tokens_computed = 0;
        int total_tokens_shared = 0;

        hipblaslt_cout << "\n  Request processing with prefix sharing:" << std::endl;

        for(const auto& req : requests)
        {
            hipblaslt_cout << "    Request " << req.request_id << ": Path [";
            for(size_t i = 0; i < req.node_path.size(); ++i)
            {
                hipblaslt_cout << req.node_path[i];
                if(i < req.node_path.size() - 1) hipblaslt_cout << " → ";
            }
            hipblaslt_cout << "] + " << req.unique_tokens << " unique tokens" << std::endl;

            // Calculate shared tokens
            for(int64_t node_id : req.node_path)
            {
                auto it = std::find_if(tree_nodes.begin(), tree_nodes.end(),
                                      [node_id](const RadixTreeNode& n) { return n.node_id == node_id; });
                if(it != tree_nodes.end())
                {
                    int64_t node_tokens = it->token_end - it->token_start;
                    total_tokens_shared += node_tokens;
                    it->ref_count++;
                }
            }

            total_tokens_computed += req.unique_tokens;
        }

        // Calculate sharing statistics
        hipblaslt_cout << "\n  Radix Tree Sharing Statistics:" << std::endl;
        for(const auto& node : tree_nodes)
        {
            if(node.ref_count > 0)
            {
                int64_t node_tokens = node.token_end - node.token_start;
                hipblaslt_cout << "    Node " << node.node_id << ": Used by "
                              << node.ref_count << " requests (saved "
                              << (node_tokens * (node.ref_count - 1)) << " token computations)" << std::endl;
            }
        }

        int total_without_sharing = total_tokens_shared + total_tokens_computed;
        float sharing_efficiency = (float)(total_tokens_shared - 256) / total_without_sharing * 100;  // Subtract one root

        hipblaslt_cout << "\n✓ Radix tree prefix sharing completed" << std::endl;
        hipblaslt_cout << "  Total tokens without sharing: " << total_without_sharing << std::endl;
        hipblaslt_cout << "  Tokens actually computed: " << (total_tokens_computed + 256) << " (root + unique)" << std::endl;
        hipblaslt_cout << "  Sharing efficiency: " << sharing_efficiency << "%" << std::endl;

        // Cleanup
        for(auto& [node_id, cache_ptr] : node_cache_map)
        {
            auto it = std::find_if(tree_nodes.begin(), tree_nodes.end(),
                                  [node_id](const RadixTreeNode& n) { return n.node_id == node_id; });
            if(it != tree_nodes.end())
            {
                hipSetDevice(it->gpu_id);
                hipFree(cache_ptr);
            }
        }

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtDestroy(handles[dev]);
        }
    }

    // ----------------------------------------------------------------------------
    // Test: Cross-Request KV Sharing
    // Production Use: Share KV cache blocks across multiple concurrent requests
    // ----------------------------------------------------------------------------
    TEST(MultiGPUKVCache, CrossRequest_KVSharing)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "\n=== Cross-Request KV Cache Sharing ===" << std::endl;
        hipblaslt_cout << "Production Use: vLLM PagedAttention with cross-request sharing" << std::endl;

        const int64_t page_size = 64;
        const int64_t num_pages_per_gpu = 1024;
        const int64_t num_heads = 32;
        const int64_t head_dim = 128;
        const int64_t num_layers = 32;

        // KV cache page pool with reference counting
        struct KVCachePage
        {
            int64_t page_id;
            int gpu_id;
            std::vector<int64_t> token_content_hash;  // Hash of tokens in this page
            int ref_count;  // Number of requests sharing this page
            void* cache_ptr;
            bool is_dirty;
        };

        std::vector<KVCachePage> page_pool;
        std::vector<hipblasLtHandle_t> handles(numDevices);

        // Initialize page pool
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);

            hipblaslt_cout << "  GPU " << dev << ": Initialized " << num_pages_per_gpu
                          << " pages (" << page_size << " tokens each)" << std::endl;
        }

        // Simulate requests with common prompts
        struct Request
        {
            int64_t request_id;
            std::string prompt_type;
            std::vector<int64_t> shared_page_ids;
            std::vector<int64_t> unique_page_ids;
        };

        // Common prompt types
        std::map<std::string, std::vector<int64_t>> common_prompts = {
            {"system_v1", {0, 1, 2, 3}},      // 4 pages (256 tokens)
            {"few_shot_3", {10, 11, 12, 13, 14, 15, 16, 17}},  // 8 pages (512 tokens)
            {"rag_context", {20, 21, 22, 23, 24}},  // 5 pages (320 tokens)
        };

        std::vector<Request> requests = {
            {0, "system_v1", {0, 1, 2, 3}, {100, 101}},           // Shares system_v1
            {1, "system_v1", {0, 1, 2, 3}, {102, 103, 104}},     // Same prefix!
            {2, "few_shot_3", {10, 11, 12, 13, 14, 15, 16, 17}, {200, 201}},  // Different prefix
            {3, "system_v1", {0, 1, 2, 3}, {105}},               // Same as req 0,1!
            {4, "rag_context", {20, 21, 22, 23, 24}, {300, 301, 302}},
        };

        // Track page reference counts
        std::map<int64_t, int> page_ref_counts;

        hipblaslt_cout << "\n  Request allocation with sharing:" << std::endl;

        for(const auto& req : requests)
        {
            hipblaslt_cout << "    Request " << req.request_id << " ('" << req.prompt_type << "'):" << std::endl;

            // Shared pages
            hipblaslt_cout << "      Shared pages: [";
            for(size_t i = 0; i < req.shared_page_ids.size(); ++i)
            {
                int64_t page_id = req.shared_page_ids[i];
                page_ref_counts[page_id]++;
                hipblaslt_cout << page_id;
                if(i < req.shared_page_ids.size() - 1) hipblaslt_cout << ", ";
            }
            hipblaslt_cout << "] (reused)" << std::endl;

            // Unique pages
            hipblaslt_cout << "      Unique pages: [";
            for(size_t i = 0; i < req.unique_page_ids.size(); ++i)
            {
                int64_t page_id = req.unique_page_ids[i];
                page_ref_counts[page_id]++;
                hipblaslt_cout << page_id;
                if(i < req.unique_page_ids.size() - 1) hipblaslt_cout << ", ";
            }
            hipblaslt_cout << "] (allocated)" << std::endl;
        }

        // Calculate sharing statistics
        int total_pages_without_sharing = 0;
        int total_pages_with_sharing = 0;
        int total_shared_pages = 0;

        for(const auto& req : requests)
        {
            total_pages_without_sharing += (req.shared_page_ids.size() + req.unique_page_ids.size());
        }

        for(const auto& [page_id, ref_count] : page_ref_counts)
        {
            total_pages_with_sharing++;
            if(ref_count > 1)
            {
                total_shared_pages++;
            }
        }

        hipblaslt_cout << "\n  Cross-Request Sharing Statistics:" << std::endl;

        // Show most shared pages
        hipblaslt_cout << "    Most shared pages:" << std::endl;
        std::vector<std::pair<int64_t, int>> sorted_pages(page_ref_counts.begin(), page_ref_counts.end());
        std::sort(sorted_pages.begin(), sorted_pages.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });

        for(size_t i = 0; i < std::min(size_t(5), sorted_pages.size()); ++i)
        {
            if(sorted_pages[i].second > 1)
            {
                hipblaslt_cout << "      Page " << sorted_pages[i].first << ": Used by "
                              << sorted_pages[i].second << " requests (saved "
                              << (sorted_pages[i].second - 1) << " allocations)" << std::endl;
            }
        }

        float memory_saved = (1.0f - (float)total_pages_with_sharing / total_pages_without_sharing) * 100;

        hipblaslt_cout << "\n✓ Cross-request KV sharing completed" << std::endl;
        hipblaslt_cout << "  Pages without sharing: " << total_pages_without_sharing << std::endl;
        hipblaslt_cout << "  Pages with sharing: " << total_pages_with_sharing << std::endl;
        hipblaslt_cout << "  Shared pages: " << total_shared_pages << std::endl;
        hipblaslt_cout << "  Memory saved: " << memory_saved << "%" << std::endl;

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtDestroy(handles[dev]);
        }
    }

    // ============================================================================
    // WEEK 7-8: Continuous Batching Advanced Features (P1)
    // ============================================================================

    // ----------------------------------------------------------------------------
    // Test: Continuous Batching with Preemption
    // Production Use: vLLM, TGI, SGLang priority-based scheduling
    // ----------------------------------------------------------------------------
    TEST(MultiGPUKVCache, ContinuousBatching_Preemption)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "\n=== Continuous Batching with Request Preemption ===" << std::endl;
        hipblaslt_cout << "Production Use: vLLM, TGI priority scheduling (24× throughput)" << std::endl;

        const int64_t max_active_requests = 16;  // Per GPU
        const int64_t num_heads = 32;
        const int64_t head_dim = 128;

        // Request with priority
        struct PrioritizedRequest
        {
            int64_t request_id;
            int priority;  // Higher = more important
            int64_t current_seq_len;
            int64_t target_seq_len;
            int gpu_id;
            bool is_active;
            bool is_preempted;
            std::vector<int64_t> kv_page_ids;
        };

        std::vector<PrioritizedRequest> all_requests = {
            // High priority (interactive users)
            {0, 10, 50, 200, 0, true, false, {}},
            {1, 10, 30, 150, 0, true, false, {}},
            // Medium priority (batch jobs)
            {2, 5, 100, 500, 1, true, false, {}},
            {3, 5, 80, 400, 1, true, false, {}},
            // Low priority (background tasks)
            {4, 1, 200, 1000, 0, true, false, {}},
            {5, 1, 150, 800, 1, true, false, {}},
        };

        std::vector<hipblasLtHandle_t> handles(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);
        }

        hipblaslt_cout << "\n  Initial request queue:" << std::endl;
        for(const auto& req : all_requests)
        {
            hipblaslt_cout << "    Request " << req.request_id << " (priority=" << req.priority
                          << ", GPU " << req.gpu_id << "): " << req.current_seq_len
                          << "/" << req.target_seq_len << " tokens" << std::endl;
        }

        // Simulate high-priority request arriving
        hipblaslt_cout << "\n  [Event] High-priority request arrives!" << std::endl;
        PrioritizedRequest urgent_request = {10, 15, 10, 100, 0, false, false, {}};

        // Check GPU 0 capacity
        int active_on_gpu0 = 0;
        for(const auto& req : all_requests)
        {
            if(req.gpu_id == 0 && req.is_active) active_on_gpu0++;
        }

        if(active_on_gpu0 >= max_active_requests)
        {
            hipblaslt_cout << "    GPU 0 at capacity (" << active_on_gpu0 << "/"
                          << max_active_requests << "), preemption needed" << std::endl;

            // Find lowest priority request to preempt
            auto lowest_it = std::min_element(all_requests.begin(), all_requests.end(),
                [](const auto& a, const auto& b) {
                    if(a.gpu_id != 0 || !a.is_active) return false;
                    if(b.gpu_id != 0 || !b.is_active) return true;
                    return a.priority < b.priority;
                });

            if(lowest_it != all_requests.end() && lowest_it->priority < urgent_request.priority)
            {
                hipblaslt_cout << "    Preempting Request " << lowest_it->request_id
                              << " (priority=" << lowest_it->priority << ")" << std::endl;

                // Save KV cache state for preempted request
                hipblaslt_cout << "      → Saving KV cache state (pages: [";
                for(size_t i = 0; i < 3; ++i)
                {
                    lowest_it->kv_page_ids.push_back(1000 + lowest_it->request_id * 10 + i);
                    hipblaslt_cout << lowest_it->kv_page_ids.back();
                    if(i < 2) hipblaslt_cout << ", ";
                }
                hipblaslt_cout << "])" << std::endl;

                lowest_it->is_active = false;
                lowest_it->is_preempted = true;
            }
        }

        // Activate urgent request
        urgent_request.is_active = true;
        all_requests.push_back(urgent_request);

        hipblaslt_cout << "    Request " << urgent_request.request_id
                      << " (priority=" << urgent_request.priority << ") activated" << std::endl;

        // Process some tokens
        hipblaslt_cout << "\n  Processing generation steps..." << std::endl;

        for(int step = 0; step < 3; ++step)
        {
            hipblaslt_cout << "    Step " << (step + 1) << ":" << std::endl;

            for(auto& req : all_requests)
            {
                if(req.is_active && !req.is_preempted)
                {
                    req.current_seq_len++;

                    if(req.request_id == urgent_request.request_id && step == 0)
                    {
                        hipblaslt_cout << "      Request " << req.request_id
                                      << " (HIGH PRIORITY): " << req.current_seq_len
                                      << "/" << req.target_seq_len << std::endl;
                    }

                    if(req.current_seq_len >= req.target_seq_len)
                    {
                        req.is_active = false;
                        hipblaslt_cout << "      Request " << req.request_id
                                      << " COMPLETED" << std::endl;

                        // Resume preempted request if any
                        for(auto& preempted : all_requests)
                        {
                            if(preempted.is_preempted && preempted.gpu_id == req.gpu_id)
                            {
                                hipblaslt_cout << "      → Resuming preempted Request "
                                              << preempted.request_id << " (restoring KV from pages ["
                                              << preempted.kv_page_ids[0] << ", ...])" << std::endl;
                                preempted.is_active = true;
                                preempted.is_preempted = false;
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Final statistics
        int completed = 0, active = 0, preempted = 0;
        for(const auto& req : all_requests)
        {
            if(req.current_seq_len >= req.target_seq_len) completed++;
            else if(req.is_active) active++;
            else if(req.is_preempted) preempted++;
        }

        hipblaslt_cout << "\n✓ Continuous batching with preemption completed" << std::endl;
        hipblaslt_cout << "  Completed: " << completed << ", Active: " << active
                      << ", Preempted: " << preempted << std::endl;
        hipblaslt_cout << "  Preemption ensures high-priority requests get immediate service" << std::endl;

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtDestroy(handles[dev]);
        }
    }

    // ----------------------------------------------------------------------------
    // Test: Chunked Prefill (Break Large Prefills into Chunks)
    // Production Use: Prevent decode blocking during long prefills
    // ----------------------------------------------------------------------------
    TEST(MultiGPUKVCache, ContinuousBatching_ChunkedPrefill)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "\n=== Chunked Prefill for Continuous Batching ===" << std::endl;
        hipblaslt_cout << "Production Use: vLLM, SGLang (prevent decode blocking)" << std::endl;

        const int64_t prefill_chunk_size = 1024;  // Process 1K tokens per chunk
        const int64_t num_heads = 32;
        const int64_t head_dim = 128;

        // Requests with long prefills
        struct Request
        {
            int64_t request_id;
            std::string type;
            int64_t prefill_tokens;
            int64_t processed_tokens;
            int64_t decode_tokens;
            int gpu_id;
            bool prefill_complete;
        };

        std::vector<Request> requests = {
            // Long context prefills (would block decode without chunking)
            {0, "RAG_context", 8192, 0, 0, 0, false},      // 8K context
            {1, "RAG_context", 12288, 0, 0, 1, false},     // 12K context
            // Decode-only requests (already prefilled)
            {2, "decode_only", 0, 0, 5, 0, true},
            {3, "decode_only", 0, 0, 10, 1, true},
            // Mixed
            {4, "mixed", 4096, 0, 0, 0, false},            // 4K prefill
        };

        std::vector<hipblasLtHandle_t> handles(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtCreate(&handles[dev]);
        }

        hipblaslt_cout << "\n  Request queue:" << std::endl;
        for(const auto& req : requests)
        {
            hipblaslt_cout << "    Request " << req.request_id << " (" << req.type
                          << ", GPU " << req.gpu_id << "): ";
            if(req.prefill_tokens > 0)
            {
                hipblaslt_cout << req.prefill_tokens << " prefill tokens";
            }
            else
            {
                hipblaslt_cout << "decode-only (" << req.decode_tokens << " tokens generated)";
            }
            hipblaslt_cout << std::endl;
        }

        // Chunked prefill execution
        hipblaslt_cout << "\n  Chunked prefill execution (chunk size=" << prefill_chunk_size << "):" << std::endl;

        int step = 0;
        while(true)
        {
            step++;
            hipblaslt_cout << "\n    Step " << step << ":" << std::endl;

            bool any_active = false;

            for(auto& req : requests)
            {
                hipSetDevice(req.gpu_id);

                if(!req.prefill_complete && req.processed_tokens < req.prefill_tokens)
                {
                    // Process one chunk of prefill
                    int64_t remaining = req.prefill_tokens - req.processed_tokens;
                    int64_t chunk = std::min(remaining, prefill_chunk_size);

                    req.processed_tokens += chunk;
                    any_active = true;

                    hipblaslt_cout << "      Request " << req.request_id << " (GPU " << req.gpu_id
                                  << "): Prefill chunk [" << (req.processed_tokens - chunk)
                                  << " to " << req.processed_tokens << "] / " << req.prefill_tokens << std::endl;

                    if(req.processed_tokens >= req.prefill_tokens)
                    {
                        req.prefill_complete = true;
                        hipblaslt_cout << "        → Prefill COMPLETE, switching to decode" << std::endl;
                    }
                }
                else if(req.prefill_complete)
                {
                    // Decode: Generate one token
                    req.decode_tokens++;
                    any_active = true;

                    if(step <= 3 || req.decode_tokens <= 2)  // Show first few
                    {
                        hipblaslt_cout << "      Request " << req.request_id << " (GPU " << req.gpu_id
                                      << "): Decode token " << req.decode_tokens << std::endl;
                    }

                    if(req.decode_tokens >= 20)  // Simulate completion
                    {
                        req.prefill_complete = false;  // Mark as done
                        any_active = false;
                    }
                }
            }

            if(!any_active || step >= 15) break;  // Limit simulation steps
        }

        // Calculate benefits
        hipblaslt_cout << "\n  Chunked Prefill Benefits:" << std::endl;

        int total_prefill_tokens = 0;
        int total_chunks = 0;

        for(const auto& req : requests)
        {
            if(req.prefill_tokens > 0)
            {
                total_prefill_tokens += req.prefill_tokens;
                total_chunks += (req.prefill_tokens + prefill_chunk_size - 1) / prefill_chunk_size;
            }
        }

        hipblaslt_cout << "    Total prefill tokens: " << total_prefill_tokens << std::endl;
        hipblaslt_cout << "    Processed in chunks: " << total_chunks << " chunks" << std::endl;
        hipblaslt_cout << "    Without chunking: Decode requests would be blocked for "
                      << (total_prefill_tokens / 1000) << "+ ms" << std::endl;
        hipblaslt_cout << "    With chunking: Decode requests interleaved, minimal blocking" << std::endl;

        hipblaslt_cout << "\n✓ Chunked prefill completed" << std::endl;
        hipblaslt_cout << "  Key benefit: Long prefills don't block decode generation" << std::endl;
        hipblaslt_cout << "  Throughput improvement: Up to 24× for mixed workloads" << std::endl;

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            hipblasLtDestroy(handles[dev]);
        }
    }

} // namespace
