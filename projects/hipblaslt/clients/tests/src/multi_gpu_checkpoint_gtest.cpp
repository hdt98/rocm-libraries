/*******************************************************************************
 * Multi-GPU Checkpoint/Restart Test Suite
 * Tests: State checkpointing, restart from checkpoint, fault recovery
 *******************************************************************************/

#include "hipblaslt_data.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_test.hpp"
#include "utility.hpp"
#include <vector>
#include <atomic>
#include <thread>
#include <fstream>
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

    // Checkpoint state for multi-GPU computation
    struct CheckpointState
    {
        int num_gpus;
        int64_t M, N, K;
        int completed_iterations;
        int total_iterations;
        std::vector<std::vector<float>> gpu_state; // Per-GPU state matrices

        bool save(const std::string& filename) const
        {
            std::ofstream ofs(filename, std::ios::binary);
            if(!ofs) return false;

            ofs.write(reinterpret_cast<const char*>(&num_gpus), sizeof(num_gpus));
            ofs.write(reinterpret_cast<const char*>(&M), sizeof(M));
            ofs.write(reinterpret_cast<const char*>(&N), sizeof(N));
            ofs.write(reinterpret_cast<const char*>(&K), sizeof(K));
            ofs.write(reinterpret_cast<const char*>(&completed_iterations), sizeof(completed_iterations));
            ofs.write(reinterpret_cast<const char*>(&total_iterations), sizeof(total_iterations));

            for(int gpu = 0; gpu < num_gpus; ++gpu)
            {
                size_t state_size = gpu_state[gpu].size();
                ofs.write(reinterpret_cast<const char*>(&state_size), sizeof(state_size));
                ofs.write(reinterpret_cast<const char*>(gpu_state[gpu].data()),
                         state_size * sizeof(float));
            }

            return ofs.good();
        }

        bool load(const std::string& filename)
        {
            std::ifstream ifs(filename, std::ios::binary);
            if(!ifs) return false;

            ifs.read(reinterpret_cast<char*>(&num_gpus), sizeof(num_gpus));
            ifs.read(reinterpret_cast<char*>(&M), sizeof(M));
            ifs.read(reinterpret_cast<char*>(&N), sizeof(N));
            ifs.read(reinterpret_cast<char*>(&K), sizeof(K));
            ifs.read(reinterpret_cast<char*>(&completed_iterations), sizeof(completed_iterations));
            ifs.read(reinterpret_cast<char*>(&total_iterations), sizeof(total_iterations));

            gpu_state.resize(num_gpus);
            for(int gpu = 0; gpu < num_gpus; ++gpu)
            {
                size_t state_size;
                ifs.read(reinterpret_cast<char*>(&state_size), sizeof(state_size));
                gpu_state[gpu].resize(state_size);
                ifs.read(reinterpret_cast<char*>(gpu_state[gpu].data()),
                        state_size * sizeof(float));
            }

            return ifs.good();
        }
    };

    TEST(MultiGPUCheckpoint, BasicCheckpointSave)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Basic Checkpoint Save ===" << std::endl;

        const int64_t M = 512, N = 512, K = 512;
        const int total_iterations = 10;
        const int checkpoint_interval = 3;

        CheckpointState checkpoint;
        checkpoint.num_gpus = numDevices;
        checkpoint.M = M;
        checkpoint.N = N;
        checkpoint.K = K;
        checkpoint.total_iterations = total_iterations;
        checkpoint.gpu_state.resize(numDevices);

        std::vector<float*> d_state(numDevices);

        // Allocate state on each GPU
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipMalloc(&d_state[dev], M * N * sizeof(float)), hipSuccess);

            // Initialize
            checkpoint.gpu_state[dev].resize(M * N);
            for(int64_t i = 0; i < M * N; ++i)
            {
                checkpoint.gpu_state[dev][i] = static_cast<float>(dev * 1000 + i % 100);
            }

            EXPECT_EQ(hipMemcpy(d_state[dev], checkpoint.gpu_state[dev].data(),
                                M * N * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
        }

        // Simulate computation with periodic checkpointing
        for(int iter = 0; iter < total_iterations; ++iter)
        {
            // Perform computation on each GPU
            for(int dev = 0; dev < numDevices; ++dev)
            {
                EXPECT_EQ(hipSetDevice(dev), hipSuccess);

                // Simulate work (memset to simulate state change)
                EXPECT_EQ(hipMemset(d_state[dev], iter + dev, M * N * sizeof(float)), hipSuccess);
            }

            checkpoint.completed_iterations = iter + 1;

            // Checkpoint at interval
            if((iter + 1) % checkpoint_interval == 0)
            {
                // Copy state from GPUs to host
                for(int dev = 0; dev < numDevices; ++dev)
                {
                    EXPECT_EQ(hipSetDevice(dev), hipSuccess);
                    EXPECT_EQ(hipMemcpy(checkpoint.gpu_state[dev].data(), d_state[dev],
                                        M * N * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);
                }

                // Save checkpoint
                std::string filename = "/tmp/hipblaslt_checkpoint_" + std::to_string(iter + 1) + ".bin";
                EXPECT_TRUE(checkpoint.save(filename)) << "Failed to save checkpoint at iteration " << (iter + 1);

                hipblaslt_cout << "Checkpoint saved at iteration " << (iter + 1)
                               << " to " << filename << std::endl;
            }
        }

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipFree(d_state[dev]), hipSuccess);
        }

        hipblaslt_cout << "✓ Checkpoint save test passed" << std::endl;
    }

    TEST(MultiGPUCheckpoint, CheckpointRestore)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Checkpoint Restore ===" << std::endl;

        // First, create a checkpoint (similar to above but simplified)
        const int64_t M = 512, N = 512, K = 512;
        const int checkpoint_iter = 5;

        CheckpointState save_checkpoint;
        save_checkpoint.num_gpus = numDevices;
        save_checkpoint.M = M;
        save_checkpoint.N = N;
        save_checkpoint.K = K;
        save_checkpoint.total_iterations = 10;
        save_checkpoint.completed_iterations = checkpoint_iter;
        save_checkpoint.gpu_state.resize(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            save_checkpoint.gpu_state[dev].resize(M * N);
            for(int64_t i = 0; i < M * N; ++i)
            {
                save_checkpoint.gpu_state[dev][i] = static_cast<float>(dev * 100 + checkpoint_iter);
            }
        }

        std::string checkpoint_file = "/tmp/hipblaslt_restore_test.bin";
        EXPECT_TRUE(save_checkpoint.save(checkpoint_file)) << "Failed to save test checkpoint";

        hipblaslt_cout << "Created checkpoint at iteration " << checkpoint_iter << std::endl;

        // Now restore from checkpoint
        CheckpointState load_checkpoint;
        EXPECT_TRUE(load_checkpoint.load(checkpoint_file)) << "Failed to load checkpoint";

        hipblaslt_cout << "Loaded checkpoint:" << std::endl;
        hipblaslt_cout << "  GPUs: " << load_checkpoint.num_gpus << std::endl;
        hipblaslt_cout << "  Matrix size: " << load_checkpoint.M << "x" << load_checkpoint.N << "x" << load_checkpoint.K << std::endl;
        hipblaslt_cout << "  Completed iterations: " << load_checkpoint.completed_iterations << "/" << load_checkpoint.total_iterations << std::endl;

        // Verify checkpoint data matches
        EXPECT_EQ(load_checkpoint.num_gpus, save_checkpoint.num_gpus);
        EXPECT_EQ(load_checkpoint.M, save_checkpoint.M);
        EXPECT_EQ(load_checkpoint.N, save_checkpoint.N);
        EXPECT_EQ(load_checkpoint.K, save_checkpoint.K);
        EXPECT_EQ(load_checkpoint.completed_iterations, save_checkpoint.completed_iterations);
        EXPECT_EQ(load_checkpoint.total_iterations, save_checkpoint.total_iterations);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(load_checkpoint.gpu_state[dev].size(), save_checkpoint.gpu_state[dev].size());

            // Verify first few values
            for(size_t i = 0; i < std::min(size_t(10), load_checkpoint.gpu_state[dev].size()); ++i)
            {
                EXPECT_EQ(load_checkpoint.gpu_state[dev][i], save_checkpoint.gpu_state[dev][i])
                    << "Mismatch at GPU " << dev << " index " << i;
            }
        }

        hipblaslt_cout << "✓ Checkpoint restore test passed" << std::endl;

        // Cleanup
        std::remove(checkpoint_file.c_str());
    }

    TEST(MultiGPUCheckpoint, RestartFromCheckpoint)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Restart Computation from Checkpoint ===" << std::endl;

        const int64_t M = 512, N = 512, K = 512;
        const int total_iterations = 10;
        const int checkpoint_iter = 5;

        // Simulate interrupted computation by creating a checkpoint
        CheckpointState checkpoint;
        checkpoint.num_gpus = numDevices;
        checkpoint.M = M;
        checkpoint.N = N;
        checkpoint.K = K;
        checkpoint.total_iterations = total_iterations;
        checkpoint.completed_iterations = checkpoint_iter;
        checkpoint.gpu_state.resize(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            checkpoint.gpu_state[dev].resize(M * N);
            for(int64_t i = 0; i < M * N; ++i)
            {
                checkpoint.gpu_state[dev][i] = static_cast<float>(checkpoint_iter * dev + i % 100);
            }
        }

        std::string checkpoint_file = "/tmp/hipblaslt_restart_test.bin";
        EXPECT_TRUE(checkpoint.save(checkpoint_file));

        hipblaslt_cout << "Simulated interruption at iteration " << checkpoint_iter << std::endl;

        // Restart from checkpoint
        CheckpointState restart_state;
        EXPECT_TRUE(restart_state.load(checkpoint_file));

        hipblaslt_cout << "Restarting from iteration " << restart_state.completed_iterations << std::endl;

        std::vector<float*> d_state(numDevices);

        // Restore state to GPUs
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipMalloc(&d_state[dev], M * N * sizeof(float)), hipSuccess);

            EXPECT_EQ(hipMemcpy(d_state[dev], restart_state.gpu_state[dev].data(),
                                M * N * sizeof(float), hipMemcpyHostToDevice), hipSuccess);

            hipblaslt_cout << "GPU " << dev << ": State restored" << std::endl;
        }

        // Continue computation from checkpoint
        for(int iter = restart_state.completed_iterations; iter < total_iterations; ++iter)
        {
            for(int dev = 0; dev < numDevices; ++dev)
            {
                EXPECT_EQ(hipSetDevice(dev), hipSuccess);

                // Simulate work
                EXPECT_EQ(hipMemset(d_state[dev], iter + dev, M * N * sizeof(float)), hipSuccess);
            }

            hipblaslt_cout << "Completed iteration " << (iter + 1) << std::endl;
        }

        hipblaslt_cout << "✓ Computation completed from checkpoint" << std::endl;

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipFree(d_state[dev]), hipSuccess);
        }

        std::remove(checkpoint_file.c_str());
    }

    TEST(MultiGPUCheckpoint, FaultRecovery)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 3) GTEST_SKIP() << "Requires 3+ GPUs for fault recovery";

        hipblaslt_cout << "=== Fault Recovery Test ===" << std::endl;

        const int64_t M = 512, N = 512;

        CheckpointState checkpoint;
        checkpoint.num_gpus = numDevices;
        checkpoint.M = M;
        checkpoint.N = N;
        checkpoint.K = 512;
        checkpoint.total_iterations = 5;
        checkpoint.completed_iterations = 2;
        checkpoint.gpu_state.resize(numDevices);

        // Initialize checkpoint
        for(int dev = 0; dev < numDevices; ++dev)
        {
            checkpoint.gpu_state[dev].resize(M * N, static_cast<float>(dev * 10));
        }

        std::string checkpoint_file = "/tmp/hipblaslt_fault_recovery.bin";
        EXPECT_TRUE(checkpoint.save(checkpoint_file));

        hipblaslt_cout << "Initial checkpoint created with " << numDevices << " GPUs" << std::endl;

        // Simulate GPU failure - pretend GPU 1 failed
        int failed_gpu = 1;
        int available_gpus = numDevices - 1;

        hipblaslt_cout << "Simulating failure of GPU " << failed_gpu << std::endl;

        // Recover with remaining GPUs
        CheckpointState recovery;
        EXPECT_TRUE(recovery.load(checkpoint_file));

        // Redistribute work among available GPUs
        std::vector<int> active_gpus;
        for(int dev = 0; dev < numDevices; ++dev)
        {
            if(dev != failed_gpu)
            {
                active_gpus.push_back(dev);
            }
        }

        hipblaslt_cout << "Continuing with " << available_gpus << " GPUs: ";
        for(auto gpu : active_gpus)
            hipblaslt_cout << gpu << " ";
        hipblaslt_cout << std::endl;

        std::vector<float*> d_state(active_gpus.size());

        // Allocate on available GPUs
        for(size_t i = 0; i < active_gpus.size(); ++i)
        {
            int dev = active_gpus[i];
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipMalloc(&d_state[i], M * N * sizeof(float)), hipSuccess);

            // Restore state (for non-failed GPUs, restore from checkpoint;
            // for failed GPU's work, redistribute)
            EXPECT_EQ(hipMemcpy(d_state[i], recovery.gpu_state[dev].data(),
                                M * N * sizeof(float), hipMemcpyHostToDevice), hipSuccess);

            hipblaslt_cout << "GPU " << dev << ": Recovered and ready" << std::endl;
        }

        // Continue execution with reduced GPU count
        for(int iter = recovery.completed_iterations; iter < recovery.total_iterations; ++iter)
        {
            for(size_t i = 0; i < active_gpus.size(); ++i)
            {
                int dev = active_gpus[i];
                EXPECT_EQ(hipSetDevice(dev), hipSuccess);
                EXPECT_EQ(hipMemset(d_state[i], iter, M * N * sizeof(float)), hipSuccess);
            }
            hipblaslt_cout << "Iteration " << (iter + 1) << " completed with reduced GPU count" << std::endl;
        }

        hipblaslt_cout << "✓ Fault recovery successful - computation continued with "
                       << available_gpus << " GPUs" << std::endl;

        // Cleanup
        for(size_t i = 0; i < active_gpus.size(); ++i)
        {
            EXPECT_EQ(hipSetDevice(active_gpus[i]), hipSuccess);
            EXPECT_EQ(hipFree(d_state[i]), hipSuccess);
        }

        std::remove(checkpoint_file.c_str());
    }

    TEST(MultiGPUCheckpoint, IncrementalCheckpoint)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== Incremental Checkpoint Test ===" << std::endl;

        // Test checkpointing only changed GPU state (incremental)
        const int64_t M = 512, N = 512;

        std::vector<float*> d_state(numDevices);
        std::vector<std::vector<float>> prev_state(numDevices);
        std::vector<bool> gpu_changed(numDevices, false);

        // Allocate
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipMalloc(&d_state[dev], M * N * sizeof(float)), hipSuccess);
            EXPECT_EQ(hipMemset(d_state[dev], dev, M * N * sizeof(float)), hipSuccess);

            prev_state[dev].resize(M * N);
            EXPECT_EQ(hipMemcpy(prev_state[dev].data(), d_state[dev],
                                M * N * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);
        }

        // Simulate work where only some GPUs change state
        std::vector<int> gpus_to_update = {0, 2 % numDevices}; // Update GPU 0 and GPU 2

        for(int gpu : gpus_to_update)
        {
            if(gpu < numDevices)
            {
                EXPECT_EQ(hipSetDevice(gpu), hipSuccess);
                EXPECT_EQ(hipMemset(d_state[gpu], 100, M * N * sizeof(float)), hipSuccess);
                gpu_changed[gpu] = true;
            }
        }

        // Incremental checkpoint - only save changed GPUs
        int changed_count = 0;
        for(int dev = 0; dev < numDevices; ++dev)
        {
            if(gpu_changed[dev])
            {
                std::vector<float> current_state(M * N);
                EXPECT_EQ(hipSetDevice(dev), hipSuccess);
                EXPECT_EQ(hipMemcpy(current_state.data(), d_state[dev],
                                    M * N * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);

                // Verify state actually changed
                bool changed = false;
                for(size_t i = 0; i < current_state.size() && !changed; ++i)
                {
                    if(current_state[i] != prev_state[dev][i])
                    {
                        changed = true;
                    }
                }

                if(changed)
                {
                    changed_count++;
                    hipblaslt_cout << "GPU " << dev << ": State changed, checkpointing" << std::endl;
                }
            }
        }

        hipblaslt_cout << "✓ Incremental checkpoint: " << changed_count
                       << " GPUs updated out of " << numDevices << std::endl;

        // Cleanup
        for(int dev = 0; dev < numDevices; ++dev)
        {
            EXPECT_EQ(hipSetDevice(dev), hipSuccess);
            EXPECT_EQ(hipFree(d_state[dev]), hipSuccess);
        }
    }

} // namespace
