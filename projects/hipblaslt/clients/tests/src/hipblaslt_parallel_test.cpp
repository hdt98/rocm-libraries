/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "hipblaslt_parallel_test.hpp"
#include "utility.hpp"

#ifndef _WIN32
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <thread>
#include <cstdio>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <hip/hip_runtime.h>

// Function to run tests in parallel across multiple GPUs
int run_tests_parallel_gpus(int argc, char** argv, int num_gpus)
{
    hipblaslt_cout << "\n========================================" << std::endl;
    hipblaslt_cout << "Parallel GPU Execution Mode" << std::endl;
    hipblaslt_cout << "Running tests across " << num_gpus << " GPUs" << std::endl;
    hipblaslt_cout << "========================================\n" << std::endl;

    // Check available GPUs
    int available_gpus = 0;
    if(hipGetDeviceCount(&available_gpus) != hipSuccess || available_gpus < 1)
    {
        hipblaslt_cerr << "Error: No GPUs detected" << std::endl;
        return 1;
    }

    if(num_gpus > available_gpus)
    {
        hipblaslt_cerr << "Warning: Requested " << num_gpus << " GPUs but only "
                       << available_gpus << " available. Using " << available_gpus << " GPUs."
                       << std::endl;
        num_gpus = available_gpus;
    }

    // Get list of all tests by calling gtest with --gtest_list_tests
    // IMPORTANT: Include --gtest_filter so we only split filtered tests
    std::vector<std::string> all_tests;
    std::string test_list_output = "/tmp/hipblaslt_test_list_" + std::to_string(getpid()) + ".txt";

    // Build command to list tests, INCLUDING any --gtest_filter
    std::string list_cmd = std::string(argv[0]) + " --gtest_list_tests";
    for(int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if(arg.find("--num_gpus") == std::string::npos)
        {
            list_cmd += " \"" + arg + "\"";  // Quote arguments to preserve filters
        }
    }
    list_cmd += " > " + test_list_output + " 2>&1";

    if(system(list_cmd.c_str()) != 0)
    {
        hipblaslt_cerr << "Error: Failed to list tests" << std::endl;
        return 1;
    }

    // Parse test list
    std::ifstream test_file(test_list_output);
    if(!test_file.is_open())
    {
        hipblaslt_cerr << "Error: Could not open test list file" << std::endl;
        return 1;
    }

    std::string current_suite;
    std::string line;
    while(std::getline(test_file, line))
    {
        if(line.empty())
            continue;

        // Remove trailing whitespace
        while(!line.empty() && (line.back() == ' ' || line.back() == '\t' || line.back() == '\r'))
            line.pop_back();

        if(line.empty())
            continue;

        // Suite names don't start with spaces
        if(line[0] != ' ')
        {
            current_suite = line;
        }
        else
        {
            // Test names start with spaces - remove leading whitespace
            size_t start = line.find_first_not_of(" \t");
            if(start != std::string::npos && !current_suite.empty())
            {
                std::string test_name = line.substr(start);

                // Strip comment part (everything after '#')
                size_t comment_pos = test_name.find('#');
                if(comment_pos != std::string::npos)
                {
                    test_name = test_name.substr(0, comment_pos);
                    // Remove trailing whitespace before the comment
                    while(!test_name.empty() && (test_name.back() == ' ' || test_name.back() == '\t'))
                        test_name.pop_back();
                }

                all_tests.push_back(current_suite + test_name);
            }
        }
    }
    test_file.close();
    remove(test_list_output.c_str());

    if(all_tests.empty())
    {
        hipblaslt_cerr << "Error: No tests found to run" << std::endl;
        return 1;
    }

    hipblaslt_cout << "Found " << all_tests.size() << " tests total" << std::endl;

    size_t tests_per_gpu = (all_tests.size() + num_gpus - 1) / num_gpus;
    hipblaslt_cout << "Tests per GPU: ~" << tests_per_gpu << std::endl;

    // Calculate and display OpenMP thread distribution
    const char* env_threads = getenv("OMP_NUM_THREADS");
    int current_threads = env_threads ? std::atoi(env_threads) : std::thread::hardware_concurrency();
    int threads_per_gpu = std::max(1, current_threads / num_gpus);
    hipblaslt_cout << "OpenMP threads per GPU: " << threads_per_gpu
                   << " (total available: " << current_threads << ")" << std::endl;
    hipblaslt_cout << std::endl;

    // Split tests across GPUs using Google Test's built-in sharding
    std::vector<pid_t> child_pids;
    std::vector<std::string> output_files;

    for(int gpu = 0; gpu < num_gpus; gpu++)
    {
        std::string output_file = "/tmp/hipblaslt_gpu" + std::to_string(gpu) + "_" +
                                  std::to_string(getpid()) + ".log";
        output_files.push_back(output_file);

        hipblaslt_cout << "GPU " << gpu << ": Running ~" << tests_per_gpu
                       << " tests" << std::endl;

        pid_t pid = fork();
        if(pid == 0)
        {
            // Child process - run tests on this GPU

            // Set which GPU to use
            std::string gpu_env = std::to_string(gpu);
            setenv("HIP_VISIBLE_DEVICES", gpu_env.c_str(), 1);

            // Set optimal OpenMP threads per GPU process
            // Get current OMP_NUM_THREADS setting, or use hardware concurrency
            const char* env_threads = getenv("OMP_NUM_THREADS");
            int current_threads = env_threads ? std::atoi(env_threads) : std::thread::hardware_concurrency();

            // Divide CPU threads among GPU processes to avoid oversubscription
            // Each GPU process should use (total_threads / num_gpus) threads
            int threads_per_gpu = std::max(1, current_threads / num_gpus);
            setenv("OMP_NUM_THREADS", std::to_string(threads_per_gpu).c_str(), 1);

            // Use Google Test's built-in sharding
            setenv("GTEST_TOTAL_SHARDS", std::to_string(num_gpus).c_str(), 1);
            setenv("GTEST_SHARD_INDEX", std::to_string(gpu).c_str(), 1);

            // Redirect output to log file to avoid interleaved output
            int fd = open(output_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if(fd >= 0)
            {
                dup2(fd, STDOUT_FILENO);
                dup2(fd, STDERR_FILENO);
                close(fd);
            }

            // Build new argv
            std::vector<const char*> new_argv;
            std::vector<std::string> arg_storage; // Store modified arguments

            new_argv.push_back(argv[0]);

            for(int i = 1; i < argc; i++)
            {
                std::string arg = argv[i];

                // Skip --num_gpus
                if(arg.find("--num_gpus") != std::string::npos)
                    continue;

                // Keep --gtest_filter (it will be applied before sharding)
                // Sharding happens AFTER filtering

                // Modify --gtest_output to include GPU number
                if(arg.find("--gtest_output=") == 0)
                {
                    size_t colon_pos = arg.find(":");
                    std::string format;
                    std::string filename;

                    if(colon_pos != std::string::npos)
                    {
                        // Format: --gtest_output=json:file.json
                        format = arg.substr(15, colon_pos - 15); // After "=" before ":"
                        filename = arg.substr(colon_pos + 1);
                    }
                    else
                    {
                        // Format: --gtest_output=json (uses default filename)
                        format = arg.substr(15);
                        if(format == "json")
                        {
                            filename = "test_detail.json"; // GTest default
                        }
                        else
                        {
                            // For other formats, just pass through
                            new_argv.push_back(argv[i]);
                            continue;
                        }
                    }

                    // Insert GPU number before file extension
                    size_t dot_pos = filename.rfind(".");
                    std::string new_filename;
                    if(dot_pos != std::string::npos)
                    {
                        new_filename = filename.substr(0, dot_pos) + "_gpu" +
                                     std::to_string(gpu) + filename.substr(dot_pos);
                    }
                    else
                    {
                        new_filename = filename + "_gpu" + std::to_string(gpu);
                    }

                    std::string new_output_arg = "--gtest_output=" + format + ":" + new_filename;
                    arg_storage.push_back(new_output_arg);
                    new_argv.push_back(arg_storage.back().c_str());
                    continue;
                }

                new_argv.push_back(argv[i]);
            }

            new_argv.push_back(nullptr);

            // Execute
            execvp(argv[0], const_cast<char* const*>(new_argv.data()));

            // If exec fails
            hipblaslt_cerr << "Failed to exec for GPU " << gpu << std::endl;
            exit(1);
        }
        else if(pid > 0)
        {
            child_pids.push_back(pid);
        }
        else
        {
            hipblaslt_cerr << "Error: Failed to fork for GPU " << gpu << std::endl;
            return 1;
        }
    }

    // Wait for all children and collect results (without printing yet)
    hipblaslt_cout << "\nWaiting for all GPUs to complete..." << std::endl;

    int total_exit_code = 0;
    int gpus_passed     = 0;
    int gpus_failed     = 0;
    int total_tests_ran    = 0;
    int total_tests_passed = 0;
    int total_tests_failed = 0;
    std::vector<int> exit_codes(child_pids.size());
    std::vector<bool> normal_exit(child_pids.size());
    std::vector<std::string> gpu_summaries(child_pids.size());
    std::vector<int> gpu_tests_ran(child_pids.size(), 0);
    std::vector<int> gpu_tests_passed(child_pids.size(), 0);
    std::vector<int> gpu_tests_failed(child_pids.size(), 0);
    std::vector<double> gpu_time_ms(child_pids.size(), 0.0);

    // Wait for ALL children first before printing any results
    for(size_t i = 0; i < child_pids.size(); i++)
    {
        int status;
        waitpid(child_pids[i], &status, 0);

        if(WIFEXITED(status))
        {
            normal_exit[i] = true;
            exit_codes[i]  = WEXITSTATUS(status);
            if(exit_codes[i] == 0)
            {
                gpus_passed++;
            }
            else
            {
                gpus_failed++;
                total_exit_code = 1;
            }
        }
        else
        {
            normal_exit[i] = false;
            exit_codes[i]  = -1;
            gpus_failed++;
            total_exit_code = 1;
        }

        // Extract summary from log file and parse test counts
        std::ifstream log_file(output_files[i]);
        if(log_file.is_open())
        {
            std::string line;
            std::vector<std::string> summary_lines;
            bool in_summary = false;

            while(std::getline(log_file, line))
            {
                // Parse total tests ran: "[==========] 87 tests from 1 test suite ran."
                if(line.find("[==========]") != std::string::npos && line.find("tests") != std::string::npos && line.find("ran.") != std::string::npos)
                {
                    in_summary = true;
                    // Extract number of tests
                    size_t pos = line.find("]");
                    if(pos != std::string::npos)
                    {
                        std::istringstream iss(line.substr(pos + 1));
                        iss >> gpu_tests_ran[i];
                    }
                }

                // Parse passed tests: "[  PASSED  ] 87 tests."
                if(line.find("[  PASSED  ]") != std::string::npos && line.find("tests") != std::string::npos)
                {
                    size_t pos = line.find("]");
                    if(pos != std::string::npos)
                    {
                        std::istringstream iss(line.substr(pos + 1));
                        iss >> gpu_tests_passed[i];
                    }
                }

                // Parse failed tests: "[  FAILED  ] 2 tests, listed below:"
                if(line.find("[  FAILED  ]") != std::string::npos && line.find("tests") != std::string::npos)
                {
                    size_t pos = line.find("]");
                    if(pos != std::string::npos)
                    {
                        std::istringstream iss(line.substr(pos + 1));
                        iss >> gpu_tests_failed[i];
                    }
                }

                // Parse time: "[==========] 87 tests from 1 test suite ran. (1234 ms total)"
                if(line.find("[==========]") != std::string::npos && line.find("ms total") != std::string::npos)
                {
                    size_t pos = line.find("(");
                    if(pos != std::string::npos)
                    {
                        std::istringstream iss(line.substr(pos + 1));
                        iss >> gpu_time_ms[i];
                    }
                }

                if(in_summary)
                {
                    summary_lines.push_back(line);
                }
            }
            log_file.close();

            // Store the summary for this GPU
            if(!summary_lines.empty())
            {
                for(const auto& summary_line : summary_lines)
                {
                    gpu_summaries[i] += summary_line + "\n";
                }
            }

            // Accumulate totals
            total_tests_ran += gpu_tests_ran[i];
            total_tests_passed += gpu_tests_passed[i];
            total_tests_failed += gpu_tests_failed[i];
        }
    }

    // Now print all results together
    hipblaslt_cout << "\n========================================" << std::endl;
    hipblaslt_cout << "Parallel GPU Test Summary" << std::endl;
    hipblaslt_cout << "========================================\n" << std::endl;

    // Print individual GPU results with their GTest summaries
    for(size_t i = 0; i < child_pids.size(); i++)
    {
        hipblaslt_cout << "GPU " << i << ":" << std::endl;

        if(!gpu_summaries[i].empty())
        {
            hipblaslt_cout << gpu_summaries[i];
        }
        else if(normal_exit[i])
        {
            hipblaslt_cout << "  Exit code: " << exit_codes[i] << std::endl;
        }
        else
        {
            hipblaslt_cout << "  Terminated abnormally" << std::endl;
        }
        hipblaslt_cout << std::endl;
    }

    // Calculate average time
    double total_time_ms = 0.0;
    int valid_times = 0;
    for(size_t i = 0; i < gpu_time_ms.size(); i++)
    {
        if(gpu_time_ms[i] > 0.0)
        {
            total_time_ms += gpu_time_ms[i];
            valid_times++;
        }
    }
    double avg_time_ms = (valid_times > 0) ? (total_time_ms / valid_times) : 0.0;

    hipblaslt_cout << "----------------------------------------" << std::endl;
    hipblaslt_cout << "OVERALL SUMMARY (across all GPUs):" << std::endl;
    hipblaslt_cout << "Total tests run:  " << total_tests_ran << std::endl;
    hipblaslt_cout << "Total PASSED:     " << total_tests_passed << std::endl;
    hipblaslt_cout << "Total FAILED:     " << total_tests_failed << std::endl;
    if(avg_time_ms > 0.0)
    {
        hipblaslt_cout << "Average time:     " << avg_time_ms << " ms" << std::endl;
    }
    hipblaslt_cout << "\nGPU Summary:" << std::endl;
    hipblaslt_cout << "GPUs used:        " << num_gpus << std::endl;
    hipblaslt_cout << "GPUs passed:      " << gpus_passed << std::endl;
    hipblaslt_cout << "GPUs failed:      " << gpus_failed << std::endl;

    // Clean up log files on success to avoid accumulation in /tmp on CI systems
    if(total_exit_code == 0)
    {
        hipblaslt_cout << "\nAll tests passed - cleaning up log files" << std::endl;
        for(size_t i = 0; i < output_files.size(); i++)
        {
            remove(output_files[i].c_str());
        }
    }
    else
    {
        hipblaslt_cout << "\nLog files saved in:" << std::endl;
        for(size_t i = 0; i < output_files.size(); i++)
        {
            hipblaslt_cout << "  GPU " << i << ": " << output_files[i] << std::endl;
        }
    }
    hipblaslt_cout << "========================================\n" << std::endl;

    return total_exit_code;
}
#endif // _WIN32
