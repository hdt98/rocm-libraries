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

#include "hipblaslt_data.hpp"
#include "hipblaslt_parse_data.hpp"
#include "hipblaslt_test.hpp"
#include "test_cleanup.hpp"
#include "utility.hpp"
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <chrono>
#include <thread>
#ifndef _WIN32
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

using namespace testing;

class ConfigurableEventListener : public TestEventListener
{
    TestEventListener* const eventListener;
    std::atomic_size_t       skipped_tests{0}; // Number of skipped tests.
    std::atomic_size_t       current_test_number{0}; // Current test number (incremental counter).

public:
    bool showTestCases      = true; // Show the names of each test case.
    bool showTestNames      = true; // Show the names of each test.
    bool showSuccesses      = true; // Show each success.
    bool showInlineFailures = true; // Show each failure as it occurs.
    bool showEnvironment    = true; // Show the setup of the global environment.
    bool showInlineSkips    = true; // Show when we skip a test.

    explicit ConfigurableEventListener(TestEventListener* theEventListener)
        : eventListener(theEventListener)
    {
    }

    ~ConfigurableEventListener() override
    {
        delete eventListener;
    }

    void OnTestProgramStart(const UnitTest& unit_test) override
    {
        eventListener->OnTestProgramStart(unit_test);
    }

    void OnTestIterationStart(const UnitTest& unit_test, int iteration) override
    {
        eventListener->OnTestIterationStart(unit_test, iteration);
    }

    void OnEnvironmentsSetUpStart(const UnitTest& unit_test) override
    {
        if(showEnvironment)
            eventListener->OnEnvironmentsSetUpStart(unit_test);
    }

    void OnEnvironmentsSetUpEnd(const UnitTest& unit_test) override
    {
        if(showEnvironment)
            eventListener->OnEnvironmentsSetUpEnd(unit_test);
    }

    void OnTestCaseStart(const TestCase& test_case) override
    {
        if(showTestCases)
            eventListener->OnTestCaseStart(test_case);
    }

    void OnTestStart(const TestInfo& test_info) override
    {
        ++current_test_number;
        // Get total test count (after filtering)
        int total_tests = UnitTest::GetInstance()->test_to_run_count();
        // Always print test number with total, regardless of showTestNames setting
        hipblaslt_cout << "[Test #" << current_test_number << "/" << total_tests << "] " << std::flush;
        if(showTestNames)
            eventListener->OnTestStart(test_info);
    }

    void OnTestPartResult(const TestPartResult& result) override
    {
        if(!strcmp(result.message(), LIMITED_MEMORY_STRING_GTEST))
        {
            if(showInlineSkips)
                hipblaslt_cout << "Skipped test due to limited memory environment." << std::endl;
            ++skipped_tests;
        }
        else if(!strcmp(result.message(), TOO_MANY_DEVICES_STRING_GTEST))
        {
            if(showInlineSkips)
                hipblaslt_cout << "Skipped test due to too few GPUs." << std::endl;
            ++skipped_tests;
        }
        eventListener->OnTestPartResult(result);
    }

    void OnTestEnd(const TestInfo& test_info) override
    {
        if(test_info.result()->Failed() ? showInlineFailures : showSuccesses)
            eventListener->OnTestEnd(test_info);
    }

    void OnTestCaseEnd(const TestCase& test_case) override
    {
        if(showTestCases)
            eventListener->OnTestCaseEnd(test_case);
    }

    void OnEnvironmentsTearDownStart(const UnitTest& unit_test) override
    {
        if(showEnvironment)
            eventListener->OnEnvironmentsTearDownStart(unit_test);
    }

    void OnEnvironmentsTearDownEnd(const UnitTest& unit_test) override
    {
        if(showEnvironment)
            eventListener->OnEnvironmentsTearDownEnd(unit_test);
    }

    void OnTestIterationEnd(const UnitTest& unit_test, int iteration) override
    {
        eventListener->OnTestIterationEnd(unit_test, iteration);
    }

    void OnTestProgramEnd(const UnitTest& unit_test) override
    {
        if(skipped_tests)
            hipblaslt_cout << "[ SKIPPED  ] " << skipped_tests << " tests." << std::endl;
        eventListener->OnTestProgramEnd(unit_test);
    }
};

// Set the listener for Google Tests
static void hipblaslt_set_listener()
{
    // remove the default listener
    auto& listeners       = testing::UnitTest::GetInstance()->listeners();
    auto  default_printer = listeners.Release(listeners.default_result_printer());

    // add our listener, by default everything is on (the same as using the default listener)
    // here I am turning everything off so I only see the 3 lines for the result
    // (plus any failures at the end), like:

    // [==========] Running 149 tests from 53 test cases.
    // [==========] 149 tests from 53 test cases ran. (1 ms total)
    // [  PASSED  ] 149 tests.
    //
    auto* listener       = new ConfigurableEventListener(default_printer);
    auto* gtest_listener = getenv("GTEST_LISTENER");

    if(gtest_listener && !strcmp(gtest_listener, "NO_PASS_LINE_IN_LOG"))
    {
        listener->showTestNames      = false;
        listener->showSuccesses      = false;
        listener->showInlineFailures = false;
        listener->showInlineSkips    = false;
    }

    listeners.Append(listener);
}

static int hipblaslt_version()
{
    int                    version;
    hipblaslt_local_handle handle;
    hipblasLtGetVersion(handle, &version);
    return version;
}

static void hipblaslt_print_usage_warning()
{
    std::string warning(
        "parsing of test data may take a couple minutes before any test output appears...");

    hipblaslt_cout << "info: " << warning << "\n" << std::endl;
}

static std::string hipblaslt_capture_args(int argc, char** argv)
{
    std::ostringstream cmdLine;
    cmdLine << "command line: ";
    for(int i = 0; i < argc; i++)
    {
        if(argv[i])
            cmdLine << std::string(argv[i]) << " ";
    }
    return cmdLine.str();
}

static void hipblaslt_print_args(const std::string& args)
{
    hipblaslt_cout << args << std::endl;
    hipblaslt_cout.flush();
}

// Device Query
static void hipblaslt_set_test_device()
{
    hipDeviceProp_t props;
    int device_id    = 0;
    int device_count = query_device_property(device_id, props);
    if(device_count <= device_id)
    {
        hipblaslt_cerr << "Error: invalid device ID. There may not be such device ID." << std::endl;
        exit(-1);
    }
    set_device(device_id);
}

#ifndef _WIN32
// Helper function to merge multiple GTest JSON output files into one
static void merge_gtest_json_files(const std::vector<std::string>& input_files,
                                     const std::string& output_file)
{
    if(input_files.empty())
        return;

    std::ofstream merged_output(output_file);
    if(!merged_output.is_open())
    {
        hipblaslt_cerr << "Warning: Could not create merged JSON file: " << output_file << std::endl;
        return;
    }

    // Simple approach: read entire JSON content from each file
    int total_tests = 0;
    int total_failures = 0;
    int total_disabled = 0;
    int total_errors = 0;
    double total_time = 0.0;
    std::string timestamp;
    std::vector<std::string> all_testsuite_contents;

    // Read each JSON file
    for(const auto& input_file : input_files)
    {
        std::ifstream input(input_file);
        if(!input.is_open())
            continue;

        // Read entire file into string
        std::stringstream buffer;
        buffer << input.rdbuf();
        std::string content = buffer.str();
        input.close();

        // Extract top-level fields (only from first line after field name)
        size_t pos;

        // Extract timestamp (only once)
        if(timestamp.empty())
        {
            pos = content.find("\"timestamp\":");
            if(pos != std::string::npos)
            {
                size_t start = content.find("\"", pos + 12);
                size_t end = content.find("\"", start + 1);
                if(start != std::string::npos && end != std::string::npos)
                {
                    timestamp = content.substr(start + 1, end - start - 1);
                }
            }
        }

        // Sum up numeric fields - find FIRST occurrence only (top level)
        pos = content.find("\"tests\":");
        if(pos != std::string::npos && pos < content.find("\"testsuites\""))
        {
            total_tests += std::atoi(content.substr(pos + 8).c_str());
        }

        pos = content.find("\"failures\":");
        if(pos != std::string::npos && pos < content.find("\"testsuites\""))
        {
            total_failures += std::atoi(content.substr(pos + 11).c_str());
        }

        pos = content.find("\"disabled\":");
        if(pos != std::string::npos && pos < content.find("\"testsuites\""))
        {
            total_disabled += std::atoi(content.substr(pos + 11).c_str());
        }

        pos = content.find("\"errors\":");
        if(pos != std::string::npos && pos < content.find("\"testsuites\""))
        {
            total_errors += std::atoi(content.substr(pos + 9).c_str());
        }

        pos = content.find("\"time\":");
        if(pos != std::string::npos && pos < content.find("\"testsuites\""))
        {
            size_t start = content.find("\"", pos + 7);
            size_t end = content.find("s\"", start + 1);
            if(start != std::string::npos && end != std::string::npos)
            {
                total_time += std::atof(content.substr(start + 1, end - start - 1).c_str());
            }
        }

        // Extract testsuites array content (everything between [ and ])
        pos = content.find("\"testsuites\": [");
        if(pos != std::string::npos)
        {
            size_t start = pos + 15; // After "testsuites": [
            size_t end = content.rfind("]");
            if(start < end && end != std::string::npos)
            {
                std::string testsuites_content = content.substr(start, end - start);
                // Trim whitespace
                while(!testsuites_content.empty() &&
                      (testsuites_content.front() == ' ' || testsuites_content.front() == '\n'))
                    testsuites_content.erase(0, 1);
                while(!testsuites_content.empty() &&
                      (testsuites_content.back() == ' ' || testsuites_content.back() == '\n' || testsuites_content.back() == ','))
                    testsuites_content.pop_back();

                if(!testsuites_content.empty())
                {
                    all_testsuite_contents.push_back(testsuites_content);
                }
            }
        }
    }

    // Write merged JSON
    merged_output << "{\n";
    merged_output << "  \"tests\": " << total_tests << ",\n";
    merged_output << "  \"failures\": " << total_failures << ",\n";
    merged_output << "  \"disabled\": " << total_disabled << ",\n";
    merged_output << "  \"errors\": " << total_errors << ",\n";
    if(!timestamp.empty())
    {
        merged_output << "  \"timestamp\": \"" << timestamp << "\",\n";
    }
    merged_output << "  \"time\": \"" << total_time << "s\",\n";
    merged_output << "  \"name\": \"AllTests\",\n";
    merged_output << "  \"testsuites\": [\n";

    for(size_t i = 0; i < all_testsuite_contents.size(); i++)
    {
        merged_output << all_testsuite_contents[i];
        if(i < all_testsuite_contents.size() - 1)
        {
            merged_output << ",\n";
        }
        else
        {
            merged_output << "\n";
        }
    }

    merged_output << "  ]\n";
    merged_output << "}\n";
    merged_output.close();

    hipblaslt_cout << "Merged JSON output saved to: " << output_file << std::endl;
}

// Function to run tests in parallel across multiple GPUs
static int run_tests_parallel_gpus(int argc, char** argv, int num_gpus)
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

    // Track original JSON output file for merging later
    std::string original_json_output;
    bool has_json_output = false;

    // Check if user specified --gtest_output=json:file or just --gtest_output=json
    for(int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if(arg.find("--gtest_output=") == 0)
        {
            size_t colon_pos = arg.find(":");
            if(colon_pos != std::string::npos)
            {
                // Format: --gtest_output=json:filename.json
                std::string format = arg.substr(15, colon_pos - 15);
                if(format == "json")
                {
                    original_json_output = arg.substr(colon_pos + 1);
                    has_json_output = true;
                }
            }
            else
            {
                // Format: --gtest_output=json (uses default filename)
                std::string format = arg.substr(15);
                if(format == "json")
                {
                    original_json_output = "test_detail.json"; // GTest default
                    has_json_output = true;
                }
            }
        }
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
        if(arg.find("--parallel_gpus") == std::string::npos)
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

                // Skip --parallel_gpus
                if(arg.find("--parallel_gpus") != std::string::npos)
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

            // Add internal flag to indicate this is a child process in parallel mode
            std::string internal_flag = "--internal-parallel-child";
            arg_storage.push_back(internal_flag);
            new_argv.push_back(arg_storage.back().c_str());

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
    hipblaslt_cout << "\nLog files saved in:" << std::endl;
    for(size_t i = 0; i < output_files.size(); i++)
    {
        hipblaslt_cout << "  GPU " << i << ": " << output_files[i] << std::endl;
    }
    hipblaslt_cout << "========================================\n" << std::endl;

    // Merge JSON output files if --gtest_output=json was specified
    if(has_json_output && !original_json_output.empty())
    {
        hipblaslt_cout << "\nMerging JSON output files..." << std::endl;

        // Build list of GPU-specific JSON files
        std::vector<std::string> json_files;
        for(size_t i = 0; i < child_pids.size(); i++)
        {
            // Construct the GPU-specific filename
            size_t dot_pos = original_json_output.rfind(".");
            std::string gpu_json_file;
            if(dot_pos != std::string::npos)
            {
                gpu_json_file = original_json_output.substr(0, dot_pos) + "_gpu" +
                                  std::to_string(i) + original_json_output.substr(dot_pos);
            }
            else
            {
                gpu_json_file = original_json_output + "_gpu" + std::to_string(i);
            }
            json_files.push_back(gpu_json_file);
        }

        // Merge all JSON files into the original filename
        merge_gtest_json_files(json_files, original_json_output);

        hipblaslt_cout << "\nIndividual GPU JSON files:" << std::endl;
        for(size_t i = 0; i < json_files.size(); i++)
        {
            hipblaslt_cout << "  GPU " << i << ": " << json_files[i] << std::endl;
        }
    }

    return total_exit_code;
}
#endif // _WIN32

/*****************
 * Main function *
 *****************/
int main(int argc, char** argv)
{
    std::string args = hipblaslt_capture_args(argc, argv);

    // Check for --help to add our custom options
    for(int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if(arg == "--help" || arg == "-h" || arg == "-?" || arg == "/?" || arg == "--help-all")
        {
            hipblaslt_cout << "\nhipBLASLt Test Options:\n";
            hipblaslt_cout << "  --parallel_gpus=N\n";
            hipblaslt_cout << "  --parallel_gpus N\n";
            hipblaslt_cout << "      Run tests in parallel across N GPUs. Tests are automatically\n";
            hipblaslt_cout << "      split evenly across the specified number of GPUs. Each GPU\n";
            hipblaslt_cout << "      runs its assigned tests independently.\n";
            hipblaslt_cout << "      Example: ./hipblaslt-test --parallel_gpus 8 --gtest_filter=\"*smoke*\"\n";
            hipblaslt_cout << "      Note: If --gtest_output=json:file.json is specified, individual\n";
            hipblaslt_cout << "            GPU results are saved as file_gpu0.json, file_gpu1.json, etc.,\n";
            hipblaslt_cout << "            and automatically merged into file.json at the end.\n";
            hipblaslt_cout << "\n";
            hipblaslt_cout << "  --verbose\n";
            hipblaslt_cout << "  -v\n";
            hipblaslt_cout << "      Enable verbose output in parallel GPU mode. Shows command line\n";
            hipblaslt_cout << "      arguments for each GPU process.\n";
            hipblaslt_cout << "\n";
            break;
        }
    }

#ifndef _WIN32
    // Check for --parallel_gpus argument
    int parallel_gpus = 0;
    for(int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if(arg.find("--parallel_gpus=") == 0)
        {
            parallel_gpus = std::atoi(arg.substr(16).c_str());
            break;
        }
        else if(arg == "--parallel_gpus" && i + 1 < argc)
        {
            parallel_gpus = std::atoi(argv[i + 1]);
            break;
        }
    }

    // If parallel GPUs requested, use parallel execution
    if(parallel_gpus > 1)
    {
        return run_tests_parallel_gpus(argc, argv, parallel_gpus);
    }
#endif

    // Check if this is a child process in parallel mode
    bool is_parallel_child = false;
    bool verbose_mode = false;
    for(int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if(arg == "--internal-parallel-child")
        {
            is_parallel_child = true;
        }
        if(arg == "--verbose" || arg == "-v")
        {
            verbose_mode = true;
        }
    }

    // Normal single-GPU execution path

    // Optional timing for parallel child processes
    auto start_time = std::chrono::high_resolution_clock::now();

    // Set signal handler
    hipblaslt_test_sigaction();

    if(is_parallel_child && verbose_mode)
    {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time).count();
        hipblaslt_cout << "[Timing] Signal handler: " << elapsed << "ms" << std::endl;
    }

    hipblaslt_print_version();

    if(is_parallel_child && verbose_mode)
    {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time).count();
        hipblaslt_cout << "[Timing] Print version: " << elapsed << "ms" << std::endl;
    }

    // Set test device
    hipblaslt_set_test_device();

    if(is_parallel_child && verbose_mode)
    {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time).count();
        hipblaslt_cout << "[Timing] Set device: " << elapsed << "ms" << std::endl;
    }

    hipblaslt_print_usage_warning();

    // Set data file path
    hipblaslt_parse_data(argc, argv, hipblaslt_exepath() + "hipblaslt_gtest.data");

    if(is_parallel_child && verbose_mode)
    {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time).count();
        hipblaslt_cout << "[Timing] Parse data file: " << elapsed << "ms" << std::endl;
    }

    // Initialize Google Tests
    testing::InitGoogleTest(&argc, argv);

    if(is_parallel_child && verbose_mode)
    {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time).count();
        hipblaslt_cout << "[Timing] Init Google Test: " << elapsed << "ms" << std::endl;
    }

    // Free up all temporary data generated during test creation
    test_cleanup::cleanup();

    if(is_parallel_child && verbose_mode)
    {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time).count();
        hipblaslt_cout << "[Timing] Cleanup: " << elapsed << "ms" << std::endl;
    }

    // Set Google Test listener
    hipblaslt_set_listener();

    if(is_parallel_child && verbose_mode)
    {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time).count();
        hipblaslt_cout << "[Timing] Set listener: " << elapsed << "ms" << std::endl;
    }

    // Run the tests
    int status = RUN_ALL_TESTS();

    if(is_parallel_child && verbose_mode)
    {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time).count();
        hipblaslt_cout << "[Timing] Run tests: " << elapsed << "ms" << std::endl;
    }

    // Failures printed at end for reporting so repeat version info
    hipblaslt_print_version();

    // Print command line at the end
    // Skip in parallel child mode unless verbose is enabled
    if(!is_parallel_child || verbose_mode)
    {
        hipblaslt_print_args(args);
    }

    //hipblaslt_shutdown();

    return status;
}
