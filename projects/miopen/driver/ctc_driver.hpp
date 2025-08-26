/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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
#pragma once

#include "InputFlags.hpp"
#include "ctc_verify.hpp"
#include "driver.hpp"
#include "random.hpp"
#include "timer.hpp"
#include "util_driver.hpp"
#include "util_file.hpp"

#include <miopen/miopen.h>

#include <../test/verify.hpp>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>
#include <stdexcept>

// Helper function to load vector data from files
template<typename T>
void LoadVectorFromFile(const std::string& filename, std::vector<T>& vec)
{
    std::ifstream infile(filename);
    if(!infile)
        throw std::runtime_error("Could not open file: " + filename);
    
    T val;
    size_t idx = 0;
    while(infile >> val && idx < vec.size())
    {
        vec[idx++] = val;
    }
    
    if(idx != vec.size())
        throw std::runtime_error("File size mismatch for " + filename + 
                                ", expected " + std::to_string(vec.size()) + 
                                ", got " + std::to_string(idx));
}

template <typename Tgpu, typename Tref = Tgpu>
class CTCDriver : public Driver
{
public:
    CTCDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&probsDesc);
        miopenCreateTensorDescriptor(&gradientsDesc);

        miopenCreateCTCLossDescriptor(&ctcLossDesc);
        workspace_dev = nullptr;
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetProbabilityTensorLengthsFromCmdLine();

    int SetCTCLossDescriptorFromCmdLineArgs();
    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunBackwardGPU() override;
    int VerifyForward() override;
    int VerifyBackward() override;

    int RunCTCLossCPU();

    ~CTCDriver() override
    {
        miopenDestroyTensorDescriptor(probsDesc);
        miopenDestroyTensorDescriptor(gradientsDesc);

        miopenDestroyCTCLossDescriptor(ctcLossDesc);
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t probsDesc;
    miopenTensorDescriptor_t gradientsDesc;

    std::unique_ptr<GPUMem> probs_dev;
    std::unique_ptr<GPUMem> losses_dev;
    std::unique_ptr<GPUMem> gradients_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<Tgpu> probs;
    std::vector<int> labels;
    std::vector<int> labelLengths;
    std::vector<int> inputLengths;
    std::vector<Tgpu> losses;
    std::vector<Tgpu> gradients;
    std::vector<Tgpu> workspace;

    std::vector<Tref> losses_host;
    std::vector<Tref> gradients_host;
    std::vector<Tref> workspace_host;

    miopenCTCLossDescriptor_t ctcLossDesc;

    int batch_size;
    int max_time_step;
    int num_class;
    int blank_lb;
    int verbose;
    bool apply_softmax;
    miopenCTCLossAlgo_t ctc_algo;

    void _Log_Losses_And_Gradients();
};

template <typename Tgpu, typename Tref>
int CTCDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CTCDriver<Tgpu, Tref>::GetandSetData()
{
    verbose       = inflags.GetValueInt("verbose");
    batch_size    = inflags.GetValueInt("batchsize");
    num_class     = inflags.GetValueInt("num_class");

    inputLengths = std::vector<int>(batch_size, 0);
    labelLengths = std::vector<int>(batch_size, 0);

    std::string input_lengths_file = inflags.GetValueStr("input_lengths_file");
    if(input_lengths_file.empty())
        throw std::runtime_error("input_lengths_file is required. Please specify --input_lengths_file");
    LoadVectorFromFile(input_lengths_file, inputLengths);

    std::string label_lengths_file = inflags.GetValueStr("label_lengths_file");
    if(label_lengths_file.empty())
        throw std::runtime_error("label_lengths_file is required. Please specify --label_lengths_file");
    LoadVectorFromFile(label_lengths_file, labelLengths);

    max_time_step = *std::max_element(inputLengths.begin(), inputLengths.end());
    ctc_algo      = miopenCTCLossAlgo_t(inflags.GetValueInt("ctcalgo"));
    blank_lb      = inflags.GetValueInt("blank_label_id");
    apply_softmax = inflags.GetValueInt("apply_softmax_layer") == 1;

    std::vector<int> prob_dim = GetProbabilityTensorLengthsFromCmdLine();
    miopenSetTensorDescriptor(probsDesc, miopenFloat, 3, prob_dim.data(), nullptr);
    miopenSetTensorDescriptor(gradientsDesc, miopenFloat, 3, prob_dim.data(), nullptr);

    SetCTCLossDescriptorFromCmdLineArgs();

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CTCDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward CTC == 1 (Default=1)", "int");
    inflags.AddInputFlag(
        "apply_softmax_layer", 'm', "1", "Apply == 1, Not apply == 0 (Default=1)", "int");
    inflags.AddInputFlag("blank_label_id", 'b', "0", "Index of blank label (Default=0)", "int");
    inflags.AddInputFlag(
        "num_class", 'c', "9", "Number of classes without blank (Default=9)", "int");
    inflags.AddInputFlag("batchsize", 'n', "4", "Mini-batch size (Default=4)", "int");
    inflags.AddInputFlag("iter", 'i', "1", "Number of Iterations (Default=1)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify CTC losses and gradients (Default=1)", "int");
    inflags.AddInputFlag("verify_path",
                         'v',
                         "1",
                         "Verify Path for CTC losses and gradients: fast 1, regular 0 (Default=1)",
                         "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");
    inflags.AddInputFlag("dump_output", 'o', "0", "Dumps the output buffers (Default=0)", "int");
    inflags.AddInputFlag(
        "ctcalgo",
        'a',
        "0",
        "MIOPEN_CTC_LOSS_ALGO_DETERMINISTIC or MIOPEN_CTC_LOSS_ALGO_NON_DETERMINISTIC (Default=0)",
        "int");
    
    // Add file input flags
    inflags.AddInputFlag("probs_file", 'P', "", "Path to input probs file", "string");
    inflags.AddInputFlag("grads_file", 'G', "", "Path to input grads file", "string");
    inflags.AddInputFlag("labels_file", 'L', "", "Path to input labels file", "string");
    inflags.AddInputFlag("label_lengths_file", 'Q', "", "Path to input label lengths file", "string");
    inflags.AddInputFlag("input_lengths_file", 'R', "", "Path to input input lengths file", "string");

    inflags.AddInputFlag("verbose", 'k', "0", "Verbose GPU and CPU Gradients (Default=0)", "int");

    return 0;
}

template <typename Tgpu, typename Tref>
std::vector<int> CTCDriver<Tgpu, Tref>::GetProbabilityTensorLengthsFromCmdLine()
{
    std::vector<int> in_len = inputLengths;
    int batch_sz            = inflags.GetValueInt("batchsize");
    int class_sz            = inflags.GetValueInt("num_class") + 1;
    int time_step           = *std::max_element(in_len.begin(), in_len.end());

    return std::vector<int>({time_step, batch_sz, class_sz});
}

template <typename Tgpu, typename Tref>
int CTCDriver<Tgpu, Tref>::SetCTCLossDescriptorFromCmdLineArgs()
{
    miopenSetCTCLossDescriptor(ctcLossDesc, miopenFloat, blank_lb, apply_softmax);
    //  Framework implementation: To only get the frist two arguments, follow the example below:
    //  miopenDataType_t datatype;
    //  miopenGetCTCLossDescriptor(ctcLossDesc, &datatype, nullptr, nullptr);
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CTCDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{    
    size_t probs_sz  = batch_size * (num_class + 1) * max_time_step;
    size_t labels_sz = std::accumulate(labelLengths.begin(), labelLengths.end(), 0ULL);
    size_t workSpaceSize;
    size_t workSpaceSizeCPU;

    // Get file paths from command line flags - all files are required
    std::string probs_file = inflags.GetValueStr("probs_file");
    std::string grads_file = inflags.GetValueStr("grads_file");
    std::string labels_file = inflags.GetValueStr("labels_file");

    // Initialize vectors with proper sizes
    labels = std::vector<int>(labels_sz, 0);

    miopenGetCTCLossWorkspaceSize(GetHandle(),
                                  probsDesc,
                                  gradientsDesc,
                                  labels.data(),
                                  labelLengths.data(),
                                  inputLengths.data(),
                                  ctc_algo,
                                  ctcLossDesc,
                                  &workSpaceSize);

    GetCTCLossWorkspaceSizeCPU<Tgpu>(miopen::deref(probsDesc).GetLengths(),
                                     miopen::deref(gradientsDesc).GetLengths(),
                                     labels.data(),
                                     labelLengths.data(),
                                     inputLengths.data(),
                                     ctc_algo,
                                     &workSpaceSizeCPU);

    DEFINE_CONTEXT(ctx);
#if MIOPEN_BACKEND_OPENCL
    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#endif

    probs_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, probs_sz, sizeof(Tgpu)));
    losses_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, batch_size, sizeof(Tgpu)));
    gradients_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, probs_sz, sizeof(Tgpu)));
    workspace_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, workSpaceSize, sizeof(char)));

    probs          = std::vector<Tgpu>(probs_sz);
    losses         = std::vector<Tgpu>(batch_size, 0);
    losses_host    = std::vector<Tref>(batch_size, 0);
    gradients      = std::vector<Tgpu>(probs_sz, 0);
    gradients_host = std::vector<Tref>(probs_sz, 0);
    workspace      = std::vector<Tgpu>(workSpaceSize / sizeof(Tgpu), 0);
    workspace_host = std::vector<Tref>(workSpaceSizeCPU / sizeof(Tref), 0);

    // Load data from files - all files are required
    if(probs_file.empty())
        throw std::runtime_error("probs_file is required. Please specify --probs_file");
    LoadVectorFromFile(probs_file, probs);

    if(grads_file.empty())
        throw std::runtime_error("grads_file is required. Please specify --grads_file");
    LoadVectorFromFile(grads_file, gradients);

    if(labels_file.empty())
        throw std::runtime_error("labels_file is required. Please specify --labels_file");
    LoadVectorFromFile(labels_file, labels);

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile("dump_probs.bin", probs.data(), probs_sz);
        dumpBufferToFile("dump_labels.bin", labels.data(), labels_sz);
        dumpBufferToFile("dump_labelLengths.bin", labelLengths.data(), batch_size);
        dumpBufferToFile("dump_inputLengths.bin", inputLengths.data(), batch_size);
    }

    status_t status;
    status = probs_dev->ToGPU(q, probs.data());
    status |= losses_dev->ToGPU(q, losses.data());
    status |= gradients_dev->ToGPU(q, gradients.data());
    status |= workspace_dev->ToGPU(q, workspace.data());

    if(status != STATUS_SUCCESS)
        printf("Error copying data to GPU\n");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CTCDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME
    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenCTCLoss(GetHandle(),
                      probsDesc,
                      probs_dev->GetMem(),
                      labels.data(),
                      labelLengths.data(),
                      inputLengths.data(),
                      losses_dev->GetMem(),
                      gradientsDesc,
                      gradients_dev->GetMem(),
                      ctc_algo,
                      ctcLossDesc,
                      workspace_dev->GetMem(),
                      workspace_dev->GetSize());

        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);
        kernel_total_time += time;
        if(i == 0)
            kernel_first_time = time;
    }

    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        if(WALL_CLOCK)
            printf("Wall-clock Time CTC Loss Elapsed: %f ms\n",
                   t.gettime_ms() / inflags.GetValueInt("iter"));

        int iter = inflags.GetValueInt("iter");
        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        printf("GPU Kernel Time Forward Conv. Elapsed: %f ms (average)\n", kernel_average_time);
    }

    losses_dev->FromGPU(GetStream(), losses.data());
    gradients_dev->FromGPU(GetStream(), gradients.data());
    workspace_dev->FromGPU(GetStream(), workspace.data());

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile("dump_losses_gpu.bin", losses.data(), losses.size());
        dumpBufferToFile("dump_gradients_gpu.bin", gradients.data(), gradients.size());
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CTCDriver<Tgpu, Tref>::RunCTCLossCPU()
{
    RunCTCLossCPUVerify<Tgpu, Tref>(num_class,
                                    miopen::deref(probsDesc).GetLengths(),
                                    miopen::deref(probsDesc).GetStrides(),
                                    miopen::deref(gradientsDesc).GetLengths(),
                                    miopen::deref(gradientsDesc).GetStrides(),
                                    probs,
                                    labels,
                                    labelLengths,
                                    inputLengths,
                                    losses_host,
                                    gradients_host,
                                    workspace_host,
                                    blank_lb,
                                    apply_softmax,
                                    inflags.GetValueInt("verify_path"));

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile("dump_losses_cpu.bin", losses_host.data(), losses_host.size());
        dumpBufferToFile("dump_gradients_cpu.bin", gradients_host.data(), gradients_host.size());
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CTCDriver<Tgpu, Tref>::VerifyForward()
{
    {
        RunCTCLossCPU();
    }

    auto error1 = miopen::rms_range(losses_host, losses);
    auto error2 = miopen::rms_range(gradients_host, gradients);

    if(verbose) _Log_Losses_And_Gradients();

    const double tolerance1 = 1e-5;
    const double tolerance2 = 1e-3;
    if(!std::isfinite(error1) || error1 > tolerance1)
    {
        std::cout << std::string("CTC loss FAILED: ") << error1 << std::endl;
    }
    else
    {
        printf("CTC loss Verifies on CPU and GPU\n");
    }
    if(!std::isfinite(error2) || error2 > tolerance2)
    {
        std::cout << std::string("CTC gradient FAILED: ") << error2 << std::endl;
    }
    else
    {
        printf("CTC gradient Verifies on CPU and GPU\n");
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CTCDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CTCDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
void CTCDriver<Tgpu, Tref>::_Log_Losses_And_Gradients()
{
    std::cout << "Losses GPU: ";
    for(int i = 0; i < losses.size(); i++)
        std::cout << losses[i] << " ";
    std::cout << std::endl;
    std::cout << "Losses CPU: ";
    for(int i = 0; i < losses_host.size(); i++)
        std::cout << losses_host[i] << " ";
    std::cout << std::endl;

    std::cout << "Gradient GPU: ";
    for(auto v : gradients)
        std::cout << v << " ";
    std::cout << std::endl;
    std::cout << "Gradient CPU: ";
    for(auto v : gradients_host)
        std::cout << v << " ";
    std::cout << std::endl;
}
