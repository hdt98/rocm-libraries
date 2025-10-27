// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "MiopenBatchnormFwdTrainingPlan.hpp"
#include "MiopenUtils.hpp"

namespace miopen_legacy_plugin
{

// We have made the intentional decision to hardcode the batchnorm mode to miopenBNSpatial
// rather than making it configurable and adding extra complexity.
const miopenBatchNormMode_t MIOPEN_BATCHNORM_MODE_TRAINING = miopenBNSpatial;

BatchnormFwdTrainingParams::BatchnormFwdTrainingParams(
    const hipdnn_sdk::data_objects::BatchnormAttributes& attributes,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap)
    : _x(miopen_utils::createTensor(tensorMap, attributes.x_tensor_uid()))
    , _y(miopen_utils::createTensor(tensorMap, attributes.y_tensor_uid()))
    , _scale(miopen_utils::createTensor(tensorMap, attributes.scale_tensor_uid()))
    , _bias(miopen_utils::createTensor(tensorMap, attributes.bias_tensor_uid()))
    , _epsilon(miopen_utils::createTensor(tensorMap, attributes.epsilon_tensor_uid()))
{
    // Save mean and inv_variance are optional (controlled by MIO_SAVE_MEAN_VARIANCE)
    if(attributes.mean_tensor_uid().has_value())
    {
        _mean = miopen_utils::createTensor(tensorMap, attributes.mean_tensor_uid().value());
    }

    if(attributes.inv_variance_tensor_uid().has_value())
    {
        _invVariance
            = miopen_utils::createTensor(tensorMap, attributes.inv_variance_tensor_uid().value());
    }

    // Check if running statistics are provided
    if(attributes.prev_running_mean_tensor_uid().has_value()
       && attributes.prev_running_variance_tensor_uid().has_value()
       && attributes.momentum_tensor_uid().has_value()
       && attributes.next_running_mean_tensor_uid().has_value()
       && attributes.next_running_variance_tensor_uid().has_value())
    {
        _prevRunningMean = miopen_utils::createTensor(
            tensorMap, attributes.prev_running_mean_tensor_uid().value());
        _prevRunningVariance = miopen_utils::createTensor(
            tensorMap, attributes.prev_running_variance_tensor_uid().value());
        _momentum = miopen_utils::createTensor(tensorMap, attributes.momentum_tensor_uid().value());
        _nextRunningMean = miopen_utils::createTensor(
            tensorMap, attributes.next_running_mean_tensor_uid().value());
        _nextRunningVariance = miopen_utils::createTensor(
            tensorMap, attributes.next_running_variance_tensor_uid().value());
        _hasRunningStats = true;
    }
}

const MiopenTensor& BatchnormFwdTrainingParams::x() const
{
    return _x;
}

const MiopenTensor& BatchnormFwdTrainingParams::y() const
{
    return _y;
}

const MiopenTensor& BatchnormFwdTrainingParams::scale() const
{
    return _scale;
}

const MiopenTensor& BatchnormFwdTrainingParams::bias() const
{
    return _bias;
}

const MiopenTensor& BatchnormFwdTrainingParams::epsilon() const
{
    return _epsilon;
}

bool BatchnormFwdTrainingParams::hasSaveMeanVariance() const
{
    return _mean.has_value() && _invVariance.has_value();
}

const MiopenTensor& BatchnormFwdTrainingParams::mean() const
{
    return _mean.value();
}

const MiopenTensor& BatchnormFwdTrainingParams::invVariance() const
{
    return _invVariance.value();
}

bool BatchnormFwdTrainingParams::hasRunningStats() const
{
    return _hasRunningStats;
}

const MiopenTensor& BatchnormFwdTrainingParams::prevRunningMean() const
{
    return _prevRunningMean.value();
}

const MiopenTensor& BatchnormFwdTrainingParams::prevRunningVariance() const
{
    return _prevRunningVariance.value();
}

const MiopenTensor& BatchnormFwdTrainingParams::momentum() const
{
    return _momentum.value();
}

const MiopenTensor& BatchnormFwdTrainingParams::nextRunningMean() const
{
    return _nextRunningMean.value();
}

const MiopenTensor& BatchnormFwdTrainingParams::nextRunningVariance() const
{
    return _nextRunningVariance.value();
}

BatchnormFwdTrainingPlan::BatchnormFwdTrainingPlan(BatchnormFwdTrainingParams&& trainingParams)
    : _trainingParams(std::move(trainingParams))
{
}

size_t BatchnormFwdTrainingPlan::getWorkspaceSize(
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle) const
{
    // No workspace needed for batchnorm training
    return 0;
}

void BatchnormFwdTrainingPlan::execute(const HipdnnEnginePluginHandle& handle,
                                       const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                       uint32_t numDeviceBuffers,
                                       [[maybe_unused]] void* workspace) const
{
    // Hardcoded values consistent with the pattern used in inference
    auto alpha = static_cast<float>(1);
    auto beta = static_cast<float>(0);

    // Get epsilon from device buffer (points to host memory for scalar pass-by-value tensors)
    auto epsilonBuffer = miopen_utils::findDeviceBuffer(
        _trainingParams.epsilon().uid(), deviceBuffers, numDeviceBuffers);
    auto epsilon = static_cast<double>(*static_cast<const float*>(epsilonBuffer.ptr));

    // Get momentum if running stats exist
    double expAvgFactor;
    if(_trainingParams.hasRunningStats())
    {
        auto momentumBuffer = miopen_utils::findDeviceBuffer(
            _trainingParams.momentum().uid(), deviceBuffers, numDeviceBuffers);
        expAvgFactor = static_cast<double>(*static_cast<const float*>(momentumBuffer.ptr));
    }
    else
    {
        expAvgFactor = 0.0;
    }

    // Get all required device buffers
    auto xBuffer = miopen_utils::findDeviceBuffer(
        _trainingParams.x().uid(), deviceBuffers, numDeviceBuffers);
    auto yBuffer = miopen_utils::findDeviceBuffer(
        _trainingParams.y().uid(), deviceBuffers, numDeviceBuffers);
    auto scaleBuffer = miopen_utils::findDeviceBuffer(
        _trainingParams.scale().uid(), deviceBuffers, numDeviceBuffers);
    auto biasBuffer = miopen_utils::findDeviceBuffer(
        _trainingParams.bias().uid(), deviceBuffers, numDeviceBuffers);

    // Handle save mean/variance if provided (optional)
    void* resultSaveMeanPtr = nullptr;
    void* resultSaveInvVariancePtr = nullptr;

    if(_trainingParams.hasSaveMeanVariance())
    {
        auto meanBuffer = miopen_utils::findDeviceBuffer(
            _trainingParams.mean().uid(), deviceBuffers, numDeviceBuffers);
        auto invVarianceBuffer = miopen_utils::findDeviceBuffer(
            _trainingParams.invVariance().uid(), deviceBuffers, numDeviceBuffers);

        resultSaveMeanPtr = meanBuffer.ptr;
        resultSaveInvVariancePtr = invVarianceBuffer.ptr;
    }

    // Handle running statistics if provided
    void* resultRunningMeanPtr = nullptr;
    void* resultRunningVariancePtr = nullptr;

    if(_trainingParams.hasRunningStats())
    {
        auto nextRunningMeanBuffer = miopen_utils::findDeviceBuffer(
            _trainingParams.nextRunningMean().uid(), deviceBuffers, numDeviceBuffers);
        auto nextRunningVarianceBuffer = miopen_utils::findDeviceBuffer(
            _trainingParams.nextRunningVariance().uid(), deviceBuffers, numDeviceBuffers);

        resultRunningMeanPtr = nextRunningMeanBuffer.ptr;
        resultRunningVariancePtr = nextRunningVarianceBuffer.ptr;
    }

    THROW_ON_MIOPEN_FAILURE(
        miopenBatchNormalizationForwardTraining(handle.miopenHandle,
                                                MIOPEN_BATCHNORM_MODE_TRAINING,
                                                &alpha,
                                                &beta,
                                                _trainingParams.x().tensorDescriptor(),
                                                xBuffer.ptr,
                                                _trainingParams.y().tensorDescriptor(),
                                                yBuffer.ptr,
                                                _trainingParams.scale().tensorDescriptor(),
                                                scaleBuffer.ptr,
                                                biasBuffer.ptr,
                                                expAvgFactor,
                                                resultRunningMeanPtr,
                                                resultRunningVariancePtr,
                                                epsilon,
                                                resultSaveMeanPtr,
                                                resultSaveInvVariancePtr));
}

}
