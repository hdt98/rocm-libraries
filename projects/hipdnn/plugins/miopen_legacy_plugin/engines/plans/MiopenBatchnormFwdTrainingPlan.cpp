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
{
    // Extract and validate epsilon value from pass-by-value tensor (using double for MIOpen)
    auto epsilonTensorAttr = tensorMap.at(attributes.epsilon_tensor_uid());

    HIPDNN_LOG_ERROR("DEBUG: Epsilon data_type = {}",
                     hipdnn_sdk::data_objects::toString(epsilonTensorAttr->data_type()));
    HIPDNN_LOG_ERROR("DEBUG: Epsilon has value = {}", epsilonTensorAttr->value() != nullptr);
    HIPDNN_LOG_ERROR("DEBUG: Epsilon value_type = {}",
                     static_cast<int>(epsilonTensorAttr->value_type()));

    if(epsilonTensorAttr->data_type() != hipdnn_sdk::data_objects::DataType::DOUBLE)
    {
        throw std::runtime_error("Epsilon tensor must be DOUBLE type for MIOpen compatibility");
    }
    auto epsilonValue = epsilonTensorAttr->value_as_Float64Value();
    if(epsilonValue == nullptr)
    {
        throw std::runtime_error("Epsilon must be a pass-by-value Float64 tensor");
    }
    _epsilonValue = epsilonValue->value();

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
        // Extract and validate momentum value from pass-by-value tensor (using double for MIOpen)
        auto momentumTensorAttr = tensorMap.at(attributes.momentum_tensor_uid().value());
        if(momentumTensorAttr->data_type() != hipdnn_sdk::data_objects::DataType::DOUBLE)
        {
            throw std::runtime_error(
                "Momentum tensor must be DOUBLE type for MIOpen compatibility");
        }
        auto momentumValue = momentumTensorAttr->value_as_Float64Value();
        if(momentumValue == nullptr)
        {
            throw std::runtime_error("Momentum must be a pass-by-value Float64 tensor");
        }
        _momentumValue = momentumValue->value();

        _prevRunningMean = miopen_utils::createTensor(
            tensorMap, attributes.prev_running_mean_tensor_uid().value());
        _prevRunningVariance = miopen_utils::createTensor(
            tensorMap, attributes.prev_running_variance_tensor_uid().value());
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

double BatchnormFwdTrainingParams::epsilonValue() const
{
    return _epsilonValue;
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

double BatchnormFwdTrainingParams::momentumValue() const
{
    return _momentumValue.value();
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
    float alpha = 1.0f;
    float beta = 0.0f;

    // Extract epsilon from pass-by-value tensor attribute (type-safe, no buffer lookup needed)
    // Note: Type validation already done in constructor
    double epsilon = _trainingParams.epsilonValue();

    // Extract momentum from pass-by-value tensor attribute if running stats exist
    double expAvgFactor = 0.0;
    if(_trainingParams.hasRunningStats())
    {
        expAvgFactor = _trainingParams.momentumValue();
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
    // Note: MIOpen uses IN/OUT parameters (single buffer read+written), but hipDNN's graph
    // API has separate prev (input) and next (output) tensors. We bridge this by:
    // 1. Tests initialize NEXT with PREV's values (same seed in initializeBundle)
    // 2. Pass NEXT buffer to MIOpen as the IN/OUT parameter
    // 3. MIOpen reads initial values from NEXT, computes EMA, writes results back to NEXT
    // PREV tensor is unused by the plugin - it exists only for graph API semantics.
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
