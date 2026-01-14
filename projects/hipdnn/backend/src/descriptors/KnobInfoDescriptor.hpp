// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "BackendDescriptor.hpp"
#include <flatbuffers/detached_buffer.h>
#include <hipdnn_data_sdk/data_objects/knob_value_generated.h>
#include <memory>

namespace hipdnn_backend
{

class KnobInfoDescriptor : public HipdnnBackendDescriptorImpl<KnobInfoDescriptor>
{
private:
    flatbuffers::DetachedBuffer _knobSerializedBuffer;
    bool _knobDataSet = false;

public:
    KnobInfoDescriptor() = default;

    void finalize() override;

    void getAttribute(hipdnnBackendAttributeName_t attributeName,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t requestedElementCount,
                      int64_t* elementCount,
                      void* arrayOfElements) const override;

    void setAttribute(hipdnnBackendAttributeName_t attributeName,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t elementCount,
                      const void* arrayOfElements) override;

    static hipdnnBackendDescriptorType_t getStaticType();

    // Set the knob data from an existing Knob flatbuffer
    void setKnobData(const hipdnn_data_sdk::data_objects::Knob* knob);

    std::string toString() const override;
};

} // namespace hipdnn_backend
