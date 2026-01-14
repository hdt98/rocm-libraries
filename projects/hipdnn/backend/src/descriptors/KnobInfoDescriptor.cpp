// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "KnobInfoDescriptor.hpp"
#include "BackendEnumStringUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include <cstring>

namespace hipdnn_backend
{

void KnobInfoDescriptor::finalize()
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "KnobInfoDescriptor::finalize() failed: Already finalized.");

    THROW_IF_FALSE(_knobDataSet,
                   HIPDNN_STATUS_BAD_PARAM,
                   "KnobInfoDescriptor::finalize() failed: Knob data is not set.");

    HipdnnBackendDescriptorImpl<KnobInfoDescriptor>::finalize();
}

void KnobInfoDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                      hipdnnBackendAttributeType_t attributeType,
                                      int64_t requestedElementCount,
                                      int64_t* elementCount,
                                      void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "KnobInfoDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT:
        THROW_IF_NE(attributeType,
                    HIPDNN_TYPE_VOID_PTR,
                    HIPDNN_STATUS_BAD_PARAM,
                    "KnobInfoDescriptor failed to get serialized knob: Invalid attribute type.");

        THROW_IF_NE(requestedElementCount,
                    1,
                    HIPDNN_STATUS_BAD_PARAM,
                    "KnobInfoDescriptor failed to get serialized knob: Invalid element count.");

        THROW_IF_NULL(arrayOfElements,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "KnobInfoDescriptor failed to get serialized knob: Null pointer.");

        if(elementCount != nullptr)
        {
            *elementCount = 1;
        }

        // Copy the serialized data to a new buffer for the caller
        {
            auto dataSize = _knobSerializedBuffer.size();
            auto* buffer = new uint8_t[dataSize];
            std::memcpy(buffer, _knobSerializedBuffer.data(), dataSize);
            *static_cast<void**>(arrayOfElements) = buffer;
        }
        break;

    case HIPDNN_ATTR_KNOB_INFO_TYPE:
    case HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE:
    case HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE:
    case HIPDNN_ATTR_KNOB_INFO_STRIDE:
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("KnobInfoDescriptor::getAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }
}

void KnobInfoDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                      [[maybe_unused]] hipdnnBackendAttributeType_t attributeType,
                                      [[maybe_unused]] int64_t elementCount,
                                      [[maybe_unused]] const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "KnobInfoDescriptor::setAttribute() failed: Already finalized.");

    // KnobInfoDescriptor doesn't support setting attributes directly
    // The knob data is set through the setKnobData() method
    throw HipdnnException(
        HIPDNN_STATUS_NOT_SUPPORTED,
        std::string("KnobInfoDescriptor::setAttribute() is not supported for attribute ")
            + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
}

hipdnnBackendDescriptorType_t KnobInfoDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_KNOB_INFO_DESCRIPTOR;
}

void KnobInfoDescriptor::setKnobData(const hipdnn_data_sdk::data_objects::Knob* knob)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "KnobInfoDescriptor::setKnobData() failed: Already finalized.");

    THROW_IF_NULL(knob,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "KnobInfoDescriptor::setKnobData() failed: Knob is null.");

    // Serialize the knob data by unpacking and repacking it
    // This creates a standalone copy that we own
    flatbuffers::FlatBufferBuilder builder;

    // First, unpack the knob to get a native object
    auto knobT = knob->UnPack();

    // Then pack it into our own buffer
    builder.Finish(hipdnn_data_sdk::data_objects::Knob::Pack(builder, knobT));

    _knobSerializedBuffer = builder.Release();
    _knobDataSet = true;
}

std::string KnobInfoDescriptor::toString() const
{
    std::string str = "KnobInfoDescriptor: {";
    str += "knobDataSet=" + std::string(_knobDataSet ? "true" : "false");
    str += ", bufferSize=" + std::to_string(_knobSerializedBuffer.size());
    str += "}";
    return str;
}

} // namespace hipdnn_backend
