// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <memory>
#include <vector>

#include <hipdnn_data_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/KnobWrapper.hpp>

namespace hipdnn_plugin_sdk
{

class IEngineDetails
{
public:
    virtual ~IEngineDetails() = default;

    virtual const hipdnn_data_sdk::data_objects::EngineDetails& getEngineDetails() const = 0;
    virtual bool isValid() const = 0;
    virtual int64_t engineId() const = 0;
    virtual uint32_t knobCount() const = 0;
    virtual const hipdnn_data_sdk::data_objects::Knob& getKnob(uint32_t index) const = 0;
    virtual const IKnob& getKnobWrapper(uint32_t index) const = 0;
    virtual const std::vector<std::unique_ptr<IKnob>>& knobWrappers() const = 0;
};

class EngineDetailsWrapper : public IEngineDetails
{
public:
    explicit EngineDetailsWrapper(const void* buffer, size_t size)
    {
        if(buffer != nullptr)
        {
            flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buffer), size);
            if(verifier.VerifyBuffer<hipdnn_data_sdk::data_objects::EngineDetails>())
            {
                _shallowEngineDetails
                    = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::EngineDetails>(buffer);
            }
        }
    }

    const hipdnn_data_sdk::data_objects::EngineDetails& getEngineDetails() const override
    {
        throwIfNotValid();
        return *_shallowEngineDetails;
    }

    bool isValid() const override
    {
        return _shallowEngineDetails != nullptr;
    }

    int64_t engineId() const override
    {
        throwIfNotValid();

        return _shallowEngineDetails->engine_id();
    }

    uint32_t knobCount() const override
    {
        throwIfNotValid();
        auto knobs = _shallowEngineDetails->knobs();
        if(knobs == nullptr)
        {
            return 0;
        }
        return static_cast<uint32_t>(knobs->size());
    }

    const hipdnn_data_sdk::data_objects::Knob& getKnob(uint32_t index) const override
    {
        throwIfNotValid();

        auto knobs = _shallowEngineDetails->knobs();
        if(knobs == nullptr)
        {
            throw std::out_of_range("No knobs in engine details");
        }

        if(index >= knobs->size())
        {
            throw std::out_of_range("Index out of range for engine knobs");
        }

        return *knobs->Get(index);
    }

    const IKnob& getKnobWrapper(uint32_t index) const override
    {
        throwIfNotValid();

        lazyInitKnobWrappers();

        if(index >= _knobWrappers.size())
        {
            throw std::out_of_range("Index out of range for engine knobs");
        }
        return *_knobWrappers[index];
    }

    const std::vector<std::unique_ptr<IKnob>>& knobWrappers() const override
    {
        lazyInitKnobWrappers();

        return _knobWrappers;
    }

private:
    void throwIfNotValid() const
    {
        if(!isValid())
        {
            throw std::invalid_argument("Engine details is not valid");
        }
    }

    void lazyInitKnobWrappers() const
    {
        if(_knobWrappers.empty())
        {
            auto knobs = _shallowEngineDetails->knobs();
            if(knobs == nullptr)
            {
                return; // No knobs to initialize
            }

            _knobWrappers.reserve(knobs->size());
            for(const auto knob : *knobs)
            {
                _knobWrappers.push_back(std::make_unique<KnobWrapper>(knob));
            }
        }
    }

    // Pointer to the flatbuffer representation of the engine details. We do not own this memory
    // as were just reading from the buffer passed during construction.
    const hipdnn_data_sdk::data_objects::EngineDetails* _shallowEngineDetails = nullptr;

    // Lazy-initialized state for knob wrappers
    mutable std::vector<std::unique_ptr<IKnob>> _knobWrappers;
};

}
