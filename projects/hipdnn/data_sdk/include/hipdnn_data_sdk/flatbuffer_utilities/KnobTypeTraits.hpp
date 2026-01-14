// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/data_objects/knob_value_generated.h>

namespace hipdnn_plugin_sdk
{

namespace detail
{
// Maps a KnobValue type to its enum value at compile time
template <typename T>
struct KnobValueTypeTraits;

template <>
struct KnobValueTypeTraits<hipdnn_data_sdk::data_objects::IntValue>
{
    static constexpr auto VALUE = hipdnn_data_sdk::data_objects::KnobValue::IntValue;
};

template <>
struct KnobValueTypeTraits<hipdnn_data_sdk::data_objects::FloatValue>
{
    static constexpr auto VALUE = hipdnn_data_sdk::data_objects::KnobValue::FloatValue;
};

template <>
struct KnobValueTypeTraits<hipdnn_data_sdk::data_objects::StringValue>
{
    static constexpr auto VALUE = hipdnn_data_sdk::data_objects::KnobValue::StringValue;
};

// Maps a KnobConstraint type to its enum value at compile time
template <typename T>
struct KnobConstraintTypeTraits;

template <>
struct KnobConstraintTypeTraits<hipdnn_data_sdk::data_objects::IntConstraint>
{
    static constexpr auto VALUE = hipdnn_data_sdk::data_objects::KnobConstraint::IntConstraint;
};

template <>
struct KnobConstraintTypeTraits<hipdnn_data_sdk::data_objects::FloatConstraint>
{
    static constexpr auto VALUE = hipdnn_data_sdk::data_objects::KnobConstraint::FloatConstraint;
};

template <>
struct KnobConstraintTypeTraits<hipdnn_data_sdk::data_objects::StringConstraint>
{
    static constexpr auto VALUE = hipdnn_data_sdk::data_objects::KnobConstraint::StringConstraint;
};
} // namespace detail

} // namespace hipdnn_plugin_sdk
