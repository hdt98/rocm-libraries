// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <type_traits>

#include <miopen/ck_builder/factories/device_operation_instance_factory.hpp>

// The CK (old) factory header is not included here. The caller must include the
// appropriate CK factory dispatch header (e.g.,
// <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp>)
// before instantiating MetaDeviceOperationInstanceFactory.

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

inline bool ShouldUseNewFactory() { return true; }

template <typename T>
concept BoolOrBoolCallable =
    std::same_as<T, bool> || (std::invocable<T> && std::same_as<std::invoke_result_t<T>, bool>);

/// MetaDeviceOperationInstanceFactory is a switchable factory for CK kernel instances.
///
/// It can return instances from either the old (CK) factory or the new (CK Builder) factory
/// depending on the template parameter UseNewFactoryCheck:
///
/// - A callable returning bool (default: ShouldUseNewFactory) - both factories are compiled,
///   selection happens at runtime. Useful for A/B testing via environment variables.
/// - A compile-time bool (true/false) - only the selected factory is compiled.
///
/// The static GetInstances() function on both OldFactory and NewFactory must have matching
/// return types. In CK this is accomplished by returning a vector of unique_ptrs to a common
/// base class.
///
/// Usage:
///   // Runtime switch (default - both factories compiled):
///   using Factory = MetaDeviceOperationInstanceFactory<DeviceOp>;
///
///   // Compile-time switch to new factory only:
///   using Factory = MetaDeviceOperationInstanceFactory<DeviceOp, true>;
///
///   // Compile-time switch to old factory only:
///   using Factory = MetaDeviceOperationInstanceFactory<DeviceOp, false>;
///
///   // Custom runtime check:
///   using Factory = MetaDeviceOperationInstanceFactory<DeviceOp, MyCustomCheck>;
template <typename DeviceOp,
          BoolOrBoolCallable auto UseNewFactoryCheck =

#ifdef CK_EXPERIMENTAL_BUILDER
              ShouldUseNewFactory
#else
              false
#endif

          >
struct MetaDeviceOperationInstanceFactory
{
    using NewFactory = miopen::conv::ck_builder::instance::DeviceOperationInstanceFactory<DeviceOp>;
    using OldFactory =
        ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<DeviceOp>;

    static auto GetInstances()
    {
        if constexpr(std::is_invocable_v<decltype(UseNewFactoryCheck)>)
        {
            // Runtime dispatch - both factories are compiled
            if(UseNewFactoryCheck())
            {
                return NewFactory::GetInstances();
            }
            else
            {
                return OldFactory::GetInstances();
            }
        }
        else
        {
            // Compile-time dispatch - only the selected factory is compiled
            if constexpr(UseNewFactoryCheck)
            {
                return NewFactory::GetInstances();
            }
            else
            {
                return OldFactory::GetInstances();
            }
        }
    }
};

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
