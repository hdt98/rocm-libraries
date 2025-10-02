#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/Utilities/LazySingleton.hpp>
#include <rocRoller/Utilities/Settings.hpp>

namespace rocRoller
{
// Macro for explicit template instantiation
#define INSTANTIATE_LAZY_SINGLETON(Type) template class LazySingleton<Type>;

    // Explicit instantiations for non-templated singleton classes
    INSTANTIATE_LAZY_SINGLETON(GPUArchitectureLibrary)
    INSTANTIATE_LAZY_SINGLETON(Settings)
    INSTANTIATE_LAZY_SINGLETON(Data)

// Undefine to avoid leaking the macro outside this file
#undef INSTANTIATE_LAZY_SINGLETON
}