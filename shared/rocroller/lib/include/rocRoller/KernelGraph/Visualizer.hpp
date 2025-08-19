#pragma once

#include <map>

#include <rocRoller/Utilities/Logging.hpp>

#include <rocRoller/KernelGraph/KernelGraph_fwd.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        void visualize(KernelGraph const&           graph,
                       std::string const&           windowName      = "rocRoller Graph",
                       UnrollColouring const&       unrollColouring = {},
                       NaryArgumentColouring const& naryColouring   = {},
                       std::string const&           asmFileName     = "");
    }
}
