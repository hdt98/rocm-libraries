// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>

#include <rocRoller/KernelGraph/ControlToCoordinateMapper.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller::KernelGraph
{
    template <typename T>
    inline void ControlToCoordinateMapper::connect(int control, int coordinate, int subDimension)
    {
        connect(control, coordinate, Connections::TypeAndSubDimension{name<T>(), subDimension});
    }

    template <typename T>
    inline void ControlToCoordinateMapper::disconnect(int control, int coordinate, int subDimension)
    {
        disconnect(control, coordinate, Connections::TypeAndSubDimension{name<T>(), subDimension});
    }

    template <typename T>
    inline int ControlToCoordinateMapper::get(int control, int subDimension) const
    {
        return get(control, Connections::TypeAndSubDimension{name<T>(), subDimension});
    }

    inline int ControlToCoordinateMapper::get(int control, Connections::ConnectionSpec conn) const
    {
        auto iter = m_map.find(control);

        if(iter == m_map.end() or iter->second.find(conn) == iter->second.end())
            return -1;

        return iter->second.at(conn);
    }

    inline int ControlToCoordinateMapper::get(int control, NaryArgument arg) const
    {
        return get(control, Connections::JustNaryArgument{arg});
    }

    /**
     * @brief Connect a control node to a coordinate using a WaveGroupBranch specifier.
     *
     * Used by MergeConditionalLoads. waveGroup=0 for A-side, waveGroup=1 for B-side.
     */
    template <typename T>
    inline void ControlToCoordinateMapper::connectWaveGroup(int control,
                                                            int coordinate,
                                                            int waveGroup)
    {
        connect(control, coordinate, Connections::WaveGroupBranch{name<T>(), waveGroup});
    }

    /**
     * @brief Retrieve the coordinate connected via WaveGroupBranch for the given wave group.
     *
     * Returns -1 if no such connection exists (non-merged op).
     * Presence of waveGroup=0 signals a merged (conditional) load.
     */
    template <typename T>
    inline int ControlToCoordinateMapper::getWaveGroup(int control, int waveGroup) const
    {
        return get(control, Connections::WaveGroupBranch{name<T>(), waveGroup});
    }
}
