# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(rocroller-private-interface
    INTERFACE
    FILE_SET HEADERS
    BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/AllocatingObserver.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/FileWritingObserver.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/RegisterLivenessObserver.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/SupportedInstructionObserver.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/WaitcntObserver.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/WaitcntObserver_impl.hpp"

        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/FunctionalUnit/MEMObserver.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/FunctionalUnit/MEMObserver_impl.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/FunctionalUnit/MFMAObserver.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/FunctionalUnit/WMMAObserver.hpp"

        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/WaitState/BufferStoreDwordXXRead.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/WaitState/OPSEL94x.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/WaitState/VALUTransWrite94x.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/WaitState/VALUWriteReadlane94x.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/WaitState/VALUWriteSGPRVCC.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/WaitState/VALUWriteSGPRVMEM.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/WaitState/VALUWriteVCCVDIVFMAS.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/WaitState/VCMPXWrite94x.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/WaitState/WaitStateObserver_impl.hpp"

        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/WaitState/MFMA/ACCVGPRReadWrite.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/WaitState/MFMA/ACCVGPRWriteWrite.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/WaitState/MFMA/CMPXWriteExec.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/WaitState/MFMA/DGEMM16x16x4Write.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/WaitState/MFMA/DGEMM4x4x4Write.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/WaitState/MFMA/DLWrite.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/WaitState/MFMA/VALUWrite.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/WaitState/MFMA/XDLReadSrcC908.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/WaitState/MFMA/XDLReadSrcC90a.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/WaitState/MFMA/XDLReadSrcC94x.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/WaitState/MFMA/XDLWrite908.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/WaitState/MFMA/XDLWrite90a.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/WaitState/MFMA/XDLWrite94x.hpp"

        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/WaitState/WMMA/WMMAReadSrcD.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/WaitState/WMMA/WMMAWrite.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocRoller/Scheduling/Observers/WaitState/WMMA/WMMAWriteSrcD.hpp"
)
