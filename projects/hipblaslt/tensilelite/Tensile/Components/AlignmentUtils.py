################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""
Memory Alignment Optimization Utilities

This module provides utility functions and classes for implementing memory
alignment optimizations to improve cache line utilization and overall
performance for misaligned strides.

Three main techniques are provided:
1. Address Interleaving - Remap row indices for better alignment
2. LDS Re-alignment - Fix alignment using Local Data Share with multi-buffering
3. Row Re-alignment - Group rows by alignment pattern
"""

from rocisa.code import Module
from rocisa.container import vgpr, sgpr
from rocisa.instruction import (
    VMovB32, VMulLOU32, VAddU32, VSubU32, VLShiftRightB32,
    VMulHiU32, VAndB32, VOrB32,
    SMovB32, SMulI32, SAddU32, SAddCU32, SLShiftLeftB32,
    SLShiftRightB32, SAndB32
)
from math import log2, ceil


class AddressInterleaveCalculator:
    """
    Calculates interleaved addresses for better cache line alignment.
    
    Instead of linear addressing: J = Rn + Kn * Sn
    Uses interleaved addressing: J = Rn * tileSize + Kn
    
    This spreads consecutive tiles across different cache lines, improving
    alignment when stride is not a multiple of cache line size.
    """
    
    def __init__(self, kernel, tP):
        """
        Initialize address interleave calculator.
        
        Args:
            kernel: Kernel configuration dictionary
            tP: Tensor parameters (A or B)
        """
        self.kernel = kernel
        self.tP = tP
        self.tc = tP["tensorChar"]
        self.tileSize = kernel.get("AddressInterleaveTileSize", 16)
        self.alignment = kernel.get("AddressInterleaveAlignment", 256)
        self.direction = kernel.get("AddressInterleaveDirection", "N")
        
    def isEnabled(self):
        """Check if address interleaving is enabled for this tensor."""
        if not self.kernel.get("EnableAddressInterleave", False):
            return False
        
        # Only apply to the specified direction
        isCorrectDirection = (
            (self.direction == "N" and not self.tP["isA"]) or
            (self.direction == "M" and self.tP["isA"])
        )
        
        return isCorrectDirection
    
    def computeInterleavedRowIndex(self, rnVgpr, knVgpr, dstVgpr):
        """
        Compute interleaved row index: J = Rn * tileSize + Kn
        
        Args:
            rnVgpr: VGPR containing relative row number within tile (Rn)
            knVgpr: VGPR containing kernel coordinate in dimension (Kn)
            dstVgpr: Destination VGPR for result
            
        Returns:
            Module with generated instructions
        """
        module = Module("ComputeInterleavedRowIndex%s" % self.tc)
        
        if not self.isEnabled():
            # Just copy Rn to dst if not enabled
            module.add(VMovB32(dst=vgpr(dstVgpr), src=vgpr(rnVgpr),
                              comment="No interleaving, J = Rn"))
            return module
        
        module.addComment0("Address Interleave: J = Rn * %d + Kn" % self.tileSize)
        
        # tmp = Rn * tileSize
        tmpVgpr = dstVgpr  # Can reuse destination as temp
        if self.tileSize == 16:
            # Optimize for power of 2: left shift by 4
            module.add(VLShiftLeftB32(
                dst=vgpr(tmpVgpr),
                src=vgpr(rnVgpr),
                shiftHex=hex(4),
                comment="Rn * 16 via left shift"))
        else:
            module.add(VMulLOU32(
                dst=vgpr(tmpVgpr),
                src0=vgpr(rnVgpr),
                src1=self.tileSize,
                comment="Rn * %d" % self.tileSize))
        
        # J = tmp + Kn
        module.add(VAddU32(
            dst=vgpr(dstVgpr),
            src0=vgpr(tmpVgpr),
            src1=vgpr(knVgpr),
            comment="J = Rn * tileSize + Kn"))
        
        return module
    
    def computeAlignedKStart(self, rowVgpr, strideVgpr, dstVgpr, tmpVgprBase):
        """
        Calculate aligned K starting position: K_start = ceil((J * stride) / alignment) * alignment
        
        Args:
            rowVgpr: VGPR containing row index J
            strideVgpr: SGPR containing stride value
            dstVgpr: Destination VGPR for aligned K start
            tmpVgprBase: Base VGPR for temporary calculations (needs 3 VGPRs)
            
        Returns:
            Module with generated instructions
        """
        module = Module("ComputeAlignedKStart%s" % self.tc)
        
        if not self.isEnabled():
            # Return 0 if not enabled
            module.add(VMovB32(dst=vgpr(dstVgpr), src=0,
                              comment="No alignment, K_start = 0"))
            return module
        
        module.addComment0("Compute aligned K start: ceil((J * stride) / %d) * %d" %
                          (self.alignment, self.alignment))
        
        tmp0 = tmpVgprBase
        tmp1 = tmpVgprBase + 1
        tmp2 = tmpVgprBase + 2
        
        # tmp = J * stride (64-bit result)
        # Low 32 bits
        module.add(VMulLOU32(
            dst=vgpr(tmp0),
            src0=vgpr(rowVgpr),
            src1=sgpr(strideVgpr),
            comment="(J * stride) low"))
        
        # High 32 bits
        module.add(VMulHiU32(
            dst=vgpr(tmp1),
            src0=vgpr(rowVgpr),
            src1=sgpr(strideVgpr),
            comment="(J * stride) high"))
        
        # Calculate alignment_mask = alignment - 1
        alignmentMask = self.alignment - 1
        
        # Round up: add (alignment - 1) before masking
        module.add(VAddU32(
            dst=vgpr(tmp0),
            src0=vgpr(tmp0),
            src1=alignmentMask,
            comment="Add alignment-1 for ceiling"))
        
        # Mask off lower bits to get aligned address
        alignmentBits = int(log2(self.alignment))
        
        # Shift right then left to clear lower bits
        # Or use AND with inverted mask
        inverseMask = ~alignmentMask & 0xFFFFFFFF
        
        module.add(VAndB32(
            dst=vgpr(dstVgpr),
            src0=vgpr(tmp0),
            src1=inverseMask,
            comment="Mask to alignment boundary"))
        
        return module


class LDSAlignmentManager:
    """
    Manages LDS (Local Data Share) alignment with multi-buffering.
    
    Calculates per-row offsets to handle misaligned strides by:
    1. Computing aligned read pointers: read_ptr[row] = (stride * row) & ~(KERNEL_K - 1)
    2. Computing LDS offsets: lds_offset[row] = stride * row - read_ptr[row]
    3. Managing multi-buffer rotation for overlapped read/compute/write
    """
    
    def __init__(self, kernel, tP):
        """
        Initialize LDS alignment manager.
        
        Args:
            kernel: Kernel configuration dictionary
            tP: Tensor parameters (A or B)
        """
        self.kernel = kernel
        self.tP = tP
        self.tc = tP["tensorChar"]
        self.numBuffers = kernel.get("LDSAlignmentBuffers", 2)
        self.alignment = kernel.get("LDSAlignmentTarget", 128)
        self.kernelK = kernel.get("LDSAlignmentKernelK", 64)
        self.bpe = tP["bpe"]  # bytes per element
        
    def isEnabled(self):
        """Check if LDS alignment is enabled."""
        return self.kernel.get("EnableLDSAlignment", False)
    
    def calculateLDSSize(self, baseSize):
        """
        Calculate LDS size needed for alignment with multi-buffering.
        
        Args:
            baseSize: Base LDS size without alignment
            
        Returns:
            Required LDS size with alignment buffers
        """
        if not self.isEnabled():
            return baseSize
        
        # Each row needs KERNEL_K * NUM_BUFFERS * bpe bytes
        perRowSize = self.kernelK * self.numBuffers * self.bpe
        
        # Total depends on number of rows being processed
        return perRowSize
    
    def computeReadPointer(self, rowIdx, strideVgpr, dstVgpr, tmpVgprBase):
        """
        Compute aligned read pointer for a row.
        read_pointer[row] = (stride * row) & ~(KERNEL_K - 1)
        
        Args:
            rowIdx: VGPR or immediate containing row index
            strideVgpr: SGPR containing stride
            dstVgpr: Destination VGPR for read pointer
            tmpVgprBase: Base VGPR for temporaries (needs 2)
            
        Returns:
            Module with generated instructions
        """
        module = Module("ComputeReadPointer%s_Row" % self.tc)
        
        if not self.isEnabled():
            return module
        
        module.addComment0("Compute read pointer: (stride * row) & ~(KERNEL_K - 1)")
        
        tmp0 = tmpVgprBase
        tmp1 = tmpVgprBase + 1
        
        # tmp = stride * row
        if isinstance(rowIdx, int):
            module.add(VMulLOU32(
                dst=vgpr(tmp0),
                src0=sgpr(strideVgpr),
                src1=rowIdx,
                comment="stride * row (immediate)"))
        else:
            module.add(VMulLOU32(
                dst=vgpr(tmp0),
                src0=vgpr(rowIdx),
                src1=sgpr(strideVgpr),
                comment="stride * row"))
        
        # Mask to KERNEL_K boundary
        mask = ~(self.kernelK - 1) & 0xFFFFFFFF
        module.add(VAndB32(
            dst=vgpr(dstVgpr),
            src0=vgpr(tmp0),
            src1=mask,
            comment="Align to KERNEL_K=%d boundary" % self.kernelK))
        
        return module
    
    def computeLDSOffset(self, rowIdx, strideVgpr, readPtrVgpr, dstVgpr, tmpVgpr):
        """
        Compute LDS offset for a row.
        lds_offset[row] = stride * row - read_pointer[row]
        
        Args:
            rowIdx: VGPR or immediate containing row index
            strideVgpr: SGPR containing stride
            readPtrVgpr: VGPR containing read pointer (from computeReadPointer)
            dstVgpr: Destination VGPR for LDS offset
            tmpVgpr: Temporary VGPR
            
        Returns:
            Module with generated instructions
        """
        module = Module("ComputeLDSOffset%s_Row" % self.tc)
        
        if not self.isEnabled():
            return module
        
        module.addComment0("Compute LDS offset: stride * row - read_pointer[row]")
        
        # tmp = stride * row
        if isinstance(rowIdx, int):
            module.add(VMulLOU32(
                dst=vgpr(tmpVgpr),
                src0=sgpr(strideVgpr),
                src1=rowIdx,
                comment="stride * row"))
        else:
            module.add(VMulLOU32(
                dst=vgpr(tmpVgpr),
                src0=vgpr(rowIdx),
                src1=sgpr(strideVgpr),
                comment="stride * row"))
        
        # lds_offset = (stride * row) - read_pointer
        module.add(VSubU32(
            dst=vgpr(dstVgpr),
            src0=vgpr(tmpVgpr),
            src1=vgpr(readPtrVgpr),
            comment="LDS offset = (stride*row) - read_ptr"))
        
        return module
    
    def computeLDSAddress(self, ldsBase, row, ldsOffsetVgpr, laneIdx, iteration,
                         dstVgpr, tmpVgpr):
        """
        Compute wrapped LDS address for read/write.
        lds_addr = ldsBase + KERNEL_K * NUM_BUFFERS * row + 
                   (lds_offset + laneIdx + KERNEL_K * iter) % (KERNEL_K * NUM_BUFFERS)
        
        Args:
            ldsBase: Base LDS address (immediate or SGPR)
            row: Row number (immediate)
            ldsOffsetVgpr: VGPR containing LDS offset for this row
            laneIdx: VGPR containing lane index
            iteration: Iteration number (immediate or VGPR)
            dstVgpr: Destination VGPR for LDS address
            tmpVgpr: Temporary VGPR
            
        Returns:
            Module with generated instructions
        """
        module = Module("ComputeLDSAddress%s_Row%d" % (self.tc, row))
        
        if not self.isEnabled():
            return module
        
        module.addComment0("Compute wrapped LDS address for row %d" % row)
        
        bufferSize = self.kernelK * self.numBuffers
        rowOffset = bufferSize * row
        
        # Constant part: ldsBase + row_offset
        ldsConst = ldsBase + rowOffset
        
        # Variable part: (lds_offset + laneIdx + KERNEL_K * iter) % buffer_size
        
        # tmp = lds_offset + laneIdx
        module.add(VAddU32(
            dst=vgpr(tmpVgpr),
            src0=vgpr(ldsOffsetVgpr),
            src1=vgpr(laneIdx),
            comment="lds_offset + laneIdx"))
        
        # tmp = tmp + KERNEL_K * iter
        if isinstance(iteration, int):
            iterOffset = self.kernelK * iteration
            if iterOffset > 0:
                module.add(VAddU32(
                    dst=vgpr(tmpVgpr),
                    src0=vgpr(tmpVgpr),
                    src1=iterOffset,
                    comment="Add iteration offset"))
        else:
            module.add(VMulLOU32(
                dst=vgpr(dstVgpr),
                src0=self.kernelK,
                src1=vgpr(iteration),
                comment="KERNEL_K * iter"))
            module.add(VAddU32(
                dst=vgpr(tmpVgpr),
                src0=vgpr(tmpVgpr),
                src1=vgpr(dstVgpr),
                comment="Add iteration offset"))
        
        # Modulo operation: tmp % buffer_size
        # For power of 2, this is just AND with (buffer_size - 1)
        if (bufferSize & (bufferSize - 1)) == 0:
            # Power of 2
            mask = bufferSize - 1
            module.add(VAndB32(
                dst=vgpr(tmpVgpr),
                src0=vgpr(tmpVgpr),
                src1=mask,
                comment="Wrap within buffer"))
        else:
            # General modulo - would need division
            module.addComment0("TODO: Implement general modulo for non-power-of-2 buffer size")
        
        # Final address = constant + variable
        module.add(VAddU32(
            dst=vgpr(dstVgpr),
            src0=ldsConst,
            src1=vgpr(tmpVgpr),
            comment="Final LDS address"))
        
        return module


class RowReAlignmentHelper:
    """
    Helps with row re-alignment to group rows by alignment pattern.
    
    For odd strides, groups "even" and "odd" aligned rows together,
    reducing the number of VALU operations needed for alignment shifts.
    """
    
    def __init__(self, kernel, tP):
        """
        Initialize row re-alignment helper.
        
        Args:
            kernel: Kernel configuration dictionary
            tP: Tensor parameters (A or B)
        """
        self.kernel = kernel
        self.tP = tP
        self.tc = tP["tensorChar"]
        self.groupSize = kernel.get("RowReAlignGroupSize", 16)
        
    def isEnabled(self):
        """Check if row re-alignment is enabled."""
        return self.kernel.get("EnableRowReAlignment", False)
    
    def needsAlignment(self, strideVgpr, tmpSgpr):
        """
        Check if stride requires alignment (odd stride).
        
        Args:
            strideVgpr: SGPR containing stride
            tmpSgpr: Temporary SGPR for calculation
            
        Returns:
            Module with instructions to set SCC flag
        """
        module = Module("CheckStrideAlignment%s" % self.tc)
        
        if not self.isEnabled():
            return module
        
        module.addComment0("Check if stride is odd (needs alignment)")
        
        # AND with 1 to check LSB
        module.add(SAndB32(
            dst=sgpr(tmpSgpr),
            src0=sgpr(strideVgpr),
            src1=1,
            comment="Check if stride is odd (SCC set if true)"))
        
        return module
    
    def computeRowAlignmentGroup(self, rowVgpr, strideVgpr, dstVgpr, tmpVgpr):
        """
        Compute which alignment group a row belongs to (0 = even, 1 = odd).
        
        Args:
            rowVgpr: VGPR containing row index
            strideVgpr: SGPR containing stride
            dstVgpr: Destination VGPR (0 or 1)
            tmpVgpr: Temporary VGPR
            
        Returns:
            Module with generated instructions
        """
        module = Module("ComputeRowGroup%s" % self.tc)
        
        if not self.isEnabled():
            return module
        
        module.addComment0("Compute row alignment group (even/odd)")
        
        # tmp = row * stride
        module.add(VMulLOU32(
            dst=vgpr(tmpVgpr),
            src0=vgpr(rowVgpr),
            src1=sgpr(strideVgpr),
            comment="row * stride"))
        
        # Extract LSB to determine group
        module.add(VAndB32(
            dst=vgpr(dstVgpr),
            src0=vgpr(tmpVgpr),
            src1=1,
            comment="Extract alignment bit (0=even, 1=odd)"))
        
        return module
    
    def applyGroupedShift(self, dataVgprBase, numVgprs, groupVgpr, tmpVgpr):
        """
        Apply alignment shift to grouped rows.
        Uses v_alignbyte_b32 or similar to shift by 2 bytes (16 bits).
        
        Args:
            dataVgprBase: Base VGPR of data to shift
            numVgprs: Number of VGPRs in data
            groupVgpr: VGPR containing group ID (0 or 1)
            tmpVgpr: Temporary VGPR
            
        Returns:
            Module with generated instructions
        """
        module = Module("ApplyGroupedShift%s" % self.tc)
        
        if not self.isEnabled():
            return module
        
        module.addComment0("Apply 16-bit shift to odd-aligned group")
        module.addComment0("Using conditional execution based on group ID")
        
        # TODO: Implement actual v_alignbyte_b32 instructions
        # This would shift data by 16 bits for odd-aligned rows
        # The key optimization is doing this once per group instead of per row
        
        module.addComment0("TODO: Implement v_alignbyte_b32 for grouped shift")
        
        return module


def shouldEnableAlignmentOptimization(kernel, tP):
    """
    Determine if any alignment optimization should be enabled for this kernel/tensor.
    
    Args:
        kernel: Kernel configuration dictionary
        tP: Tensor parameters
        
    Returns:
        True if any alignment optimization is enabled and applicable
    """
    # Check if K-major (transposed) - alignment matters most for K-major access
    isKMajor = (tP["isA"] and kernel["ProblemType"]["TransposeA"]) or \
               (tP["isB"] and not kernel["ProblemType"]["TransposeB"])
    
    if not isKMajor:
        return False
    
    # Check if any optimization is enabled
    enabledFeatures = [
        kernel.get("EnableAddressInterleave", False),
        kernel.get("EnableLDSAlignment", False),
        kernel.get("EnableRowReAlignment", False)
    ]
    
    return any(enabledFeatures)

