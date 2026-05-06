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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
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

import pytest

pytestmark = pytest.mark.unit

from Tensile.Common.Types import (
    IsaInfo,
    SemanticVersion,
    IsaVersion,
    DebugConfig,
    makeDebugConfig
)


class TestIsaInfo:
    """Tests for IsaInfo dataclass."""

    def test_initialization(self):
        """Test IsaInfo initialization with all fields."""
        asm_caps = {"cap1": True}
        arch_caps = {"arch1": "value"}
        reg_caps = {"reg1": 256}
        asm_bugs = {"bug1": False}

        isa_info = IsaInfo(
            asmCaps=asm_caps,
            archCaps=arch_caps,
            regCaps=reg_caps,
            asmBugs=asm_bugs
        )

        assert isa_info.asmCaps == asm_caps
        assert isa_info.archCaps == arch_caps
        assert isa_info.regCaps == reg_caps
        assert isa_info.asmBugs == asm_bugs

    def test_empty_dictionaries(self):
        """Test IsaInfo with empty dictionaries."""
        isa_info = IsaInfo(
            asmCaps={},
            archCaps={},
            regCaps={},
            asmBugs={}
        )

        assert isa_info.asmCaps == {}
        assert isa_info.archCaps == {}
        assert isa_info.regCaps == {}
        assert isa_info.asmBugs == {}


class TestSemanticVersion:
    """Tests for SemanticVersion NamedTuple."""

    def test_initialization(self):
        """Test SemanticVersion initialization."""
        version = SemanticVersion(major=1, minor=2, patch=3)

        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

    def test_equality(self):
        """Test SemanticVersion equality comparison."""
        v1 = SemanticVersion(1, 2, 3)
        v2 = SemanticVersion(1, 2, 3)
        v3 = SemanticVersion(1, 2, 4)

        assert v1 == v2
        assert v1 != v3

    def test_ordering(self):
        """Test SemanticVersion ordering."""
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(1, 2, 0)
        v3 = SemanticVersion(2, 0, 0)

        assert v1 < v2
        assert v2 < v3
        assert v1 < v3

    def test_tuple_unpacking(self):
        """Test SemanticVersion can be unpacked as tuple."""
        version = SemanticVersion(5, 10, 15)
        major, minor, patch = version

        assert major == 5
        assert minor == 10
        assert patch == 15

    def test_immutability(self):
        """Test SemanticVersion is immutable."""
        version = SemanticVersion(1, 2, 3)

        with pytest.raises(AttributeError):
            version.major = 5


class TestIsaVersion:
    """Tests for IsaVersion alias."""

    def test_is_semantic_version(self):
        """Test IsaVersion is alias for SemanticVersion."""
        assert IsaVersion is SemanticVersion

    def test_usage_as_isa_version(self):
        """Test using IsaVersion works identically to SemanticVersion."""
        isa = IsaVersion(9, 0, 10)

        assert isa.major == 9
        assert isa.minor == 0
        assert isa.patch == 10


class TestDebugConfig:
    """Tests for DebugConfig NamedTuple."""

    def test_default_initialization(self):
        """Test DebugConfig with default values."""
        config = DebugConfig()

        assert config.enableAsserts is False
        assert config.enableDebugA is False
        assert config.enableDebugB is False
        assert config.enableDebugC is False
        assert config.expectedValueC == 16.0
        assert config.forceCExpectedValue is False
        assert config.debugKernel is False
        assert config.forceGenerateKernel is False
        assert config.printSolutionRejectionReason is False
        assert config.splitGSU is False
        assert config.printIndexAssignmentInfo is False

    def test_custom_initialization(self):
        """Test DebugConfig with custom values."""
        config = DebugConfig(
            enableAsserts=True,
            enableDebugA=True,
            enableDebugB=False,
            enableDebugC=True,
            expectedValueC=32.0,
            forceCExpectedValue=True,
            debugKernel=True,
            forceGenerateKernel=False,
            printSolutionRejectionReason=True,
            splitGSU=True,
            printIndexAssignmentInfo=False
        )

        assert config.enableAsserts is True
        assert config.enableDebugA is True
        assert config.enableDebugB is False
        assert config.enableDebugC is True
        assert config.expectedValueC == 32.0
        assert config.forceCExpectedValue is True
        assert config.debugKernel is True
        assert config.forceGenerateKernel is False
        assert config.printSolutionRejectionReason is True
        assert config.splitGSU is True
        assert config.printIndexAssignmentInfo is False

    def test_immutability(self):
        """Test DebugConfig is immutable."""
        config = DebugConfig()

        with pytest.raises(AttributeError):
            config.enableAsserts = True


class TestMakeDebugConfig:
    """Tests for makeDebugConfig function."""

    def test_empty_config_returns_defaults(self):
        """Test makeDebugConfig with empty dict returns defaults."""
        config = makeDebugConfig({})

        assert config.enableAsserts is False
        assert config.enableDebugA is False
        assert config.enableDebugB is False
        assert config.enableDebugC is False
        assert config.expectedValueC == 16.0
        assert config.forceCExpectedValue is False
        assert config.debugKernel is False
        assert config.forceGenerateKernel is False
        assert config.printSolutionRejectionReason is False
        assert config.splitGSU is False
        assert config.printIndexAssignmentInfo is False

    def test_enable_asserts(self):
        """Test EnableAsserts flag."""
        config = makeDebugConfig({"EnableAsserts": True})
        assert config.enableAsserts is True

    def test_enable_debug_a(self):
        """Test EnableDebugA flag."""
        config = makeDebugConfig({"EnableDebugA": True})
        assert config.enableDebugA is True

    def test_enable_debug_b(self):
        """Test EnableDebugB flag."""
        config = makeDebugConfig({"EnableDebugB": True})
        assert config.enableDebugB is True

    def test_enable_debug_c(self):
        """Test EnableDebugC flag."""
        config = makeDebugConfig({"EnableDebugC": True})
        assert config.enableDebugC is True

    def test_expected_value_c(self):
        """Test ExpectedValueC setting."""
        config = makeDebugConfig({"ExpectedValueC": 64.0})
        assert config.expectedValueC == 64.0

    def test_force_c_expected_value(self):
        """Test ForceCExpectedValue flag."""
        config = makeDebugConfig({"ForceCExpectedValue": True})
        assert config.forceCExpectedValue is True

    def test_debug_kernel(self):
        """Test DebugKernel flag."""
        config = makeDebugConfig({"DebugKernel": True})
        assert config.debugKernel is True

    def test_force_generate_kernel(self):
        """Test ForceGenerateKernel flag."""
        config = makeDebugConfig({"ForceGenerateKernel": True})
        assert config.forceGenerateKernel is True

    def test_print_solution_rejection_reason(self):
        """Test PrintSolutionRejectionReason flag."""
        config = makeDebugConfig({"PrintSolutionRejectionReason": True})
        assert config.printSolutionRejectionReason is True

    def test_split_gsu(self):
        """Test SplitGSU flag."""
        config = makeDebugConfig({"SplitGSU": True})
        assert config.splitGSU is True

    def test_print_index_assignment_info(self):
        """Test PrintIndexAssignmentInfo flag."""
        config = makeDebugConfig({"PrintIndexAssignmentInfo": True})
        assert config.printIndexAssignmentInfo is True

    def test_multiple_flags(self):
        """Test makeDebugConfig with multiple flags."""
        config = makeDebugConfig({
            "EnableAsserts": True,
            "DebugKernel": True,
            "ExpectedValueC": 128.0,
            "SplitGSU": True
        })

        assert config.enableAsserts is True
        assert config.debugKernel is True
        assert config.expectedValueC == 128.0
        assert config.splitGSU is True
        # Other flags should remain default
        assert config.enableDebugA is False
        assert config.forceGenerateKernel is False

    def test_all_flags_enabled(self):
        """Test makeDebugConfig with all flags enabled."""
        config = makeDebugConfig({
            "EnableAsserts": True,
            "EnableDebugA": True,
            "EnableDebugB": True,
            "EnableDebugC": True,
            "ExpectedValueC": 256.0,
            "ForceCExpectedValue": True,
            "DebugKernel": True,
            "ForceGenerateKernel": True,
            "PrintSolutionRejectionReason": True,
            "SplitGSU": True,
            "PrintIndexAssignmentInfo": True
        })

        assert config.enableAsserts is True
        assert config.enableDebugA is True
        assert config.enableDebugB is True
        assert config.enableDebugC is True
        assert config.expectedValueC == 256.0
        assert config.forceCExpectedValue is True
        assert config.debugKernel is True
        assert config.forceGenerateKernel is True
        assert config.printSolutionRejectionReason is True
        assert config.splitGSU is True
        assert config.printIndexAssignmentInfo is True

    def test_ignores_unknown_keys(self):
        """Test makeDebugConfig ignores unknown config keys."""
        config = makeDebugConfig({
            "EnableAsserts": True,
            "UnknownKey": "value",
            "AnotherUnknownKey": 42
        })

        # Should only set the known key
        assert config.enableAsserts is True
        # Verify it returns a valid DebugConfig
        assert isinstance(config, DebugConfig)
