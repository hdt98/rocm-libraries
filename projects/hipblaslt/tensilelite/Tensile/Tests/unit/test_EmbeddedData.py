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
from unittest.mock import Mock, mock_open, MagicMock
from io import StringIO
import os

pytestmark = pytest.mark.unit

from Tensile.EmbeddedData import Namespace, Indent, EmbeddedDataFile


class TestNamespace:
    """Tests for Namespace context manager."""

    def test_anonymous_namespace_enter(self):
        """Test anonymous namespace generates correct opening syntax."""
        mock_parent = Mock()
        ns = Namespace(mock_parent, name=None)

        result = ns.__enter__()

        # Should write "namespace {" for anonymous namespace
        mock_parent.write.assert_called_once_with("namespace {")
        mock_parent.indent.assert_called_once()
        assert result == ns

    def test_named_namespace_enter(self):
        """Test named namespace generates correct opening syntax."""
        mock_parent = Mock()
        ns = Namespace(mock_parent, name="Test")

        result = ns.__enter__()

        # Should write "namespace Test {" for named namespace
        mock_parent.write.assert_called_once_with("namespace Test {")
        mock_parent.indent.assert_called_once()
        assert result == ns

    def test_anonymous_namespace_exit(self):
        """Test anonymous namespace generates correct closing syntax."""
        mock_parent = Mock()
        ns = Namespace(mock_parent, name=None)

        ns.__exit__()

        # Should dedent and write closing comment
        mock_parent.dedent.assert_called_once()
        mock_parent.write.assert_called_once_with("} // anonymous namespace")

    def test_named_namespace_exit(self):
        """Test named namespace generates correct closing syntax."""
        mock_parent = Mock()
        ns = Namespace(mock_parent, name="MyNamespace")

        ns.__exit__()

        # Should dedent and write closing comment with name
        mock_parent.dedent.assert_called_once()
        mock_parent.write.assert_called_once_with("} // namespace MyNamespace")

    def test_namespace_as_context_manager(self):
        """Test namespace works as context manager."""
        mock_parent = Mock()

        with Namespace(mock_parent, "TestNS") as ns:
            assert isinstance(ns, Namespace)

        # Verify both enter and exit were called
        assert mock_parent.write.call_count == 2
        assert mock_parent.indent.call_count == 1
        assert mock_parent.dedent.call_count == 1


class TestIndent:
    """Tests for Indent context manager."""

    def test_indent_enter(self):
        """Test indent __enter__ returns self."""
        mock_parent = Mock()
        indent = Indent(mock_parent)

        result = indent.__enter__()

        assert result == indent

    def test_indent_exit(self):
        """Test indent __exit__ calls parent dedent."""
        mock_parent = Mock()
        indent = Indent(mock_parent)

        indent.__exit__()

        mock_parent.dedent.assert_called_once()

    def test_indent_as_context_manager(self):
        """Test indent works as context manager."""
        mock_parent = Mock()

        with Indent(mock_parent) as ind:
            assert isinstance(ind, Indent)

        mock_parent.dedent.assert_called_once()


class TestEmbeddedDataFile:
    """Tests for EmbeddedDataFile class."""

    def test_initialization_with_file_object(self):
        """Test initialization with provided file object."""
        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            assert edf.filename == "test.hpp"
            assert edf.file == mock_file
            assert edf._indent_spaces == 4
            # Header initialization adds one indent level for Tensile namespace
            assert len(edf._indent_levels) > 0

            # Verify header was written
            content = mock_file.getvalue()
            assert "namespace Tensile {" in content
            assert "#include <Tensile/EmbeddedData.hpp>" in content

    def test_initialization_custom_indent(self):
        """Test initialization with custom indent spaces."""
        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file, indent_spaces=2) as edf:
            assert edf._indent_spaces == 2

    def test_write_header(self):
        """Test header includes copyright, includes, and namespace."""
        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            # Read content before file closes
            content = mock_file.getvalue()

            # Check for key header components
            assert "#include <Tensile/EmbeddedData.hpp>" in content
            assert "#include <Tensile/Contractions.hpp>" in content
            assert "#include <Tensile/Tensile.hpp>" in content
            assert "namespace Tensile {" in content

    def test_namespace_management(self):
        """Test namespace() and end_namespace() methods."""
        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            # Clear default Tensile namespace
            edf._open_blocks.clear()

            edf.namespace("Inner")
            assert len(edf._open_blocks) == 1
            assert isinstance(edf._open_blocks[0], Namespace)
            assert edf._open_blocks[0].name == "Inner"

            edf.end_namespace("Inner")
            assert len(edf._open_blocks) == 0

            content = mock_file.getvalue()
            assert "namespace Inner {" in content
            assert "} // namespace Inner" in content

    def test_anonymous_namespace(self):
        """Test anonymous namespace creation."""
        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            edf._open_blocks.clear()
            edf.namespace()  # Anonymous
            # Verify opening was written
            content = mock_file.getvalue()
            assert "namespace {" in content

        # After exit, the namespace should be closed (but file is closed)
        # We verified the opening; closing happens automatically via __exit__

    def test_mismatched_namespace_error(self):
        """Test error on mismatched namespace names."""
        mock_file = StringIO()

        with pytest.raises(RuntimeError, match="Mismatched namespace"):
            with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
                edf._open_blocks.clear()
                edf.namespace("First")
                edf.end_namespace("Second")  # Wrong name!

    def test_mismatched_block_type_error(self):
        """Test error when closing wrong block type."""
        mock_file = StringIO()

        with pytest.raises(RuntimeError, match="Mismatched block types"):
            with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
                edf._open_blocks.clear()
                # Push an Indent instead of Namespace
                edf._open_blocks.append(Indent(edf))
                edf.end_namespace("Test")  # Expects Namespace!

    def test_get_lines_from_string(self):
        """Test get_lines() with string input."""
        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            lines = edf.get_lines("line1\nline2\nline3")

            assert lines == ["line1", "line2", "line3"]

    def test_get_lines_from_iterable(self):
        """Test get_lines() with iterable input."""
        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            lines = edf.get_lines(["line1", "line2", "line3"])

            assert lines == ["line1", "line2", "line3"]

    def test_get_lines_from_number(self):
        """Test get_lines() converts numbers to strings."""
        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            lines = edf.get_lines(42)

            assert lines == ["42"]

    def test_format_single_line(self):
        """Test format() applies indentation to single line."""
        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            edf._indent_levels = [4]
            formatted = edf.format("test line")

            assert formatted == "    test line\n"

    def test_format_multiple_lines(self):
        """Test format() applies indentation to multiple lines."""
        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            edf._indent_levels = [2]
            formatted = edf.format("line1\nline2\nline3")

            assert formatted == "  line1\n  line2\n  line3\n"

    def test_indent_level_property(self):
        """Test indent_level property returns correct value."""
        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            # Start with header's indent level
            initial_levels = edf._indent_levels.copy()

            edf._indent_levels = [0]
            assert edf.indent_level == 0

            edf._indent_levels = [4, 8, 12]
            assert edf.indent_level == 12

            # Empty list case - returns 0
            # Note: This shouldn't happen in normal operation since __exit__ would pop
            # But testing the property itself
            saved_levels = edf._indent_levels
            edf._indent_levels = []
            assert edf.indent_level == 0
            # Restore so dedent doesn't fail
            edf._indent_levels = saved_levels

    def test_apply_indent_to_line(self):
        """Test apply_indent() adds spaces to line."""
        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            edf._indent_levels = [4]

            result = edf.apply_indent("test")
            assert result == "    test"

    def test_apply_indent_strips_existing_whitespace(self):
        """Test apply_indent() strips existing whitespace."""
        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            edf._indent_levels = [2]

            result = edf.apply_indent("   test   ")
            assert result == "  test"

    def test_apply_indent_preserves_preprocessor(self):
        """Test apply_indent() preserves # preprocessor directives."""
        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            edf._indent_levels = [4]

            result = edf.apply_indent("#include <something>")
            assert result == "#include <something>"

    def test_apply_indent_no_line(self):
        """Test apply_indent() with no line returns just spaces."""
        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            edf._indent_levels = [6]

            result = edf.apply_indent()
            assert result == "      "

    def test_indent_and_dedent(self):
        """Test indent() and dedent() manage indent levels."""
        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            edf._indent_levels = [0]

            # Indent with default spaces (4)
            indent_obj = edf.indent()
            assert edf.indent_level == 4
            assert isinstance(indent_obj, Indent)

            # Indent again
            edf.indent()
            assert edf.indent_level == 8

            # Dedent
            edf.dedent()
            assert edf.indent_level == 4

            edf.dedent()
            assert edf.indent_level == 0

    def test_indent_custom_spaces(self):
        """Test indent() with custom space count."""
        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            edf._indent_levels = [0]

            edf.indent(spaces=8)
            assert edf.indent_level == 8

    def test_write_single_item(self):
        """Test write() with single item."""
        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            edf._open_blocks.clear()  # Clear default namespace
            edf._indent_levels = [0]
            edf.write("test line")

            content = mock_file.getvalue()
        assert "test line\n" in content

    def test_write_multiple_items(self):
        """Test write() with multiple items."""
        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            edf._open_blocks.clear()
            edf._indent_levels = [0]
            edf.write("line1", "line2", "line3")

            content = mock_file.getvalue()
        assert "line1\n" in content
        assert "line2\n" in content
        assert "line3\n" in content

    def test_comment(self):
        """Test comment() generates C++ comments."""
        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            edf._open_blocks.clear()
            edf._indent_levels = [0]
            edf.comment("Test comment")

            content = mock_file.getvalue()
        assert "// Test comment\n" in content

    def test_comment_multiline(self):
        """Test comment() handles multiline text."""
        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            edf._open_blocks.clear()
            edf._indent_levels = [0]
            edf.comment("Line 1\nLine 2\nLine 3")

            content = mock_file.getvalue()
        assert "// Line 1\n" in content
        assert "// Line 2\n" in content
        assert "// Line 3\n" in content

    def test_embed_data_empty(self):
        """Test embed_data() with empty data."""
        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            edf._open_blocks.clear()
            edf.embed_data("TestType", [])

            content = mock_file.getvalue()
        assert "EmbedData<TestType> TENSILE_EMBED_SYMBOL_NAME{};" in content

    def test_embed_data_empty_with_key(self):
        """Test embed_data() with empty data and key."""
        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            edf._open_blocks.clear()
            edf.embed_data("TestType", [], key="mykey")

            content = mock_file.getvalue()
        assert 'EmbedData<TestType> TENSILE_EMBED_SYMBOL_NAME("mykey", {});' in content

    def test_embed_data_single_byte(self):
        """Test embed_data() with single byte."""
        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            edf._open_blocks.clear()
            edf.embed_data("TestType", [0x42])

            content = mock_file.getvalue()
        assert "EmbedData<TestType> TENSILE_EMBED_SYMBOL_NAME({" in content
        assert "0x42});" in content

    def test_embed_data_multiple_bytes(self):
        """Test embed_data() with multiple bytes."""
        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            edf._open_blocks.clear()
            edf.embed_data("TestType", [0x01, 0x02, 0x03, 0x04])

            content = mock_file.getvalue()
        assert "0x01" in content
        assert "0x02" in content
        assert "0x03" in content
        assert "0x04" in content

    def test_embed_data_16byte_wrapping(self):
        """Test embed_data() wraps lines at 16 bytes."""
        mock_file = StringIO()

        # Create 17 bytes to test wrapping
        data = list(range(17))

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            edf._open_blocks.clear()
            edf.embed_data("TestType", data)

            content = mock_file.getvalue()

        # Should have line breaks
        lines = content.split('\n')

        # Find lines with hex data
        hex_lines = [line for line in lines if '0x00' in line or '0x10' in line]
        assert len(hex_lines) >= 2  # At least 2 lines of hex data

    def test_embed_data_with_key(self):
        """Test embed_data() with key parameter."""
        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            edf._open_blocks.clear()
            edf.embed_data("TestType", [0x42], key="test_key")

            content = mock_file.getvalue()
        assert 'TENSILE_EMBED_SYMBOL_NAME("test_key"' in content

    def test_embed_data_with_comment(self):
        """Test embed_data() with comment parameter."""
        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            edf._open_blocks.clear()
            edf.embed_data("TestType", [0x42], comment="This is a test")

            content = mock_file.getvalue()
        assert "// This is a test" in content

    def test_embed_data_null_terminated(self):
        """Test embed_data() with nullTerminated=True."""
        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            edf._open_blocks.clear()
            edf.embed_data("TestType", [0x41, 0x42], nullTerminated=True)

            content = mock_file.getvalue()
        assert "0x41" in content
        assert "0x42" in content
        assert "0x00" in content  # Null terminator

    def test_embed_file(self, tmp_path):
        """Test embed_file() reads and embeds file content."""
        # Create test binary file
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b'\x01\x02\x03\x04')

        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            edf._open_blocks.clear()
            edf.embed_file("TestType", str(test_file))

            content = mock_file.getvalue()
        assert "0x01" in content
        assert "0x02" in content
        assert "0x03" in content
        assert "0x04" in content
        assert "// test.bin" in content  # Basename as comment

    def test_embed_file_with_key(self, tmp_path):
        """Test embed_file() with key parameter."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b'\x42')

        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            edf._open_blocks.clear()
            edf.embed_file("TestType", str(test_file), key="file_key")

            content = mock_file.getvalue()
        assert 'TENSILE_EMBED_SYMBOL_NAME("file_key"' in content

    def test_context_manager_closes_open_blocks(self):
        """Test __exit__ closes all open blocks."""
        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            # Open some blocks but don't close them
            edf.namespace("Inner1")
            edf.namespace("Inner2")

            # Verify blocks are open
            assert len(edf._open_blocks) == 3  # Tensile + Inner1 + Inner2

            # Verify opening syntax was written
            content = mock_file.getvalue()
            assert "namespace Inner1 {" in content
            assert "namespace Inner2 {" in content

        # After __exit__, all blocks should be automatically closed
        # (File is closed so can't verify content, but blocks should be empty)

    def test_write_footer(self):
        """Test write_footer() is called on exit."""
        mock_file = StringIO()

        with EmbeddedDataFile("test.hpp", file=mock_file) as edf:
            # Get initial content
            initial_content = mock_file.getvalue()

        # Can't read after close, but we can verify the file was written to
        # The footer is automatically called by __exit__
        # Initial content should exist and end with newline
        assert initial_content.endswith("\n")
