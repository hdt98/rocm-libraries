#!/usr/bin/env python3
"""
Apply CMake integration for layout-specific convolution libraries.
Modifies the main CMakeLists.txt to use layout-split structure.
"""

from pathlib import Path
import re

def apply_integration():
    cmake_file = Path('/home/AMD/bhargrea/github/rocm-libraries/projects/composablekernel/library/src/tensor_operation_instance/gpu/CMakeLists.txt')
    toplevel_lib_file = Path(__file__).parent / 'toplevel_layout_libraries.cmake'

    print(f"Reading {cmake_file}...")
    with open(cmake_file, 'r') as f:
        content = f.read()
        lines = content.splitlines()

    # Backup already created - was done manually

    # Build the list of layout-split operations
    layout_split_ops = [
        '"grouped_conv1d_fwd"',
        '"grouped_conv1d_bwd_weight"',
        '"grouped_conv2d_fwd"',
        '"grouped_conv2d_bwd_data"',
        '"grouped_conv2d_bwd_weight"',
        '"grouped_conv2d_fwd_clamp"',
        '"grouped_conv2d_fwd_bias_clamp"',
        '"grouped_conv2d_fwd_bias_bnorm_clamp"',
        '"grouped_conv2d_fwd_dynamic_op"',
        '"grouped_conv3d_fwd"',
        '"grouped_conv3d_bwd_data"',
        '"grouped_conv3d_bwd_weight"',
        '"grouped_conv3d_fwd_scale"',
        '"grouped_conv3d_fwd_clamp"',
        '"grouped_conv3d_fwd_bias_clamp"',
        '"grouped_conv3d_fwd_bias_bnorm_clamp"',
        '"grouped_conv3d_fwd_bilinear"',
        '"grouped_conv3d_fwd_convscale"',
        '"grouped_conv3d_fwd_convscale_add"',
        '"grouped_conv3d_fwd_convscale_relu"',
        '"grouped_conv3d_fwd_convinvscale"',
        '"grouped_conv3d_fwd_dynamic_op"',
        '"grouped_conv3d_fwd_scaleadd_ab"',
        '"grouped_conv3d_fwd_scaleadd_scaleadd_relu"',
        '"grouped_conv3d_bwd_data_bilinear"',
        '"grouped_conv3d_bwd_data_scale"',
        '"grouped_conv3d_bwd_weight_bilinear"',
        '"grouped_conv3d_bwd_weight_scale"',
    ]

    # CHANGE 1: Modify the add_subdirectory block
    modified_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Find the add_subdirectory block
        if 'if((add_inst EQUAL 1))' in line and i < len(lines) - 5:
            # Check if this is the right block (has get_filename_component)
            if 'get_filename_component(target_dir' in lines[i+1]:
                # Found it! Insert the skip logic
                modified_lines.append(line)  # if((add_inst EQUAL 1))
                modified_lines.append(lines[i+1])  # get_filename_component
                modified_lines.append('')
                modified_lines.append('            # Skip convolution operations that are now split by layout')
                modified_lines.append('            # These are handled explicitly below')
                modified_lines.append('            set(layout_split_ops')
                for op in layout_split_ops:
                    modified_lines.append(f'                {op}')
                modified_lines.append('            )')
                modified_lines.append('')
                modified_lines.append('            if(NOT target_dir IN_LIST layout_split_ops)')

                # Continue with original lines but indented
                i += 2  # Skip the two lines we already added
                # Add remaining lines of the block with extra indentation
                while i < len(lines) and not ('else()' in lines[i] and 'skip_instance_directory' in lines[i+1]):
                    # Add extra indent for lines inside the if block
                    if lines[i].strip() and not lines[i].strip().startswith('#'):
                        modified_lines.append('    ' + lines[i])
                    else:
                        modified_lines.append(lines[i])
                    i += 1

                    # Check if we hit the closing endif for the TARGET check
                    if 'endif()' in lines[i-1] and 'add_instance_directory' in lines[i-2]:
                        # Add the closing endif for our IN_LIST check
                        modified_lines.append('            endif()  # NOT IN_LIST check')
                        break
                continue

        modified_lines.append(line)
        i += 1

    # CHANGE 2: Add explicit subdirectory includes after ENDFOREACH
    i = 0
    final_lines = []
    while i < len(modified_lines):
        line = modified_lines[i]
        final_lines.append(line)

        # Find ENDFOREACH() that closes the directory scan
        if line.strip() == 'ENDFOREACH()' and i > 200:  # Make sure it's the right one
            # Check if previous lines have the directory scan logic
            context = '\n'.join(modified_lines[max(0, i-10):i])
            if 'add_subdirectory' in context or 'skip_instance_directory' in context:
                # This is the right ENDFOREACH
                final_lines.append('')
                final_lines.append('# ============================================================================='
)
                final_lines.append('# Layout-Split Convolution Operations (Explicit Includes)')
                final_lines.append('# ============================================================================='
)
                final_lines.append('# These operations are now split by data layout for selective linking.')
                final_lines.append('# Each operation includes multiple layout subdirectories (gnhwc, nhwgc, etc.)')
                final_lines.append('')
                final_lines.append('if(NOT HIPTENSOR_REQ_LIBS_ONLY OR MIOPEN_REQ_LIBS_ONLY)')
                final_lines.append('    message(STATUS "Including layout-split convolution operations...")')
                final_lines.append('')
                final_lines.append('    # 1D convolutions')
                final_lines.append('    add_subdirectory(grouped_conv1d_fwd)')
                final_lines.append('    add_subdirectory(grouped_conv1d_bwd_weight)')
                final_lines.append('')
                final_lines.append('    # 2D convolutions - core operations')
                final_lines.append('    add_subdirectory(grouped_conv2d_fwd)')
                final_lines.append('    add_subdirectory(grouped_conv2d_bwd_data)')
                final_lines.append('    add_subdirectory(grouped_conv2d_bwd_weight)')
                final_lines.append('')
                final_lines.append('    # 2D convolutions - variants')
                final_lines.append('    add_subdirectory(grouped_conv2d_fwd_clamp)')
                final_lines.append('    add_subdirectory(grouped_conv2d_fwd_bias_clamp)')
                final_lines.append('    add_subdirectory(grouped_conv2d_fwd_bias_bnorm_clamp)')
                final_lines.append('    add_subdirectory(grouped_conv2d_fwd_dynamic_op)')
                final_lines.append('')
                final_lines.append('    # 3D convolutions - core operations')
                final_lines.append('    add_subdirectory(grouped_conv3d_fwd)')
                final_lines.append('    add_subdirectory(grouped_conv3d_bwd_data)')
                final_lines.append('    add_subdirectory(grouped_conv3d_bwd_weight)')
                final_lines.append('')
                final_lines.append('    # 3D convolutions - variants')
                final_lines.append('    add_subdirectory(grouped_conv3d_fwd_scale)')
                final_lines.append('    add_subdirectory(grouped_conv3d_fwd_clamp)')
                final_lines.append('    add_subdirectory(grouped_conv3d_fwd_bias_clamp)')
                final_lines.append('    add_subdirectory(grouped_conv3d_fwd_bias_bnorm_clamp)')
                final_lines.append('    add_subdirectory(grouped_conv3d_fwd_bilinear)')
                final_lines.append('    add_subdirectory(grouped_conv3d_fwd_convscale)')
                final_lines.append('    add_subdirectory(grouped_conv3d_fwd_convscale_add)')
                final_lines.append('    add_subdirectory(grouped_conv3d_fwd_convscale_relu)')
                final_lines.append('    add_subdirectory(grouped_conv3d_fwd_convinvscale)')
                final_lines.append('    add_subdirectory(grouped_conv3d_fwd_dynamic_op)')
                final_lines.append('    add_subdirectory(grouped_conv3d_fwd_scaleadd_ab)')
                final_lines.append('    add_subdirectory(grouped_conv3d_fwd_scaleadd_scaleadd_relu)')
                final_lines.append('    add_subdirectory(grouped_conv3d_bwd_data_bilinear)')
                final_lines.append('    add_subdirectory(grouped_conv3d_bwd_data_scale)')
                final_lines.append('    add_subdirectory(grouped_conv3d_bwd_weight_bilinear)')
                final_lines.append('    add_subdirectory(grouped_conv3d_bwd_weight_scale)')
                final_lines.append('endif()')
                final_lines.append('')
                final_lines.append('# ============================================================================='
)
                final_lines.append('# End of Layout-Split Operations')
                final_lines.append('# ============================================================================='
)
                final_lines.append('')

        i += 1

    # CHANGE 3: Comment out old device_conv_operations and include new definitions
    i = 0
    result_lines = []
    while i < len(final_lines):
        line = final_lines[i]

        # Find the old device_conv_operations definition
        if 'add_library(device_conv_operations ${CK_DEVICE_CONV_INSTANCES})' in line:
            # Comment out the old block
            result_lines.append('# OLD: Monolithic device_conv_operations (replaced by layout-specific libraries)')
            result_lines.append('# ' + line)
            i += 1
            # Comment out the next ~20 lines (the whole if block)
            depth = 1
            while i < len(final_lines) and depth > 0:
                line = final_lines[i]
                if 'if(' in line or 'if (' in line:
                    depth += 1
                elif line.strip() == 'endif()':
                    depth -= 1

                result_lines.append('# ' + line if line.strip() else line)
                i += 1
                if depth == 0:
                    break

            # Insert the include
            result_lines.append('')
            result_lines.append('# ============================================================================='
)
            result_lines.append('# Layout-Specific Convolution Library Definitions')
            result_lines.append('# ============================================================================='
)

            # Read and insert the toplevel library definitions
            with open(toplevel_lib_file, 'r') as f:
                for lib_line in f:
                    result_lines.append(lib_line.rstrip())

            result_lines.append('# ============================================================================='
)
            result_lines.append('')
            continue

        result_lines.append(line)
        i += 1

    # Write the result
    output_file = cmake_file.parent / 'CMakeLists.txt'
    with open(output_file, 'w') as f:
        f.write('\n'.join(result_lines))

    print(f"\n✅ Integration complete!")
    print(f"Modified: {output_file}")
    print(f"\nChanges made:")
    print("  1. Added skip logic for layout-split operations in directory scan")
    print("  2. Added explicit includes for 28 layout-split operations")
    print("  3. Replaced monolithic device_conv_operations with layout-specific libraries")
    print(f"\nBackup saved as: {cmake_file}.before_layout_split")
    print("\nNext: Test the build with 'cmake --build . --target device_conv2d_nhwgc_operations'")


if __name__ == '__main__':
    apply_integration()
