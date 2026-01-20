#!/usr/bin/python3

"""Copyright (C) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
   ies of the Software, and to permit persons to whom the Software is furnished
   to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
   PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
   FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
   COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
   IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
   CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import re
import sys
import os
import platform
import subprocess
import argparse
import pathlib
from xml.dom import minidom
from fnmatch import fnmatchcase

SCRIPT_VERSION = 0.1

args = {}
param = {}
OS_info = {}
var_subs = {}

vcpkg_script = ['tdir %IDIR%',
                'git clone -b 2024.02.14 https://github.com/microsoft/vcpkg %IDIR%', 'cd %IDIR%', 'bootstrap-vcpkg.bat -disableMetrics' ]

xml_script = [ '%XML%' ]


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="""
    Checks build arguments
    """)
    # parser.add_argument('--install', required=False, default = True,  action='store_true',
    #                     help='Install dependencies (optional, default: True)')
    parser.add_argument('-i', '--install_dir', type=str, required=False, default = ("" if os.name == "nt" else "./build/deps"),
                        help='Install directory path (optional, windows default: C:\\github\\vcpkg, linux default: ./build/deps)')
    # parser.add_argument('-v', '--verbose', required=False, default = False, action='store_true',
    #                     help='Verbose install (optional, default: False)')
    return parser.parse_args()

def os_detect():
    global OS_info
    if os.name == "nt":
        OS_info["ID"] = platform.system()
    else:
        inf_file = "/etc/os-release"
        if os.path.exists(inf_file):
            with open(inf_file) as f:
                for line in f:
                    if "=" in line:
                        k,v = line.strip().split("=")
                        OS_info[k] = v.replace('"','')
    OS_info["NUM_PROC"] = os.cpu_count()
    print(OS_info)

def create_dir(dir_path):
    if os.path.isabs(dir_path):
        full_path = dir_path
    else:
        full_path = os.path.join( os.getcwd(), dir_path )
    return pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)

def delete_dir(dir_path) :
    if (not os.path.exists(dir_path)):
        return
    if os.name == "nt":
        return run_cmd( "RMDIR" , f"/S /Q {dir_path}")
    else:
        linux_path = pathlib.Path(dir_path).absolute()
        return run_cmd( "rm" , f"-rf {linux_path}")

def run_cmd(cmd):
    global args
    if (cmd.startswith('cd ')):
        return os.chdir(cmd[3:])
    if (cmd.startswith('mkdir ')):
        return create_dir(cmd[6:])
    cmdline = f"{cmd}"
    print(cmdline)
    proc = subprocess.run(cmdline, check=True, stderr=subprocess.STDOUT, shell=True)
    return proc.returncode


def build_aocl_windows():
    """Build AOCL 5.2 from source on Windows (similar to install.sh on Linux)"""
    print('***\n*** Building AOCL 5.2 from source\n***')
    
    cwd = pathlib.Path.cwd()
    
    try:
        build_dir = pathlib.Path('build/deps').absolute()
        aocl_dir = build_dir / 'aocl'
        
        # Skip if already built
        if (aocl_dir / 'install_package' / 'lib' / 'libaocl.lib').exists():
            print(f'AOCL 5.2 already built at: {aocl_dir}')
            print('(use --clean-deps or delete build/deps/aocl to rebuild)')
            return True
        
        # Create build directory
        build_dir.mkdir(parents=True, exist_ok=True)
        
        # Clone AOCL 5.2 from GitHub
        if not aocl_dir.exists():
            print('Cloning AOCL 5.2 from GitHub (branch AOCL-5.2)...')
            os.chdir(build_dir)
            cmd = 'git clone --quiet --depth 1 --branch AOCL-5.2 https://github.com/amd/aocl.git'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f'Error cloning AOCL: {result.stderr}')
                return False
            print('Clone complete')
        
        # Build AOCL with CMake (using Clang from ROCm)
        os.chdir(aocl_dir)
        install_prefix = aocl_dir / 'install_package'
        
        # AOCL requires GCC/ICC/Clang - use Clang from ROCm
        # Prefer ROCM_PATH (modern), fall back to HIP_PATH (historical)
        rocm_path = os.getenv('ROCM_PATH') or os.getenv('HIP_PATH')
        if not rocm_path:
            print('ERROR: ROCM_PATH or HIP_PATH environment variable not set.')
            print('Please ensure ROCm is properly installed and the environment is configured.')
            return False
        clang_exe = os.path.join(rocm_path, 'bin', 'clang.exe')
        clangxx_exe = os.path.join(rocm_path, 'bin', 'clang++.exe')
        
        # Use OpenMP from Visual Studio MSVC toolchain
        # Find the latest MSVC version
        msvc_base = r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC'
        if os.path.exists(msvc_base):
            msvc_versions = sorted([d for d in os.listdir(msvc_base) if os.path.isdir(os.path.join(msvc_base, d))], reverse=True)
            if msvc_versions:
                msvc_dir = os.path.join(msvc_base, msvc_versions[0])
                openmp_lib = os.path.join(msvc_dir, 'lib', 'x64', 'libomp.lib')
                openmp_include = os.path.join(msvc_dir, 'include')
                print(f'Using OpenMP from Visual Studio: {openmp_lib}')
                print(f'OpenMP include directory: {openmp_include}')
            else:
                openmp_lib = ''
                openmp_include = ''
                print('WARNING: Could not find Visual Studio OpenMP library')
        else:
            openmp_lib = ''
            openmp_include = ''
            print('WARNING: Could not find Visual Studio MSVC toolchain')
        
        # Clone and patch BLIS before running AOCL CMake
        # This way AOCL will use our patched BLIS
        blis_src_dir = build_dir / 'blis'
        if not blis_src_dir.exists():
            print('Cloning BLIS for patching...')
            os.chdir(build_dir)
            cmd = 'git clone --quiet --depth 1 --branch master https://github.com/amd/blis.git blis'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f'Error cloning BLIS: {result.stderr}')
                return False
            print('BLIS clone complete')
        
        # Patch BLIS CMakeLists.txt (do this every time, not just on first clone)
        blis_cmake = blis_src_dir / 'CMakeLists.txt'
        if blis_cmake.exists():
            print('Patching BLIS CMakeLists.txt to fix OpenMP SIMD flags and add OpenMP include path...')
            content = blis_cmake.read_text()
            
            # Fix 1: Replace MSVC-style OpenMP flag with Clang-compatible one
            old_line = '        set(COMPSIMDFLAGS /openmp:experimental)'
            new_line = '        set(COMPSIMDFLAGS -fopenmp-simd)  # Changed from /openmp:experimental for Clang'
            if old_line in content:
                content = content.replace(old_line, new_line)
                print('  ✓ Fixed OpenMP SIMD flags')
            else:
                print('  WARNING: Could not find OpenMP SIMD flags to patch')
            
            # Fix 2: Add OpenMP include directory and defines after FindOpenMP
            marker = '    find_package(OpenMP)'
            if marker in content:
                # Convert backslashes to forward slashes for CMake
                openmp_include_cmake = openmp_include.replace('\\', '/')
                omp_header_path = f"{openmp_include_cmake}/omp.h"
                
                # Create a stub header for missing OpenMP 3.0 functions (Visual Studio only has OpenMP 2.0)
                omp_stub_header = blis_src_dir / 'omp_stub.h'
                omp_stub_content = '''/* Stub implementations for OpenMP 3.0+ functions missing in Visual Studio OpenMP 2.0 */
#ifndef OMP_STUB_H
#define OMP_STUB_H

#include <omp.h>

#ifdef __cplusplus
extern "C" {
#endif

/* These functions were added in OpenMP 3.0 */
static inline int omp_get_active_level(void) {
    /* Return 0 for top level (not inside a parallel region), 
       Visual Studio OpenMP doesn't support nested parallelism anyway */
    return (omp_in_parallel() ? 1 : 0);
}

static inline int omp_get_max_active_levels(void) {
    /* Return 1 since Visual Studio OpenMP doesn't support nested parallelism */
    return 1;
}

static inline void omp_set_max_active_levels(int max_levels) {
    /* No-op since Visual Studio OpenMP doesn't support nested parallelism */
    (void)max_levels;
}

#ifdef __cplusplus
}
#endif

#endif /* OMP_STUB_H */
'''
                omp_stub_header.write_text(omp_stub_content)
                print(f'  ✓ Created OpenMP 3.0 stub header at {omp_stub_header}')
                
                insert_code = f'''{marker}
    # Add Visual Studio OpenMP include directory for Clang compatibility
    include_directories("{openmp_include_cmake}")
    # Ensure BLIS_ENABLE_OPENMP is defined at compile time
    add_definitions(-DBLIS_ENABLE_OPENMP)
    # Force include our OpenMP stub header that provides OpenMP 3.0 functions
    add_compile_options(-include "${{CMAKE_CURRENT_SOURCE_DIR}}/omp_stub.h")
    message(STATUS "Added OpenMP include directory: {openmp_include_cmake}")
    message(STATUS "Added BLIS_ENABLE_OPENMP preprocessor define")
    message(STATUS "Added force-include of OpenMP 3.0 stub header")'''
                content = content.replace(marker, insert_code)
                print('  ✓ Added OpenMP include directory, preprocessor define, and force-include')
            else:
                print('  WARNING: Could not find FindOpenMP marker')
            
            blis_cmake.write_text(content)
            print('✓ Successfully patched BLIS CMakeLists.txt')
        else:
            print(f'ERROR: BLIS CMakeLists.txt not found')
            return False
        
        # Configure AOCL build with our patched BLIS
        os.chdir(aocl_dir)
        print(f'Configuring AOCL build with CMake (using patched BLIS and Clang from {rocm_path})...')
        
        # Set CFLAGS and CXXFLAGS to include OpenMP headers
        import_env = os.environ.copy()
        import_env['CFLAGS'] = f'-I"{openmp_include}"'
        import_env['CXXFLAGS'] = f'-I"{openmp_include}"'
        
        cmake_cmd = [
            'cmake',
            '-S', '.',
            '-B', 'build',
            '-G', 'Ninja',
            '-DCMAKE_BUILD_TYPE=Release',
            f'-DCMAKE_C_COMPILER={clang_exe}',
            f'-DCMAKE_CXX_COMPILER={clangxx_exe}',
            f'-DCMAKE_C_FLAGS=-I"{openmp_include}" -fopenmp=libomp',  # Force include path and OpenMP support
            f'-DCMAKE_CXX_FLAGS=-I"{openmp_include}" -fopenmp=libomp',  # Force include path and OpenMP support
            '-DBUILD_SHARED_LIBS=OFF',
            '-DENABLE_ILP64=ON',
            '-DENABLE_AOCL_BLAS=ON',
            '-DENABLE_AOCL_UTILS=OFF',  # Disable Utils - not needed for BLAS
            '-DENABLE_AOCL_LAPACK=OFF',
            '-DENABLE_MULTITHREADING=ON',  # Enable multithreading with OpenMP
            f'-DOpenMP_libomp_LIBRARY={openmp_lib}',  # Use Visual Studio's OpenMP
            f'-DOpenMP_C_FLAGS=-fopenmp=libomp',
            f'-DOpenMP_CXX_FLAGS=-fopenmp=libomp',
            f'-DBLAS_PATH={blis_src_dir.parent}',  # Use our patched BLIS
            '-DAOCL_BLIS_ENABLE_TESTS=OFF',  # Skip tests for faster build
            f'-DCMAKE_INSTALL_PREFIX={install_prefix}'
        ]
        
        result = subprocess.run(cmake_cmd, capture_output=True, text=True, env=import_env)
        if result.returncode != 0:
            print(f'Error configuring AOCL:')
            print(result.stdout)
            print(result.stderr)
            return False
        
        print('Building AOCL (this may take 5-10 minutes)...')
        build_cmd = ['cmake', '--build', 'build', '--config', 'release', '-j', '--target', 'install']
        result = subprocess.run(build_cmd, capture_output=True, text=True, env=import_env)
        if result.returncode != 0:
            print(f'Error building AOCL:')
            print(result.stdout)
            print(result.stderr)
            return False
        
        print(f'✓ AOCL 5.2 successfully built with ILP64 support (static)')
        print(f'  Location: {install_prefix / "lib" / "libaocl.lib"}')
        return True
        
    finally:
        # Always return to original directory
        os.chdir(cwd)

def install_deps( os_node ):
    global var_subs

    cwd = pathlib.Path.absolute(pathlib.Path(os.curdir))

    # Handle AOCL build (Windows only, similar to install.sh on Linux)
    if os.name == "nt":
        aocl_node = os_node.getElementsByTagName('aocl')
        if aocl_node:
            for a in aocl_node[0].getElementsByTagName('build'):
                if a.getAttribute('enabled') == 'true':
                    build_aocl_windows()

    if os.name == "nt":
        vc_node = os_node.getElementsByTagName('vcpkg')
        if vc_node:
            cmdline = "cd %IDIR%"
            cd_vcpkg = cmdline.replace('%IDIR%', args.install_dir)
            run_cmd(cd_vcpkg)
            for p in vc_node[0].getElementsByTagName('pkg'):
                name = p.getAttribute('name')
                package = p.firstChild.data
                if name:
                    print( f'***\n*** VCPKG Installing: {name}\n***' )
                raw_cmd = p.firstChild.data
                var_cmd = raw_cmd.format_map(var_subs)
                error = run_cmd( f'vcpkg.exe install {var_cmd}')
    else:
        create_dir( args.install_dir )
        # TODO

    os.chdir(cwd)

    pip_node = os_node.getElementsByTagName('pip')
    if pip_node:
        for p in pip_node[0].getElementsByTagName('pkg'):
            name = p.getAttribute('name')
            package = p.firstChild.data
            if name:
                print( f'***\n*** Pip Installing: {name}\n***' )
            raw_cmd = p.firstChild.data
            var_cmd = raw_cmd.format_map(var_subs)
            error = run_cmd( f'pip install {var_cmd}')
        for p in pip_node[0].getElementsByTagName('req'):
            name = p.getAttribute('name')
            package = p.firstChild.data
            if name:
                print( f'***\n*** Pip Requirements: {name}\n***' )
            raw_cmd = p.firstChild.data
            var_cmd = raw_cmd.format_map(var_subs)
            requirements_file = os.path.abspath( os.path.join( cwd, var_cmd ) )
            error = run_cmd( f'pip install -r {requirements_file}')

def run_install_script(script, xml):
    '''executes a simple batch style install script, the scripts are defined at top of file'''
    global OS_info
    global args
    global var_subs
    #
    cwd = pathlib.Path.absolute(pathlib.Path(os.getcwd()))

    fail = False
    last_cmd_index = 0
    for i in range(len(script)):
        last_cmd_index = i
        cmdline = script[i]
        cmd = cmdline.replace('%IDIR%', args.install_dir)
        if cmd.startswith('tdir '):
            if pathlib.Path(cmd[5:]).exists():
                break # all further cmds skipped
            else:
                continue
        error = False
        if cmd.startswith('%XML%'):
            fileversion = xml.getElementsByTagName('fileversion')
            if len(fileversion) == 0:
                print("WARNING: Could not find the version of this xml configuration file.")
            elif len(fileversion) > 1:
                print("WARNING: Multiple version tags found.")
            else:
                version = float(fileversion[0].firstChild.data)
                if version > SCRIPT_VERSION:
                    print(f"ERROR: This xml requires script version >= {version}, running script version {SCRIPT_VERSION}")
                    exit(1)

            for var in xml.getElementsByTagName('var'):
                name = var.getAttribute('name')
                if var.hasAttribute('value'):
                    val = var.getAttribute('value')
                elif var.firstChild is not None:
                    val = var.firstChild.data
                else:
                    val = ""
                var_subs[name] = val

            for os_node in xml.getElementsByTagName('os'):
                os_names = os_node.getAttribute('names')
                os_list = os_names.split(',')
                if (OS_info['ID'] in os_list) or ("all" in os_list):
                    error = install_deps( os_node )
        else:
            error = run_cmd(cmd)
        fail = fail or error
        if fail:
            break

    os.chdir( cwd )
    if (fail):
        if (script[last_cmd_index] == "%XML%"):
            print(f"FAILED xml dependency installation!")
        else:
            print(f"ERROR running: {script[last_cmd_index]}")
        return 1
    else:
        return 0

def install_msgpack_from_source():
    """Install msgpack-c 3.0.1 from source on Windows (Boost-free, same as Linux)"""
    build_dir = pathlib.Path.cwd() / "build"
    msgpack_dir = build_dir / "deps" / "msgpack-c"
    
    # Check if already built successfully by verifying the config file exists
    msgpack_config = msgpack_dir / "install" / "lib" / "cmake" / "msgpack-cxx" / "msgpack-cxx-config.cmake"
    if msgpack_config.exists():
        print(f"msgpack-c already installed at {msgpack_dir}")
        return 0
    
    print("Installing msgpack-c 3.0.1 from source (Boost-free)...")
    
    # Create deps directory
    deps_dir = build_dir / "deps"
    deps_dir.mkdir(parents=True, exist_ok=True)
    
    cwd = pathlib.Path.cwd()
    
    try:
        os.chdir(deps_dir)
        
        # Clone msgpack-c 3.0.1 (same version as Linux)
        print("Cloning msgpack-c 3.0.1...")
        run_cmd("git -c advice.detachedHead=false clone --quiet --depth 1 --branch cpp-3.0.1 https://github.com/msgpack/msgpack-c.git")
        os.chdir("msgpack-c")
        
        # Configure and install msgpack-c (Boost-free C++ package)
        print("Configuring msgpack-c...")
        run_cmd("cmake -DMSGPACK_BUILD_TESTS=OFF -DMSGPACK_BUILD_EXAMPLES=OFF -DMSGPACK_USE_BOOST=OFF -DCMAKE_INSTALL_PREFIX=install .")
        
        print("Installing msgpack-c...")
        run_cmd("cmake --build . --config Release --target install")
        
        print(f"✓ msgpack-c 3.0.1 installed successfully (Boost-free)")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"ERROR installing msgpack-c (subprocess failed): {e}")
        return 1
    except OSError as e:
        print(f"ERROR installing msgpack-c (OS error): {e}")
        return 1
    finally:
        os.chdir(cwd)

def installation():
    global vcpkg_script
    global xml_script
    global xmlDoc

    # install
    cwd = os.getcwd()

    xmlPath = os.path.join( cwd, 'rdeps.xml')
    xmlDoc = minidom.parse( xmlPath )

    scripts = []

    if os.name == "nt" and xmlDoc.getElementsByTagName('vcpkg'):
        scripts.append( vcpkg_script )
    scripts.append( xml_script )

    for i in scripts:
        if (run_install_script(i, xmlDoc)):
            #print("Failure in script. ABORTING")
            os.chdir( cwd )
            return 1
    
    # Install msgpack from source on Windows (Boost-free)
    if os.name == "nt":
        if install_msgpack_from_source():
            print("Failed to install msgpack-c")
            os.chdir( cwd )
            return 1
    
    os.chdir( cwd )
    return 0

def main():
    global args

    os_detect()
    args = parse_args()

    if os.name == "nt" and not args.install_dir:
        vcpkg_root = os.getenv( 'VCPKG_PATH', "C:\\github\\vcpkg")
        args.install_dir = vcpkg_root

    installation()

if __name__ == '__main__':
    main()
