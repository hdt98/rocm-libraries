################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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
import os
import sys

try:
    import xdist  # noqa
except ImportError:
    @pytest.fixture(scope="session")
    def worker_id():
        return None

testdir = os.path.dirname(__file__)
moddir = os.path.dirname(testdir)
rootdir = os.path.dirname(moddir)
sys.path.append(rootdir)

def pytest_addoption(parser):
    parser.addoption("--tensile-options")
    parser.addoption("--global-parameters")
    parser.addoption("--prebuilt-client")
    parser.addoption("--no-common-build", action="store_true")
    parser.addoption("--builddir", "--client-dir")
    parser.addoption("--timing-file", default=None)
    parser.addoption("--yaml-glob", help="Glob pattern for YAML files to run.")

@pytest.fixture(scope="session")
def timing_path(pytestconfig, tmpdir_factory):
    userDir = pytestconfig.getoption("--timing-file")
    if userDir is not None:
        return userDir
    return str(tmpdir_factory.mktemp("results") / "timing.csv")

@pytest.fixture(autouse=True)
def incremental_timer(request, worker_lock_instance, timing_path):
    #import pdb
    #pdb.set_trace()
    import datetime
    start = datetime.datetime.now()
    yield
    stop = datetime.datetime.now()

    testtime = stop - start
    with worker_lock_instance:
        with open(timing_path, "a") as f:
            f.write("{},{}\n".format(request.node.name, testtime.total_seconds()))

@pytest.fixture(scope="session")
def builddir(pytestconfig, tmpdir_factory):
    userDir = pytestconfig.getoption("--builddir")
    if userDir is not None:
        return userDir
    return str(tmpdir_factory.mktemp("0_Build"))

@pytest.fixture(scope="session")
def worker_lock_path(tmp_path_factory, worker_id):
    if not worker_id:
        return None

    return tmp_path_factory.getbasetemp().parent / "client_execution.lock"

@pytest.fixture
def tensile_script_path():
    return os.path.join(moddir, 'bin', 'Tensile')

@pytest.fixture
def worker_lock_instance(worker_lock_path):
    if not worker_lock_path:
        return open(os.devnull)

    import filelock
    return filelock.FileLock(str(worker_lock_path))

@pytest.fixture
def tensile_args(pytestconfig, builddir, worker_lock_path):
    rv = []
    if worker_lock_path:
        rv += ["--client-lock", str(worker_lock_path)]

    extraOptions = pytestconfig.getoption("--tensile-options")
    if extraOptions is not None:
        rv += extraOptions.split(",")
    if pytestconfig.getoption("--global-parameters"):
        rv += ["--global-parameters", pytestconfig.getoption("--global-parameters")]
    if not pytestconfig.getoption("--no-common-build"):
        if pytestconfig.getoption("--prebuilt-client"):
            rv += ["--prebuilt-client", pytestconfig.getoption("--prebuilt-client")]

    return rv

def pytest_collection_modifyitems(items):
    """
    Mainly for tests that aren't simple YAML files (including unit tests).
    Adds a mark for the root directory name to each test.
    """
    for item in items:
        
        relpath = item.fspath.relto(testdir)
        components = relpath.split(os.path.sep)
        # print(f"Items: {item}, Testdir: {testdir}, Components: {components}")
        if len(components) > 0 and len(components[0]) > 0:
            item.add_marker(getattr(pytest.mark, components[0]))

@pytest.fixture
def useGlobalParameters(tensile_args):
    from Tensile import Common
    from Tensile import Tensile
    import argparse

    class gpUpdater:
        def __init__(self, **params):
            self.additionalParams = params

        def __enter__(self):
            argParser = argparse.ArgumentParser()
            Tensile.addCommonArguments(argParser)
            args = argParser.parse_known_args(tensile_args)[0]

            Common.restoreDefaultGlobalParameters()
            if args.CxxCompiler:
                Common.globalParameters["CxxCompiler"] = args.CxxCompiler
            isa = Common.detectGlobalCurrentISA(args.device)
            Common.assignGlobalParameters({}, isa)

            overrideParameters = Tensile.argUpdatedGlobalParameters(args)
            for key, value in overrideParameters.items():
                Common.globalParameters[key] = value

            for key, value in self.additionalParams.items():
                Common.globalParameters[key] = value

            return Common.globalParameters

        def __exit__(self, exc_type, exc_value, traceback):
            Common.restoreDefaultGlobalParameters()

    return gpUpdater

import glob
import yaml
from Tensile.Tests.common.test_config import findConfigs, findAvailableArchs, configMarks

def pytest_generate_tests(metafunc):
    if 'config' in metafunc.fixturenames:
        glob_pattern = metafunc.config.getoption("--yaml-glob")
        if glob_pattern:
            # User has provided a specific pattern for fine-grained testing.
            rootDir = os.path.dirname(os.path.dirname(__file__))
            printRoot = os.path.dirname(os.path.dirname(rootDir))
            availableArchs = findAvailableArchs()
            
            all_files = glob.glob(glob_pattern, recursive=True)
            
            params = []
            for filepath in all_files:
                relpath = os.path.relpath(filepath, printRoot)
                with open(filepath) as f:
                    doc = yaml.safe_load(f)
                
                # Basic marks from file path
                marks = configMarks(filepath, rootDir, availableArchs)
                
                global_params = doc.get("GlobalParameters", {})
                benchmark_problem_groups = doc.get("BenchmarkProblems", [])

                for group_idx, problem_group in enumerate(benchmark_problem_groups):
                    if not problem_group or len(problem_group) < 2:
                        continue
                    
                    problem_type = problem_group[0]
                    benchmark_size_groups = problem_group[1:]

                    for size_group_idx, size_group in enumerate(benchmark_size_groups):
                        # Create a new config for each problem size group
                        new_benchmark_problem_group = [problem_type, size_group]
                        problem_config = {
                            "GlobalParameters": global_params,
                            "BenchmarkProblems": [new_benchmark_problem_group]
                        }
                        
                        # Create a unique ID for each test case
                        problem_id = f"{relpath}::{group_idx}:{size_group_idx}"
                        
                        params.append(pytest.param(problem_config, marks=marks, id=problem_id))
            
            metafunc.parametrize("config", params)
        else:
            # Default behavior: run all tests found by findConfigs (file-based)
            metafunc.parametrize("config", findConfigs())

