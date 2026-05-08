"""
This dictionary is used to map specific file directory changes to the corresponding build flag and tests
"""

import os
import sys
from pathlib import Path

# TheRock is checked out at ./TheRock during CI
THEROCK_DIR = Path.cwd() / "TheRock"


def get_test_dependencies_from_therock(component_names):
    """
    Get test dependencies using TheRock's determine_rocm_test_dependencies.py.
    Returns None if TheRock or the script is not available.
    """
    test_tools_path = THEROCK_DIR / "test_tools"
    if not test_tools_path.exists():
        return None

    sys.path.insert(0, str(test_tools_path))
    try:
        from determine_rocm_test_dependencies import get_rocm_test_dependencies

        return get_rocm_test_dependencies(component_names, THEROCK_DIR)
    except ImportError:
        return None
    finally:
        if str(test_tools_path) in sys.path:
            sys.path.remove(str(test_tools_path))


subtree_to_project_map = {
    "dnn-providers/fusilli-provider": "fusilli-provider",
    "dnn-providers/hipblaslt-provider": "hipblaslt-provider",
    "dnn-providers/hip-kernel-provider": "hip-kernel-provider",
    "dnn-providers/miopen-provider": "miopen-provider",
    "dnn-providers/integration-tests": "dnn-provider-integration-tests",
    "projects/composablekernel": "miopen",
    "projects/hipblas": "blas",
    "projects/hipblas-common": "blas",
    "projects/hipblaslt": "blas",
    "projects/hipcub": "prim",
    "projects/hipdnn": "hipdnn",
    "projects/hipfft": "fft",
    "projects/hiprand": "rand",
    "projects/hipsolver": "solver",
    "projects/hipsparse": "sparse",
    "projects/hipsparselt": "sparselt",
    "projects/miopen": "miopen",
    "projects/rocblas": "blas",
    "projects/rocfft": "fft",
    "projects/rocprim": "prim",
    "projects/rocrand": "rand",
    "projects/rocsolver": "solver",
    "projects/rocsparse": "sparse",
    "projects/rocthrust": "prim",
    "projects/rocwmma": "rocwmma",
    "shared/mxdatagenerator": "blas",
    "shared/origami": "blas",
    "shared/rocroller": "blas",
    "shared/stinkytofu": "blas",
    "shared/tensile": "blas",
}

project_map = {
    "prim": {
        "cmake_options": ["-DTHEROCK_ENABLE_PRIM=ON"],
        "projects_to_test": ["rocprim", "rocthrust", "hipcub"],
    },
    "rand": {
        "cmake_options": ["-DTHEROCK_ENABLE_RAND=ON"],
        "projects_to_test": ["rocrand", "hiprand"],
    },
    "blas": {
        "cmake_options": ["-DTHEROCK_ENABLE_BLAS=ON"],
        "projects_to_test": ["hipblaslt", "rocblas", "hipblas", "rocroller"],
    },
    "miopen": {
        "cmake_options": [
            "-DTHEROCK_ENABLE_MIOPEN=ON",
            "-DTHEROCK_ENABLE_MIOPENPROVIDER=ON",
            "-DTHEROCK_ENABLE_COMPOSABLE_KERNEL=ON",
        ],
        "projects_to_test": ["miopen", "miopenprovider"],
    },
    "fft": {
        "cmake_options": ["-DTHEROCK_ENABLE_FFT=ON", "-DTHEROCK_ENABLE_RAND=ON"],
        "projects_to_test": ["hipfft", "rocfft"],
    },
    "hip-kernel-provider": {
        "cmake_options": [
            "-DTHEROCK_ENABLE_HIPKERNELPROVIDER=ON",
            "-DHIP_KERNEL_PROVIDER_ENABLE=ON",
        ],
        "projects_to_test": ["hipkernelprovider"],
    },
    "dnn-provider-integration-tests": {
        "cmake_options": [
            "-DTHEROCK_ENABLE_HIPDNN_INTEGRATION_TESTS=ON",
        ],
        "projects_to_test": ["hipdnn-integration-tests"],
    },
    "fusilli-provider": {
        "cmake_options": ["-DTHEROCK_ENABLE_IREE_LIBS=ON"],
        "projects_to_test": ["fusilliprovider"],
    },
}

# For certain math components, they are optional during building and testing.
# As they are optional, we do not want to include them as default as this takes more time in the CI.
# However, if we run a separate build for optional components, those files will be overriden as these components share the same umbrella as other projects
# Example: SPARSE is included in BLAS, but a separate build would cause overwriting of the blas_lib.tar.xz and blas_test.tar.xz and be missing libraries and tests
additional_options = {
    "sparse": {
        "cmake_options": ["-DTHEROCK_ENABLE_SPARSE=ON"],
        "projects_to_test": ["rocsparse", "hipsparse"],
        "project_to_add": "blas",
    },
    "sparselt": {
        "cmake_options": ["-DTHEROCK_ENABLE_SPARSE=ON"],
        "projects_to_test": ["hipsparselt"],
        "project_to_add": "blas",
    },
    "solver": {
        "cmake_options": ["-DTHEROCK_ENABLE_SOLVER=ON"],
        "projects_to_test": ["rocsolver", "hipsolver"],
        "project_to_add": "blas",
    },
    "hipdnn": {
        "cmake_options": [
            "-DTHEROCK_ENABLE_HIPBLASLTPROVIDER=ON",
            "-DTHEROCK_ENABLE_HIPKERNELPROVIDER=ON",
            "-DHIP_KERNEL_PROVIDER_ENABLE=ON",
            "-DTHEROCK_ENABLE_MIOPENPROVIDER=ON",
            "-DTHEROCK_ENABLE_HIPDNN_SAMPLES=ON",
            "-DTHEROCK_ENABLE_COMPOSABLE_KERNEL=ON",
            "-DTHEROCK_ENABLE_HIPDNN_INTEGRATION_TESTS=ON",
            "-DTHEROCK_ENABLE_IREE_LIBS=ON",
        ],
        "projects_to_test": [
            "hipdnn",
            "hipdnn_install",
            "hipdnn-samples",
            "miopenprovider",
            "hipblasltprovider",
            "hipkernelprovider",
            "hipdnn-integration-tests",
            "fusilliprovider",
        ],
        "project_to_add": "miopen",
    },
    "miopen-provider": {
        "cmake_options": [
            "-DTHEROCK_ENABLE_MIOPENPROVIDER=ON",
            "-DTHEROCK_ENABLE_COMPOSABLE_KERNEL=ON",
        ],
        "projects_to_test": ["miopenprovider"],
        "project_to_add": "miopen",
    },
    "hipblaslt-provider": {
        "cmake_options": [
            "-DTHEROCK_ENABLE_HIPBLASLTPROVIDER=ON",
        ],
        "projects_to_test": ["hipblasltprovider"],
        "project_to_add": "blas",
    },
    "rocwmma": {
        "cmake_options": ["-DTHEROCK_ENABLE_ROCWMMA=ON"],
        "projects_to_test": ["rocwmma"],
        "project_to_add": "blas",
    },
}

# If a project has dependencies that are also being built, we combine build options and test options
# This way, there will be no S3 upload overlap and we save redundant builds
dependency_graph = {
    "miopen": ["blas", "rand", "fusilli-provider"],
}


def extract_component_names_from_subtrees(subtrees):
    """Extract component names from subtree paths.

    E.g., 'projects/rocprim' -> 'rocprim', 'shared/tensile' -> 'tensile'
    """
    components = []
    for subtree in subtrees:
        # Extract the last part of the path as the component name
        parts = subtree.split("/")
        if len(parts) >= 2:
            components.append(parts[-1])
    return components


def collect_projects_to_run(subtrees):
    platform = os.getenv("PLATFORM")
    projects = set()
    # collect the associated subtree to project
    for subtree in subtrees:
        if subtree in subtree_to_project_map:
            projects.add(subtree_to_project_map.get(subtree))

    # For each component, get tests from TheRock's script.
    # If TheRock returns meaningful deps (more than just the component itself),
    # use them; otherwise fall back to project_map.
    component_names = extract_component_names_from_subtrees(subtrees)
    tests_per_component = {}
    components_needing_fallback = set()

    for component in component_names:
        deps = get_test_dependencies_from_therock([component])
        # Only use TheRock deps if it returns more than just the component itself
        if deps and set(deps) != {component}:
            tests_per_component[component] = deps
        else:
            components_needing_fallback.add(component)

    for project in list(projects):
        # Check if an optional math component was included.
        if project in additional_options:
            project_options_to_add = additional_options[project]

            project_to_add = project_options_to_add["project_to_add"]
            # If `project_to_add` is in included, add options to the existing `project_map` entry
            if project_to_add in projects:
                project_map[project_to_add]["cmake_options"].extend(
                    project_options_to_add["cmake_options"]
                )
                project_map[project_to_add]["projects_to_test"].extend(
                    project_options_to_add["projects_to_test"]
                )
            # If `project_to_add` is not included, only run build and tests for the optional project
            else:
                projects.add(project_to_add)
                project_map[project_to_add]["cmake_options"] = project_options_to_add[
                    "cmake_options"
                ]
                project_map[project_to_add]["projects_to_test"] = (
                    project_options_to_add["projects_to_test"]
                )

    # Check for potential dependencies
    to_remove_from_project_map = []
    for project in list(projects):
        # Check if project has a dependency combine
        if project in dependency_graph:
            for dependency in dependency_graph[project]:
                # If the dependency is also included, let's combine to avoid overlap
                if dependency in projects:
                    project_map[project]["cmake_options"].extend(
                        project_map[dependency]["cmake_options"]
                    )
                    project_map[project]["projects_to_test"].extend(
                        project_map[dependency]["projects_to_test"]
                    )
                    to_remove_from_project_map.append(dependency)

    # if dependency is included in projects and parent is found, we delete the dependency as the parent will build and test
    for to_remove_item in to_remove_from_project_map:
        projects.remove(to_remove_item)
        del project_map[to_remove_item]

    # retrieve the subtrees to checkout, cmake options to build, and projects to test
    project_to_run = []
    for project in projects:
        if project in project_map:
            project_map_data = project_map.get(project)

            # Check if platform-based additional flags are needed
            if (
                "additional_flags" in project_map_data
                and platform in project_map_data["additional_flags"]
            ):
                project_map_data["cmake_options"].extend(
                    project_map_data["additional_flags"][platform]
                )

            # To save time, only build what is needed
            project_map_data["cmake_options"].extend(["-DTHEROCK_ENABLE_ALL=OFF"])
            # To ensure uniqueness of flags and tests
            project_map_data["cmake_options"] = list(
                set(project_map_data["cmake_options"])
            )

            # Collect tests: use TheRock deps for components that have them,
            # fall back to hardcoded for components that don't.
            tests_to_run = set()

            # Add test deps for components that have them from TheRock
            for component, deps in tests_per_component.items():
                tests_to_run.update(deps)

            # For components needing fallback, use hardcoded projects_to_test
            if components_needing_fallback:
                tests_to_run.update(project_map_data["projects_to_test"])

            project_map_data["projects_to_test"] = list(tests_to_run)

            cmake_flag_options = " ".join(project_map_data["cmake_options"])
            projects_to_test_options = ",".join(project_map_data["projects_to_test"])
            project_map_data["cmake_options"] = cmake_flag_options
            project_map_data["projects_to_test"] = projects_to_test_options
            project_to_run.append(project_map_data)

    return project_to_run
