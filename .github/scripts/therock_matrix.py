"""
This dictionary is used to map specific file directory changes to the corresponding build flag and tests
"""
subtree_to_project_map = {
    "projects/hipblas": "blas",
    "projects/hipblas-common": "blas",
    "projects/hipblaslt": "blas",
    "projects/hipcub": "prim",
    "projects/hiprand": "rand",
    "projects/rocblas": "blas",
    "projects/rocprim": "prim",
    "projects/rocrand": "rand",
    "projects/rocthrust": "prim",
    "projects/rocsparse": "sparse",
    "projects/hipsparse": "sparse",
    "shared/mxdatagenerator": "blas",
    "shared/origami": "blas",
    "shared/rocroller": "blas",
    "shared/tensile": "blas"
}

project_map = {
    "prim": {
        "cmake_options": "-DTHEROCK_ENABLE_PRIM=ON -DTHEROCK_ENABLE_ALL=OFF",
        "project_to_test": "rocprim, rocthrust, hipcub",
    },
    "rand": {
        "cmake_options": "-DTHEROCK_ENABLE_RAND=ON -DTHEROCK_ENABLE_ALL=OFF",
        "project_to_test": "rocrand, hiprand",
    },
    "blas": {
        "cmake_options": "-DTHEROCK_ENABLE_BLAS=ON -DTHEROCK_ENABLE_ALL=OFF",
        "project_to_test": "hipblaslt, rocblas, hipblas",
        "subtree_checkout": "projects/hipblaslt\nprojects/hipblas-common\nprojects/rocblas\nprojects/hipblas\nshared/mxdatagenerator\nshared/rocroller\nshared/tensile",
    },
    "sparse": {
        "cmake_options": "-DTHEROCK_ENABLE_SPARSE=ON -DTHEROCK_ENABLE_ALL=OFF",
        "project_to_test": "rocsparse, hipsparse",
        "subtree_checkout": "projects/rocsparse\nprojects/hipsparse\nprojects/rocblas\nprojects/hipblas-common\nprojects/rocprim",
    }
}
