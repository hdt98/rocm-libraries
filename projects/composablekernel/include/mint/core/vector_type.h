#pragma once
#include <mint/config.h>

#if defined(MINT_BACKEND_CUDA)
#include <mint/core/vector_type/vector_type_cuda.h>
#elif defined(MINT_BACKEND_ROCM)
#include <mint/core/vector_type/vector_type_rocm.h>
#else
#error "MINT backend not supported."
#endif
