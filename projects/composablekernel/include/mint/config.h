#pragma once

#ifdef MINT_BACKEND_ROCM
#include <mint/config_rocm.h>
#elif defined(MINT_BACKEND_CUDA)
#include <mint/config_cuda.h>
#else
#error "MINT backend not supported."
#endif
