// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
#include "convsuba_device_impl.h"

Def_Test_Func(half_t, float, float, 0x17, 0, 0, 0x1);
Def_Test_Func(half_t, float, float, 0x17, 1, 0, 0x1);
Def_Test_Func(half_t, float, float, 0x17, 0, 1, 0x1);

Def_Test_Func(half_t, half_t, half_t, 0x1f, 0, 0, 0x2);
Def_Test_Func(half_t, half_t, half_t, 0x1f, 1, 0, 0x2);
Def_Test_Func(half_t, half_t, half_t, 0x1f, 0, 1, 0x2);

Def_Test_Func(bhalf_t, bhalf_t, half_t, 0x17, 0, 0, 0x4);
Def_Test_Func(bhalf_t, bhalf_t, half_t, 0x17, 0, 1, 0x4);
Def_Test_Func(bhalf_t, bhalf_t, half_t, 0x17, 1, 0, 0x4);
