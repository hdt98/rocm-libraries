// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/data_type.hpp"

template <bool KLast>
void preShuffleScaleBuffer(ck::e8m0_bexp_t* src, ck::e8m0_bexp_t* dst, int MN, int K)
{
    int MNXdlPack = 2;
    int KXdlPack  = 2;

    int XdlMNThread = 16;
    int XdlKThread  = 64 / XdlMNThread;

    int K0 = K / KXdlPack / XdlKThread; // KRepeat

    // The 4 16x128 building blocks will be packed into 1 32x256 for F4
    // The 8 16x16x128 mfma will be packed into 1 32x32x256 for F4

    // unfold the MN32xK(256/32) scale buffer
    //    4            16             2           2
    // To XdlKThread-> XdlMNThread -> KXdlPack -> MNXdlPack
    // Then, MNRepeat->KRepeat

    for(int n = 0; n < MN; ++n)
    {
        for(int k = 0; k < K; ++k)
        {
            int n0    = n / (XdlMNThread * MNXdlPack); // i MNRepeat
            int tempn = n % (XdlMNThread * MNXdlPack);
            int n1    = tempn % XdlMNThread; // i XdlMNThread
            int n2    = tempn / XdlMNThread; // i MNXdlPack

            int k0    = k / (XdlKThread * KXdlPack); // i KRepeat
            int tempk = k % (XdlKThread * KXdlPack);
            int k1    = tempk % XdlKThread; // i XdlKThread
            int k2    = tempk / XdlKThread; // i KXdlPack

            int outputIndex = n0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread * K0 +
                              k0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread +
                              k1 * MNXdlPack * KXdlPack * XdlMNThread + n1 * MNXdlPack * KXdlPack +
                              k2 * MNXdlPack + n2;

            if constexpr(KLast)
                dst[outputIndex] = src[n * K + k];
            else
                dst[outputIndex] = src[k * MN + n];
        }
    }
}

inline void preShuffleBuffer(const ck::f4x2_pk_t* src, ck::f4x2_pk_t* dst, int N, int K, int NXdl)
{
    int KPack = 16;
    int NLane = NXdl;
    int KLane = 64 / NLane;
    int K_pk  = K / 2;
    int K0    = K_pk / (KLane * KPack);
    // K -> K0 KLane KPack
    // N -> N0 NLane
    // N, K -> N0 K0 KLane NLane KPack
    int tempk;
    for(int n = 0; n < N; ++n)
    {
        for(int k = 0; k < K_pk; ++k)
        {
            int n0 = n / NLane;
            int n1 = n % NLane;

            int k0 = k / (KLane * KPack);
            tempk  = k % (KLane * KPack);
            int k1 = tempk / KPack;
            int k2 = tempk % KPack;

            int outputIndex = n0 * KPack * NLane * KLane * K0 + k0 * KPack * NLane * KLane +
                              k1 * KPack * NLane + n1 * KPack + k2;

            dst[outputIndex] = src[n * K_pk + k];
        }
    }
}
