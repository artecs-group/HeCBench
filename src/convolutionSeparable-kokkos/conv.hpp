/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef CONV_H
#define CONV_H

#include <Kokkos_Core.hpp>

#define KERNEL_RADIUS 8
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

#if defined(INTEL_GPU)
typedef Kokkos::Experimental::SYCL ExecSpace;
typedef Kokkos::Experimental::SYCLDeviceUSMSpace MemSpace;
#elif defined(NVIDIA_GPU)
typedef Kokkos::Cuda ExecSpace;
typedef Kokkos::CudaSpace MemSpace;
#else //CPU
typedef Kokkos::OpenMP ExecSpace;
typedef Kokkos::HostSpace MemSpace;
#endif

typedef Kokkos::LayoutRight Layout;
typedef Kokkos::View<float*, ExecSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>> shared_float;

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowHost(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR
);

void convolutionColumnHost(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR
);


void convolutionRows(
    Kokkos::View<float*, Layout, MemSpace> d_Dst,
    Kokkos::View<float*, Layout, MemSpace> d_Src,
    Kokkos::View<float*, Layout, MemSpace> c_Kernel,
    const int imageW,
    const int imageH,
    const int pitch
);

void convolutionColumns(
    Kokkos::View<float*, Layout, MemSpace> d_Dst,
    Kokkos::View<float*, Layout, MemSpace> d_Src,
    Kokkos::View<float*, Layout, MemSpace> c_Kernel,
    const int imageW,
    const int imageH,
    const int pitch
);

#endif
