#ifndef KERNELS
#define KERNELS

#include <Kokkos_Core.hpp>

#if defined(INTEL_GPU)
typedef Kokkos::Experimental::SYCL ExecSpace;
typedef Kokkos::Experimental::SYCLDeviceUSMSpace MemSpace; //also available SYCLSharedUSMSpace
#elif defined(NVIDIA_GPU)
typedef Kokkos::Cuda ExecSpace;
typedef Kokkos::CudaSpace MemSpace;
#else //CPU
typedef Kokkos::OpenMP ExecSpace;
typedef Kokkos::HostSpace MemSpace;
#endif

typedef Kokkos::LayoutRight Layout;

void toneMapping(
    const Kokkos::View<float*, Layout, MemSpace> input,
    Kokkos::View<float*, Layout, MemSpace> output,
    const float averageLuminance,
    const float gamma,
    const float c,
    const float delta,
    const uint width,
    const uint numChannels,
    const uint height);

#endif