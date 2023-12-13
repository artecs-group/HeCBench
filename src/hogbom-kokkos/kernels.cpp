#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstddef>
#include <Kokkos_Core.hpp>
#include "kernels.h"
#include "timer.h"

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

// grids and blocks are constant for the findPeak kernel
#define findPeakNBlocks 128
#define findPeakWidth 256

struct Peak {
  size_t pos;
  float val;
};

struct Position {
    Position(int _x, int _y) : x(_x), y(_y) { };
  int x;
  int y;
};

static Position idxToPos(const size_t idx, const int width)
{
  const int y = idx / width;
  const int x = idx % width;
  return Position(x, y);
}

static size_t posToIdx(const int width, const Position& pos)
{
  return (pos.y * width) + pos.x;
}

static Peak findPeak(Kokkos::View<float*, Layout, MemSpace> d_image, Kokkos::View<Peak*, Layout, MemSpace> d_peak, size_t size)
{
  const int nBlocks = findPeakNBlocks;
  Peak peaks[nBlocks];

  // Initialise a peaks array on the device. Each thread block will return
  // a peak. Note:  the d_peaks array is not initialized (hence avoiding the
  // memcpy), it is up to the device function to do that

  const int gws{nBlocks};
  const int lws{findPeakWidth};

  // Find peak
  Kokkos::parallel_for("find_peak", 
  Kokkos::TeamPolicy<ExecSpace>(gws, lws)
  .set_scratch_size(0, Kokkos::PerTeam(sizeof(float)*findPeakWidth + sizeof(size_t)*findPeakWidth)), 
  KOKKOS_LAMBDA(Kokkos::TeamPolicy<ExecSpace>::member_type memT){
    int tid = memT.team_rank();
    int bid = memT.league_rank();
    const int column = memT.league_rank() * memT.team_size() + memT.team_rank();
    float* maxVal = (float*) memT.team_shmem().get_shmem(findPeakWidth*sizeof(float));
    size_t* maxPos = (size_t*) memT.team_shmem().get_shmem(findPeakWidth*sizeof(size_t));

    maxVal[tid] = 0.f;
    maxPos[tid] = 0;

    for (int idx = column; idx < size; idx += findPeakWidth*findPeakNBlocks) {
      if (Kokkos::fabs(d_image[idx]) > Kokkos::fabs(maxVal[tid])) {
        maxVal[tid] = d_image[idx];
        maxPos[tid] = idx;
      }
    }

    memT.team_barrier();

    if (tid == 0) {
      d_peak[bid].val = 0.f;
      d_peak[bid].pos = 0;
      for (int i = 0; i < findPeakWidth; ++i) {
        if (Kokkos::fabs(maxVal[i]) > Kokkos::fabs(d_peak[bid].val)) {
          d_peak[bid].val = maxVal[i];
          d_peak[bid].pos = maxPos[i];
        }
      }
    }
  });

  // Get the peaks array back from the device
  {
    Kokkos::View<Peak*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> vpeaks(peaks, nBlocks);
    Kokkos::deep_copy(vpeaks, d_peak);
  }

  // Each thread block returned a peak, find the absolute maximum
  Peak p;
  p.val = 0.f;
  p.pos = 0;
  for (int i = 0; i < nBlocks; ++i) {
    if (fabsf(peaks[i].val) > fabsf(p.val)) {
      p.val = peaks[i].val;
      p.pos = peaks[i].pos;
    }
  }

  return p;
}

static void subtractPSF(
  Kokkos::View<float*, Layout, MemSpace> d_psf, const int psfWidth,
  Kokkos::View<float*, Layout, MemSpace> d_residual, const int residualWidth,
  const size_t peakPos, const size_t psfPeakPos,
  const float absPeakVal, const float gain)
{
  const int blockDim = 16;

  const int rx = idxToPos(peakPos, residualWidth).x;
  const int ry = idxToPos(peakPos, residualWidth).y;

  const int px = idxToPos(psfPeakPos, psfWidth).x;
  const int py = idxToPos(psfPeakPos, psfWidth).y;

  const int diffx = rx - px;
  const int diffy = ry - px;

  const int startx = std::max(0, rx - px);
  const int starty = std::max(0, ry - py);

  const int stopx = std::min(residualWidth - 1, rx + (psfWidth - px - 1));
  const int stopy = std::min(residualWidth - 1, ry + (psfWidth - py - 1));

  Kokkos::parallel_for("subtract_PSF", 
  Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>> ({0,0}, {stopy-starty, stopx-startx}, {blockDim,blockDim}), 
  KOKKOS_LAMBDA(const int j, const int i){
    const int x = startx + i;
    const int y = starty + j;

    // thread blocks are of size 16, but the workload is not always a multiple of 16
    if (x <= stopx && y <= stopy) {
      
      d_residual[y * residualWidth + x] -= gain * absPeakVal
        * d_psf[(y - diffy) * psfWidth + (x - diffx)];
    }
  });
}

HogbomTest::HogbomTest()
{
}

HogbomTest::~HogbomTest()
{
}

void HogbomTest::deconvolve(const std::vector<float>& dirty,
    const size_t dirtyWidth,
    const std::vector<float>& psf,
    const size_t psfWidth,
    std::vector<float>& model,
    std::vector<float>& residual)
{

  residual = dirty;

  // Initialise a peaks array on the device. Each thread block will return
  // a peak. Note:  the d_peaks array is not initialized (hence avoiding the
  // memcpy), it is up to the device function to do that
  Kokkos::View<Peak*, Layout, MemSpace> d_peaks = Kokkos::View<Peak*, Layout, MemSpace>("d_peaks", findPeakNBlocks);

  const size_t psf_size = psf.size();
  const size_t residual_size = residual.size();

  Kokkos::View<const float*, Layout, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> vpsf(&psf[0], psf_size);
  Kokkos::View<float*, Layout, MemSpace> d_psf = Kokkos::View<float*, Layout, MemSpace>("d_psf", psf_size);
  Kokkos::deep_copy(d_psf, vpsf);

  Kokkos::View<const float*, Layout, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> vresidual(&residual[0], residual_size);
  Kokkos::View<float*, Layout, MemSpace> d_residual = Kokkos::View<float*, Layout, MemSpace>("d_residual", residual_size);
  Kokkos::deep_copy(d_residual, vresidual);

  // Find peak of PSF
  Peak psfPeak = findPeak(d_psf, d_peaks, psf_size);

  Kokkos::fence();
  Stopwatch sw;
  sw.start();

  std::cout << "Found peak of PSF: " << "Maximum = " << psfPeak.val 
    << " at location " << idxToPos(psfPeak.pos, psfWidth).x << ","
    << idxToPos(psfPeak.pos, psfWidth).y << std::endl;
  assert(psfPeak.pos <= psf_size);

  for (unsigned int i = 0; i < niters; ++i) {
    // Find peak in the residual image
    Peak peak = findPeak(d_residual, d_peaks, residual_size);

    assert(peak.pos <= residual_size);

    // Check if threshold has been reached
    if (fabsf(peak.val) < threshold) {
      std::cout << "Reached stopping threshold" << std::endl;
    }

    // Subtract the PSF from the residual image 
    subtractPSF(d_psf, psfWidth, d_residual, dirtyWidth, peak.pos, psfPeak.pos, peak.val, gain);

    // Add to model
    model[peak.pos] += peak.val * gain;
  }

  Kokkos::fence();
  const double time = sw.stop();

  // Report on timings
  std::cout << "    Time " << time << " (s) " << std::endl;
  std::cout << "    Time per cycle " << time / niters * 1000 << " (ms)" << std::endl;
  std::cout << "    Cleaning rate  " << niters / time << " (iterations per second)" << std::endl;
  std::cout << "Done" << std::endl;

  // Copy device arrays back into the host 
  {
    Kokkos::View<float*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> vresidual(&residual[0], residual.size());
    Kokkos::deep_copy(vresidual, d_residual);
  }
}
