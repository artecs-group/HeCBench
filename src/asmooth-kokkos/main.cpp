#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#include "reference.hpp"

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

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
  if (argc != 5) {
     printf("./%s <image dimension> <threshold> <max box size> <iterations>\n", argv[0]);
     exit(1);
  }

  // only a square image is supported
  const int Lx = atoi(argv[1]);
  const int Ly = Lx;
  const int size = Lx * Ly;

  const int Threshold = atoi(argv[2]);
  const int MaxRad = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  const size_t size_bytes = size * sizeof(float);
  const size_t box_bytes = size * sizeof(int);
 
  // input image
  float *img = (float*) malloc (size_bytes);

  // host and device results
  float *norm = (float*) malloc (size_bytes);
  float *h_norm = (float*) malloc (size_bytes);

  int *box = (int*) malloc (box_bytes);
  int *h_box = (int*) malloc (box_bytes);

  float *out = (float*) malloc (size_bytes);
  float *h_out = (float*) malloc (size_bytes);

  srand(123);
  for (int i = 0; i < size; i++) {
    img[i] = rand() % 256;
    norm[i] = box[i] = out[i] = 0;
  }

  Kokkos::View<float*, Layout, MemSpace> d_img = Kokkos::View<float*, Layout, MemSpace>("d_img", size);
  Kokkos::View<float*, Layout, MemSpace> d_norm = Kokkos::View<float*, Layout, MemSpace>("d_norm", size);
  Kokkos::View<int*, Layout, MemSpace> d_box = Kokkos::View<int*, Layout, MemSpace>("d_box", size);
  Kokkos::View<float*, Layout, MemSpace> d_out = Kokkos::View<float*, Layout, MemSpace>("d_out", size);

  const size_t lws = 32;
  const size_t lw = 16;
  const size_t gw = (Lx*Ly+(lws-1))/lws;

  Kokkos::View<const float*, Layout, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> vimg(img, size);
  Kokkos::View<const float*, Layout, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> vnorm(norm, size);
  typedef Kokkos::View<float*, ExecSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>> shared_float;
  size_t shared_size = shared_float::shmem_size(1024);

  double time = 0;

  for (int i = 0; i < repeat; i++) {
    // restore input image
    Kokkos::deep_copy(d_img, vimg);
    // reset norm
    Kokkos::deep_copy(d_norm, vnorm);

    auto start = std::chrono::steady_clock::now();

    // launch three kernels
    Kokkos::parallel_for("smoothing", 
    Kokkos::TeamPolicy<ExecSpace>(gw, lws)
    .set_scratch_size(0, Kokkos::PerTeam(shared_size)), 
    KOKKOS_LAMBDA(Kokkos::TeamPolicy<ExecSpace>::member_type memT){
      int blockDim_x = lws;
      int blockDim_y = lws;
      int stid = memT.team_rank();
      int tid = stid % lws;
      int tjd = stid / lws;
      int gtid = memT.league_rank() * memT.team_size() + memT.team_rank();
      int i = gtid % Lx;
      int j = gtid / Lx;
      shared_float s_Img(memT.team_scratch(0), 1024);

      //part of shared memory may be unused
      if ( i < Lx && j < Ly )
        s_Img[stid] = d_img[gtid];
      
      memT.team_barrier();

      if ( i < Lx && j < Ly ){
        // Smoothing parameters
        float sum = 0.f;
        int q = 1;
        int s = q;
        int ksum = 0;

        // Continue until parameters are met
        while (sum < Threshold && q < MaxRad)
        {
          s = q;
          sum = 0.f;
          ksum = 0;

          // Normal adaptive smoothing
          for (int ii = -s; ii < s+1; ii++)
            for (int jj = -s; jj < s+1; jj++)
              if ( (i-s >= 0) && (i+s < Ly) && (j-s >= 0) && (j+s < Lx) )
              {
                ksum++;
                // Compute within bounds of block dimensions
                if( tid-s >= 0 && tid+s < blockDim_x && tjd-s >= 0 && tjd+s < blockDim_y )
                  sum += s_Img[stid + ii*blockDim_x + jj];
                // Compute block borders with global memory
                else
                  sum += d_img[gtid + ii*Lx + jj];
              }
          q++;
        }
        d_box[gtid] = s;

        // Normalization for each box
        for (int ii = -s; ii < s+1; ii++)
          for (int jj = -s; jj < s+1; jj++)
            if (ksum != 0) {
              Kokkos::atomic_add(&d_norm[gtid + ii*Lx + jj], 1.f/ksum);
            }
      }
    });

    Kokkos::parallel_for("normalize", 
    Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>> ({0,0}, {(Ly+(lw-1))/lw*lw, (Lx+(lw-1))/lw*lw}, {lw,lw}), 
    KOKKOS_LAMBDA(const int j, const int i){
      if ( i < Lx && j < Ly ) {
        int gtid = j * Lx + i;
        const float norm = d_norm[gtid];
        if (norm != 0) d_img[gtid] = d_img[gtid]/norm;
      }
    });

    Kokkos::fence();

    Kokkos::parallel_for("output", 
    Kokkos::TeamPolicy<ExecSpace>(gw, lws)
    .set_scratch_size(0, Kokkos::PerTeam(shared_size)), 
    KOKKOS_LAMBDA(Kokkos::TeamPolicy<ExecSpace>::member_type memT){
      int blockDim_x = lws;
      int blockDim_y = lws;
      int stid = memT.team_rank();
      int tid = stid % lws;
      int tjd = stid / lws;
      int gtid = memT.league_rank() * memT.team_size() + memT.team_rank();
      int i = gtid % Lx;
      int j = gtid / Lx;
      shared_float s_Img(memT.team_scratch(0), 1024);

      //part of shared memory may be unused
      if ( i < Lx && j < Ly )
        s_Img[stid] = d_img[gtid];

      memT.team_barrier();

      if ( i < Lx && j < Ly )
      {
        const int s = d_box[gtid];
        float sum = 0.f;
        int ksum  = 0;

        for (int ii = -s; ii < s+1; ii++)
          for (int jj = -s; jj < s+1; jj++)
            if ( (i-s >= 0) && (i+s < Lx) && (j-s >= 0) && (j+s < Ly) )
            {
              ksum++;
              if( tid-s >= 0 && tid+s < blockDim_x && tjd-s >= 0 && tjd+s < blockDim_y )
                sum += s_Img[stid + ii*blockDim_y + jj];
              else
                sum += d_img[gtid + ii*Ly + jj];
            }
        if ( ksum != 0 ) d_out[gtid] = sum / (float)ksum;
      }
    });

    Kokkos::fence();
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  printf("Average filtering time %lf (s)\n", (time * 1e-9) / repeat);

  {
    Kokkos::View<float*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> vout(out, size);
    Kokkos::View<float*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> vnorm(norm, size);
    Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> vbox(box, size);
    Kokkos::deep_copy(vout, d_out);
    Kokkos::deep_copy(vbox, d_box);
    Kokkos::deep_copy(vnorm, d_norm);
  }

  // verify
  reference (Lx, Ly, Threshold, MaxRad, img, h_box, h_norm, h_out);
  verify(size, MaxRad, norm, h_norm, out, h_out, box, h_box);

  free(img);
  free(norm);
  free(h_norm);
  free(box);
  free(h_box);
  free(out);
  free(h_out);
  }
  Kokkos::finalize();
  return 0;
}
