#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <Kokkos_Core.hpp>

#define BLOCK_SIZE 256

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
    printf("Usage: %s <image width> <image height> <merge> <repeat>\n", argv[0]);
    return 1;
  }

  const int width = atoi(argv[1]);
  const int height = atoi(argv[2]);
  const int merged = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  const int imgSize = width * height;
  const size_t imgSize_bytes = imgSize * sizeof(char);
  unsigned char *Img = (unsigned char*) malloc (imgSize_bytes);
  unsigned char *Bn = (unsigned char*) malloc (imgSize_bytes);
  unsigned char *Tn = (unsigned char*) malloc (imgSize_bytes);

  std::mt19937 generator (123);
  std::uniform_int_distribution<int> distribute( 0, 255 );

  for (int j = 0; j < imgSize; j++) {
    Bn[j] = distribute(generator);
    Tn[j] = 128;
  }

  Kokkos::View<unsigned char*, Layout, MemSpace> d_Img = Kokkos::View<unsigned char*, Layout, MemSpace>("d_Img", imgSize);
  Kokkos::View<unsigned char*, Layout, MemSpace> d_Img1 = Kokkos::View<unsigned char*, Layout, MemSpace>("d_Img1", imgSize);
  Kokkos::View<unsigned char*, Layout, MemSpace> d_Img2 = Kokkos::View<unsigned char*, Layout, MemSpace>("d_Img2", imgSize);
  Kokkos::View<unsigned char*, Layout, MemSpace> d_Bn = Kokkos::View<unsigned char*, Layout, MemSpace>("d_Bn", imgSize);
  Kokkos::View<unsigned char*, Layout, MemSpace> d_Mp = Kokkos::View<unsigned char*, Layout, MemSpace>("d_Mp", imgSize);
  Kokkos::View<unsigned char*, Layout, MemSpace> d_Tn = Kokkos::View<unsigned char*, Layout, MemSpace>("d_Tn", imgSize);
  Kokkos::View<unsigned char*, Layout, MemSpace> t = Kokkos::View<unsigned char*, Layout, MemSpace>("t", imgSize);

  Kokkos::View<unsigned char*, Layout, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> vBn(Bn, imgSize);
  Kokkos::View<unsigned char*, Layout, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> vTn(Tn, imgSize);
  Kokkos::View<unsigned char*, Layout, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> vImg(Img, imgSize);
  
  Kokkos::deep_copy(d_Bn, vBn);
  Kokkos::deep_copy(d_Tn, vTn);

  long time = 0;

  for (int i = 0; i < repeat; i++) {

    for (int j = 0; j < imgSize; j++) {
      Img[j] = distribute(generator);
    }

    Kokkos::deep_copy(d_Img, vImg);
    Kokkos::fence();

    // Time t   : Image   | Image1   | Image2
    // Time t+1 : Image2  | Image    | Image1
    // Time t+2 : Image1  | Image2   | Image
    t = d_Img2;
    d_Img2 = d_Img1;
    d_Img1 = d_Img;
    d_Img = t;

    if (i >= 2) {
      if (merged) {
        auto start = std::chrono::steady_clock::now();

        Kokkos::parallel_for("merged_kernel", 
        Kokkos::RangePolicy<ExecSpace> (0, imgSize),
        KOKKOS_LAMBDA(const int i){
          if (i >= imgSize) return;
          if ( Kokkos::abs(d_Img[i] - d_Img1[i]) <= d_Tn[i] && Kokkos::abs(d_Img[i] - d_Img2[i]) <= d_Tn[i] ) {
            // update background
            d_Bn[i] = 0.92f * d_Bn[i] + 0.08f * d_Img[i];

            // update threshold
            float th = 0.92f * d_Tn[i] + 0.24f * (d_Img[i] - d_Bn[i]);
            d_Tn[i] = Kokkos::fmax(th, 20.f);
          }
        });
        Kokkos::fence();

        auto end = std::chrono::steady_clock::now();
        time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      }
      else {
        auto start = std::chrono::steady_clock::now();
        Kokkos::parallel_for("k1", 
        Kokkos::RangePolicy<ExecSpace> (0, imgSize),
        KOKKOS_LAMBDA(const int i){
          if (i >= imgSize) return;
          if ( Kokkos::abs(d_Img[i] - d_Img1[i]) > d_Tn[i] || Kokkos::abs(d_Img[i] - d_Img2[i]) > d_Tn[i] )
            d_Mp[i] = 255;
          else {
            d_Mp[i] = 0;
          }
        });

        Kokkos::parallel_for("k2", 
        Kokkos::RangePolicy<ExecSpace> (0, imgSize),
        KOKKOS_LAMBDA(const int i){
          if (i >= imgSize) return;
          if ( d_Mp[i] == 0 ) d_Bn[i] = 0.92f * d_Bn[i] + 0.08f * d_Img[i];
        });

        Kokkos::parallel_for("k3", 
        Kokkos::RangePolicy<ExecSpace> (0, imgSize),
        KOKKOS_LAMBDA(const int i){
          if (i >= imgSize) return;
          if (d_Mp[i] == 0) {
            float th = 0.92f * d_Tn[i] + 0.24f * (d_Img[i] - d_Bn[i]);
            d_Tn[i] = Kokkos::fmax(th, 20.f);
          }
        });
        Kokkos::fence();

        auto end = std::chrono::steady_clock::now();
        time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      }
    }
  }

  float kernel_time = (repeat <= 2) ? 0 : (time * 1e-9f) / (repeat - 2);
  printf("Average kernel execution time: %f (s)\n", kernel_time);
  printf("Total kernel execution time: %f (s)\n", time * 1e-9f);

  Kokkos::deep_copy(vTn, d_Tn);

#ifdef VERIFY
  // verification
  int sum = 0;
  int bin[4] = {0, 0, 0, 0};
  for (int j = 0; j < imgSize; j++) {
    sum += abs(Tn[j] - 128);
    if (Tn[j] < 64)
      bin[0]++;
    else if (Tn[j] < 128)
      bin[1]++;
    else if (Tn[j] < 192)
      bin[2]++;
    else
      bin[3]++;
  }
  sum = sum / imgSize;
  printf("Average threshold change is %d\n", sum);
  printf("Bin counts are %d %d %d %d\n", bin[0], bin[1], bin[2], bin[3]);
#endif
  free(Img);
  free(Tn);
  free(Bn);
  }
  Kokkos::finalize();
  return 0;
}
