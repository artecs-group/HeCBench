/***************************************************************************
 *
 *            (C) Copyright 2007 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

/*
 * C code for creating the Q data structure for fast convolution-based
 * Hessian multiplication for arbitrary k-space trajectories.
 *
 * Inputs:
 * kx - VECTOR of kx values, same length as ky and kz
 * ky - VECTOR of ky values, same length as kx and kz
 * kz - VECTOR of kz values, same length as kx and ky
 * x  - VECTOR of x values, same length as y and z
 * y  - VECTOR of y values, same length as x and z
 * z  - VECTOR of z values, same length as x and y
 * phi - VECTOR of the Fourier transform of the spatial basis
 *      function, evaluated at [kx, ky, kz].  Same length as kx, ky, and kz.
 *
 * recommended g++ options:
 *  -O3 -lm -ffast-math -funroll-all-loops
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <malloc.h>
#include <chrono>
#include <Kokkos_Core.hpp>
#include "file.h"
#include "computeQ.hpp"

int main (int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
  char* inputFileName = argv[1];
  char* outputFileName = argv[2];
  const int iterations = atoi(argv[3]);

  int numX, numK;		/* Number of X and K values */
  float *kx, *ky, *kz;		/* K trajectory (3D vectors) */
  float *x, *y, *z;		/* X coordinates (3D vectors) */
  float *phiR, *phiI;		/* Phi values (complex) */
  float *phiMag;		/* Magnitude of Phi */
  float *Qr, *Qi;		/* Q signal (complex) */

  struct kValues* kVals;

  /* Read in data */
  inputData(inputFileName,
	    &numK, &numX,
	    &kx, &ky, &kz,
	    &x, &y, &z,
	    &phiR, &phiI);

  printf("%d pixels in output; %d samples in trajectory\n", numX, numK);

  /* Create CPU data structures */
  float* initF = new float[numX]();
  createDataStructsCPU(numK, numX, &phiMag, &Qr, &Qi);

  /* GPU section 1 (precompute PhiMag) */
  /* Mirror several data structures on the device */
  float *phiR_d = (float*)Kokkos::kokkos_malloc<MemSpace>(numK*sizeof(float));
  Kokkos::Impl::DeepCopy<MemSpace, host_memory>(phiR_d, phiR, numK*sizeof(float));

  float *phiI_d = (float*)Kokkos::kokkos_malloc<MemSpace>(numK*sizeof(float));
  Kokkos::Impl::DeepCopy<MemSpace, host_memory>(phiI_d, phiI, numK*sizeof(float));

  float *phiMag_d = (float*)Kokkos::kokkos_malloc<MemSpace>(numK*sizeof(float));

  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();

  for(int i = 0; i < iterations; i++)
    computePhiMag_GPU(numK, phiR_d, phiI_d, phiMag_d);

  Kokkos::fence();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("computePhiMag execution time: %f s\n", time * 1e-9);

  Kokkos::Impl::DeepCopy<host_memory, MemSpace>(phiMag, phiMag_d, numK*sizeof(float));
  Kokkos::fence();
  Kokkos::kokkos_free(phiMag_d);
  Kokkos::kokkos_free(phiI_d);
  Kokkos::kokkos_free(phiR_d);

  kVals = (struct kValues*)calloc(numK, sizeof (struct kValues));
  for (int k = 0; k < numK; k++) {
    kVals[k].Kx = kx[k];
    kVals[k].Ky = ky[k];
    kVals[k].Kz = kz[k];
    kVals[k].PhiMag = phiMag[k];
  }

  /* GPU section 2 */
  float *x_d = (float*)Kokkos::kokkos_malloc<MemSpace>(numX*sizeof(float));
  Kokkos::Impl::DeepCopy<MemSpace, host_memory>(x_d, x, numX*sizeof(float));

  float *y_d = (float*)Kokkos::kokkos_malloc<MemSpace>(numX*sizeof(float));
  Kokkos::Impl::DeepCopy<MemSpace, host_memory>(y_d, y, numX*sizeof(float));

  float *z_d = (float*)Kokkos::kokkos_malloc<MemSpace>(numX*sizeof(float));
  Kokkos::Impl::DeepCopy<MemSpace, host_memory>(z_d, z, numX*sizeof(float));

  float *Qr_d = (float*)Kokkos::kokkos_malloc<MemSpace>(numX*sizeof(float));
  float *Qi_d = (float*)Kokkos::kokkos_malloc<MemSpace>(numX*sizeof(float));

  Kokkos::Impl::DeepCopy<MemSpace, host_memory>(Qr_d, initF, numX*sizeof(float));
  Kokkos::Impl::DeepCopy<MemSpace, host_memory>(Qi_d, initF, numX*sizeof(float));

  Kokkos::fence();
  start = std::chrono::steady_clock::now();

  for(int i = 0; i < iterations; i++)
    computeQ_GPU(numK, numX, x_d, y_d, z_d, kVals, Qr_d, Qi_d);

  Kokkos::fence();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("computeQ time: %f s\n", time * 1e-9);

  Kokkos::Impl::DeepCopy<host_memory, MemSpace>(Qr, Qr_d, numK*sizeof(float));
  Kokkos::Impl::DeepCopy<host_memory, MemSpace>(Qi, Qi_d, numK*sizeof(float));
  Kokkos::fence();
  Kokkos::kokkos_free(x_d);
  Kokkos::kokkos_free(y_d);
  Kokkos::kokkos_free(z_d);
  Kokkos::kokkos_free(Qr_d);
  Kokkos::kokkos_free(Qi_d);

  outputData(outputFileName, Qr, Qi, numX);

  free(phiMag);
  free (kx);
  free (ky);
  free (kz);
  free (x);
  free (y);
  free (z);
  free (phiR);
  free (phiI);
  free (kVals);
  free (Qr);
  free (Qi);
  delete[] initF;
  }
  Kokkos::finalize();
  return 0;
}
