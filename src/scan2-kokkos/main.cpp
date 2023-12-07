/**********************************************************************
  Copyright �2013 Advanced Micro Devices, Inc. All rights reserved.

  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

  �   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  �   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ********************************************************************/


#include <chrono>
#include <Kokkos_Core.hpp>
#include "scan.h"

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

typedef Kokkos::HostSpace::memory_space host_memory;

void bScan(const unsigned int blockSize,
           const unsigned int len,
           float *input,
           float *output,
           float *blockSum)
{
  // set the block size
  size_t gws (len / blockSize);
  size_t lws (blockSize/2);

  Kokkos::parallel_for("lock_scan", 
  Kokkos::TeamPolicy<ExecSpace>(gws, lws)
  .set_scratch_size(0, Kokkos::PerTeam(blockSize*sizeof(float))), 
  KOKKOS_LAMBDA(Kokkos::TeamPolicy<ExecSpace>::member_type memT){
    float* block = (float*) memT.team_shmem().get_shmem(blockSize*sizeof(float));

    int tid = memT.team_rank();
    int gid = memT.league_rank() * memT.team_size() + tid;
    int bid = memT.league_rank();

    /* Cache the computational window in shared memory */
    block[2*tid]     = input[2*gid];
    block[2*tid + 1] = input[2*gid + 1];
    memT.team_barrier();

    float cache0 = block[0];
    float cache1 = cache0 + block[1];

    /* build the sum in place up the tree */
    for(int stride = 1; stride < blockSize; stride *=2) {
      if(2*tid>=stride) {
        cache0 = block[2*tid-stride]+block[2*tid];
        cache1 = block[2*tid+1-stride]+block[2*tid+1];
      }
      memT.team_barrier();

      block[2*tid] = cache0;
      block[2*tid+1] = cache1;

      memT.team_barrier();
    }

    /* store the value in sum buffer before making it to 0 */
    blockSum[bid] = block[blockSize-1];

    /*write the results back to global memory */
    if(tid==0) {
      output[2*gid]     = 0;
      output[2*gid+1]   = block[2*tid];
    } else {
      output[2*gid]     = block[2*tid-1];
      output[2*gid + 1] = block[2*tid];
    }
  });
}

void pScan(const unsigned int blockSize,
           const unsigned int len,
           float *input,
           float *output)
{
  size_t gws (1);
  size_t lws (len/2);

  Kokkos::parallel_for("partial_scan", 
  Kokkos::TeamPolicy<ExecSpace>(gws, lws)
  .set_scratch_size(0, Kokkos::PerTeam(len+1*sizeof(float))), 
  KOKKOS_LAMBDA(Kokkos::TeamPolicy<ExecSpace>::member_type memT){
    float* block = (float*) memT.team_shmem().get_shmem(len+1*sizeof(float));

    int tid = memT.team_rank();
    int gid = memT.league_rank() * memT.team_size() + tid;
    //int bid = item.get_group(0);

    /* Cache the computational window in shared memory */
    block[2*tid]     = input[2*gid];
    block[2*tid + 1] = input[2*gid + 1];
    memT.team_barrier();

    float cache0 = block[0];
    float cache1 = cache0 + block[1];

    /* build the sum in place up the tree */
    for(int stride = 1; stride < blockSize; stride *=2) {

      if(2*tid>=stride) {
        cache0 = block[2*tid-stride]+block[2*tid];
        cache1 = block[2*tid+1-stride]+block[2*tid+1];
      }
      memT.team_barrier();

      block[2*tid] = cache0;
      block[2*tid+1] = cache1;

      memT.team_barrier();
    }

    /*write the results back to global memory */
    if(tid==0) {
      output[2*gid]     = 0;
      output[2*gid+1]   = block[2*tid];
    } else {
      output[2*gid]     = block[2*tid-1];
      output[2*gid + 1] = block[2*tid];
    }
  });
}

void bAddition(const unsigned int blockSize,
               const unsigned int len,
               float *input,
               float *output)
{
  // set the block size
  size_t gws (len/blockSize);
  size_t lws (blockSize);

  // Ensycl::queue a kernel run call
  Kokkos::parallel_for("block_add", 
  Kokkos::TeamPolicy<ExecSpace>(gws, lws)
  .set_scratch_size(0, Kokkos::PerTeam(sizeof(float))), 
  KOKKOS_LAMBDA(Kokkos::TeamPolicy<ExecSpace>::member_type memT){
    float* value = (float*) memT.team_shmem().get_shmem(sizeof(float));
    int globalId = memT.league_rank() * memT.team_size() + memT.team_rank();
    int groupId = memT.league_rank();
    int localId = memT.team_rank();

    /* Only 1 thread of a group will read from global buffer */
    if(localId == 0) value[0] = input[groupId];
    memT.team_barrier();

    output[globalId] += value[0];
  });
}


/*
 * Scan for verification
 */
void scanLargeArraysCPUReference(
    float * output,
    float * input,
    const unsigned int length)
{
  output[0] = 0;

  for(unsigned int i = 1; i < length; ++i)
  {
    output[i] = input[i-1] + output[i-1];
  }
}


int main(int argc, char * argv[])
{
  Kokkos::initialize(argc, argv);
  {
  if (argc != 4) {
    std::cout << "Usage: " << argv[0] << " <repeat> <input length> <block size>\n";
    return 1;
  }
  int iterations = atoi(argv[1]);
  int length = atoi(argv[2]);
  int blockSize = atoi(argv[3]);

  if(iterations < 1)
  {
    std::cout << "Error, iterations cannot be 0 or negative. Exiting..\n";
    return -1;
  }
  if(!isPowerOf2(length))
  {
    length = roundToPowerOf2(length);
  }

  if((length/blockSize>GROUP_SIZE)&&(((length)&(length-1))!=0))
  {
    std::cout << "Invalid length: " << length << std::endl;
    return -1;
  }

  // input buffer size
  unsigned int sizeBytes = length * sizeof(float);

  float* input = (float*) malloc (sizeBytes);

  // store device results for verification
  float* output = (float*) malloc (sizeBytes);

  // random initialisation of input
  fillRandom<float>(input, length, 1, 0, 255);

  blockSize = (blockSize < length/2) ? blockSize : length/2;

  // Calculate number of passes required
  float t = std::log((float)length) / std::log((float)blockSize);
  unsigned int pass = (unsigned int)t;

  // If t is equal to pass
  if(std::fabs(t - (float)pass) < 1e-7)
  {
    pass--;
  }

  // Create input buffer on device
  float *inputBuffer = (float*)Kokkos::kokkos_malloc<MemSpace>(sizeBytes);
  Kokkos::Impl::DeepCopy<MemSpace, host_memory>(inputBuffer, input, sizeBytes);

  // Allocate output buffers
  std::vector<float*> outputBuffers(pass);

  for(unsigned int i = 0; i < pass; i++)
  {
    int size = (int)(length / std::pow((float)blockSize,(float)i));
    outputBuffers[i] = (float*)Kokkos::kokkos_malloc<MemSpace>(size*sizeof(float));
  }

  // Allocate blockSumBuffers
  std::vector<float*> blockSumBuffers(pass);

  for(unsigned int i = 0; i < pass; i++)
  {
    int size = (int)(length / std::pow((float)blockSize,(float)(i + 1)));
    blockSumBuffers[i] = (float*)Kokkos::kokkos_malloc<MemSpace>(size*sizeof(float));
  }

  // Create a tempBuffer on device
  int tempLength = (int)(length / std::pow((float)blockSize, (float)pass));

  float *tempBuffer = (float*)Kokkos::kokkos_malloc<MemSpace>(tempLength*sizeof(float));

  std::cout << "Executing kernel for " << iterations << " iterations\n";
  std::cout << "-------------------------------------------\n";

  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();

  for(int n = 0; n < iterations; n++)
  {
    // Do block-wise sum
    bScan(blockSize, length, inputBuffer, outputBuffers[0], blockSumBuffers[0]);

    for(int i = 1; i < (int)pass; i++)
    {
      int size = (int)(length / std::pow((float)blockSize,(float)i));
      bScan(blockSize, size, blockSumBuffers[i - 1], outputBuffers[i], blockSumBuffers[i]);
    }

    // Do scan to tempBuffer
    pScan(blockSize, tempLength, blockSumBuffers[pass - 1], tempBuffer);

    // Do block-addition on outputBufferss
    bAddition(blockSize, (unsigned int)(length / std::pow((float)blockSize, (float)(pass - 1))),
          tempBuffer, outputBuffers[pass - 1]);

    for(int i = pass - 1; i > 0; i--)
    {
      bAddition(blockSize, (unsigned int)(length / std::pow((float)blockSize, (float)(i - 1))),
            outputBuffers[i], outputBuffers[i - 1]);
    }
  }

  Kokkos::fence();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time of scan kernels: " << time * 1e-3f / iterations
            << " (us)\n";

  Kokkos::Impl::DeepCopy<host_memory, MemSpace>(output, outputBuffers[0], sizeBytes);
  Kokkos::fence();

  Kokkos::kokkos_free(inputBuffer);

  for(unsigned int i = 0; i < pass; i++)
  {
    Kokkos::kokkos_free(outputBuffers[i]);
    Kokkos::kokkos_free(blockSumBuffers[i]);
  }

  Kokkos::kokkos_free(tempBuffer);

#ifdef VERIFY
  // verification
  float* verificationOutput = (float*)malloc(sizeBytes);
  memset(verificationOutput, 0, sizeBytes);

  // reference implementation
  scanLargeArraysCPUReference(verificationOutput, input, length);

  // compare the results and see if they match
  if (compare<float>(output, verificationOutput, length, (float)0.001))
    std::cout << "PASS" << std::endl;
  else
    std::cout << "FAIL" << std::endl;
#endif
  free(input);
  free(output);
  free(verificationOutput);
  }
  Kokkos::finalize();
  return 0;
}
