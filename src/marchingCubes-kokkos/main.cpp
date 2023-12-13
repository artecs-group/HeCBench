//
// An implementation of Parallel Marching Blocks algorithm
//

#include <cstdio>
#include <random>
#include <chrono>
#include <Kokkos_Core.hpp>
#include "tables.h"

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

// problem size
constexpr unsigned int N(1024);
constexpr unsigned int Nd2(N / 2);
constexpr unsigned int voxelXLv1(16);
constexpr unsigned int voxelYLv1(16);
constexpr unsigned int voxelZLv1(64);
constexpr unsigned int gridXLv1((N - 1) / (voxelXLv1 - 1));
constexpr unsigned int gridYLv1((N - 1) / (voxelYLv1 - 1));
constexpr unsigned int gridZLv1((N - 1) / (voxelZLv1 - 1));
constexpr unsigned int countingThreadNumLv1(128);
constexpr unsigned int blockNum(gridXLv1* gridYLv1* gridZLv1);
constexpr unsigned int countingBlockNumLv1(blockNum / countingThreadNumLv1);

constexpr unsigned int voxelXLv2(4);
constexpr unsigned int voxelYLv2(4);
constexpr unsigned int voxelZLv2(8);
constexpr unsigned int blockXLv2(5);
constexpr unsigned int blockYLv2(5);
constexpr unsigned int blockZLv2(9);
constexpr unsigned int voxelNumLv2(blockXLv2* blockYLv2* blockZLv2);

constexpr unsigned int countingThreadNumLv2(1024);
constexpr unsigned int gridXLv2(gridXLv1* blockXLv2);
constexpr unsigned int gridYLv2(gridYLv1* blockYLv2);
//constexpr unsigned int gridZLv2(gridZLv1* blockZLv2);

KOKKOS_IMPL_INLINE_FUNCTION float f(unsigned int x, unsigned int y, unsigned int z)
{
  constexpr float d(2.0f / N);
  float xf((int(x - Nd2)) * d);//[-1, 1)
  float yf((int(z - Nd2)) * d);
  float zf((int(z - Nd2)) * d);
  return 1.f - 16.f * xf * yf * zf - 4.f * (xf * xf + yf * yf + zf * zf);
}

KOKKOS_IMPL_INLINE_FUNCTION float zeroPoint(unsigned int x, float v0, float v1, float isoValue)
{
  return ((x * (v1 - isoValue) + (x + 1) * (isoValue - v0)) / (v1 - v0) - Nd2) * (2.0f / N);
}

KOKKOS_IMPL_INLINE_FUNCTION float transformToCoord(unsigned int x)
{
  return (int(x) - int(Nd2)) * (2.0f / N);
}

void computeMinMaxLv1(float*__restrict minMax)
{
  size_t lws = 1 * voxelYLv1 * voxelXLv1;
  size_t gws = gridZLv1 * gridYLv1 * gridXLv1;
  Kokkos::parallel_for("min_max1", 
  Kokkos::TeamPolicy<ExecSpace>(gws, lws)
  .set_scratch_size(0, Kokkos::PerTeam(64*sizeof(float))), 
  KOKKOS_LAMBDA(Kokkos::TeamPolicy<ExecSpace>::member_type memT){
    float* sminMax = (float*) memT.team_shmem().get_shmem(64*sizeof(float));
    constexpr unsigned int threadNum(voxelXLv1 * voxelYLv1);
    constexpr unsigned int warpNum(threadNum / 32);
    int blockIdx_z = memT.league_rank() / (gridYLv1 * gridXLv1);
    int blockIdx_y = (memT.league_rank() % (gridYLv1 * gridXLv1)) / gridXLv1;
    int blockIdx_x = memT.league_rank() % gridXLv1;
    int threadIdx_y = memT.team_rank() / voxelXLv1;
    int threadIdx_x = memT.team_rank() % voxelXLv1;
    unsigned int x(blockIdx_x * (voxelXLv1 - 1) + threadIdx_x);
    unsigned int y(blockIdx_y * (voxelYLv1 - 1) + threadIdx_y);
    unsigned int z(blockIdx_z * (voxelZLv1 - 1));
    unsigned int tid(threadIdx_x + voxelXLv1 * threadIdx_y);
    unsigned int laneid = tid % 32;
    unsigned int blockid(blockIdx_x + gridXLv1 * (blockIdx_y + gridYLv1 * blockIdx_z));
    unsigned int warpid(tid >> 5);
    float v(f(x, y, z));
    float minV(v), maxV(v);
    for (int c0(1); c0 < voxelZLv1; ++c0)
    {
      v = f(x, y, z + c0);
      if (v < minV)minV = v;
      if (v > maxV)maxV = v;
    }
  #pragma unroll
    for (int c0(16); c0 > 0; c0 /= 2)
    {
      float t0, t1;
      t0 = KOKKOS_IF_ON_DEVICE(static_cast<float>(__shfl_down_sync(0xffffffffu, minV, c0)));
      t1 = KOKKOS_IF_ON_DEVICE(static_cast<float>(__shfl_down_sync(0xffffffffu, maxV, c0)));
      if (t0 < minV)minV = t0;
      if (t1 > maxV)maxV = t1;
    }
    if (laneid == 0)
    {
      sminMax[warpid] = minV;
      sminMax[warpid + warpNum] = maxV;
    }
    memT.team_barrier();
    if (warpid == 0)
    {
      minV = sminMax[laneid];
      maxV = sminMax[laneid + warpNum];
  #pragma unroll
      for (int c0(warpNum / 2); c0 > 0; c0 /= 2)
      {
        float t0, t1;
        t0 = KOKKOS_IF_ON_DEVICE(static_cast<float>(__shfl_down_sync(0xffffffffu, minV, c0)));
        t1 = KOKKOS_IF_ON_DEVICE(static_cast<float>(__shfl_down_sync(0xffffffffu, maxV, c0)));
        if (t0 < minV)minV = t0;
        if (t1 > maxV)maxV = t1;
      }
      if (laneid == 0)
      {
        minMax[blockid * 2] = minV;
        minMax[blockid * 2 + 1] = maxV;
      }
    }
  });
}

void compactLv1(
  float isoValue,
  const float*__restrict minMax,
  unsigned int*__restrict blockIndices,
  unsigned int*__restrict countedBlockNum)
{
  constexpr size_t lws = countingThreadNumLv1;
  constexpr size_t gws = countingBlockNumLv1;
  Kokkos::parallel_for("compact1", 
  Kokkos::TeamPolicy<ExecSpace>(gws, lws)
  .set_scratch_size(0, Kokkos::PerTeam(32*sizeof(unsigned int))), 
  KOKKOS_LAMBDA(Kokkos::TeamPolicy<ExecSpace>::member_type memT){
    unsigned int* sums = (unsigned int*) memT.team_shmem().get_shmem(32*sizeof(unsigned int));
    constexpr unsigned int warpNum(countingThreadNumLv1 / 32);
    unsigned int tid(memT.team_rank());
    unsigned int laneid = tid % 32;
    unsigned int bIdx(memT.league_rank() * countingThreadNumLv1 + tid);
    unsigned int warpid(tid >> 5);
    unsigned int test;
    if (minMax[2 * bIdx] <= isoValue && minMax[2 * bIdx + 1] >= isoValue)test = 1;
    else test = 0;
    unsigned int testSum(test);
  #pragma unroll
    for (int c0(1); c0 < 32; c0 *= 2)
    {
      unsigned int tp(KOKKOS_IF_ON_DEVICE(__shfl_up_sync(0xffffffffu, testSum, c0)));
      if (laneid >= c0) testSum += tp;
    }
    if (laneid == 31)sums[warpid] = testSum;
    memT.team_barrier();
    if (warpid == 0)
    {
      unsigned warpSum = sums[laneid];
  #pragma unroll
      for (int c0(1); c0 < warpNum; c0 *= 2)
      {
        unsigned int tp(KOKKOS_IF_ON_DEVICE(__shfl_up_sync(0xffffffffu, warpSum, c0)));
        if (laneid >= c0) warpSum += tp;
      }
      sums[laneid] = warpSum;
    }
    memT.team_barrier();
    if (warpid != 0)testSum += sums[warpid - 1];
    if (tid == countingThreadNumLv1 - 1 && testSum != 0) {
      sums[31] = Kokkos::atomic_fetch_add(countedBlockNum, testSum);
    }
    memT.team_barrier();
    if (test) blockIndices[testSum + sums[31] - 1] = bIdx;
  });
}

void computeMinMaxLv2(
  int countedBlockNumLv1,
  const unsigned int*__restrict blockIndicesLv1,
  float*__restrict minMax)
{
  const size_t lws = (blockXLv2 * blockYLv2) * (voxelXLv2 * voxelYLv2);
  const size_t gws = countedBlockNumLv1;
  Kokkos::parallel_for("min_max2", 
  Kokkos::TeamPolicy<ExecSpace>(gws, lws), 
  KOKKOS_LAMBDA(Kokkos::TeamPolicy<ExecSpace>::member_type memT){
    unsigned int tid(memT.team_rank() % (voxelXLv2 * voxelYLv2));
    unsigned int voxelOffset(memT.team_rank() / (voxelXLv2 * voxelYLv2));
    unsigned int blockIdx_x = memT.league_rank() % countedBlockNumLv1;
    unsigned int blockIndex(blockIndicesLv1[blockIdx_x]);
    unsigned int tp(blockIndex);
    unsigned int x((blockIndex % gridXLv1) * (voxelXLv1 - 1) + (voxelOffset % 5) * (voxelXLv2 - 1) + (tid & 3));
    tp /= gridXLv1;
    unsigned int y((tp % gridYLv1) * (voxelYLv1 - 1) + (voxelOffset / 5) * (voxelYLv2 - 1) + (tid >> 2));
    tp /= gridYLv1;
    unsigned int z(tp * (voxelZLv1 - 1));
    float v(f(x, y, z));
    float minV(v), maxV(v);
    unsigned int idx(2 * (voxelOffset + voxelNumLv2 * blockIdx_x));
    for (int c0(0); c0 < blockZLv2; ++c0)
    {
      for (int c1(1); c1 < voxelZLv2; ++c1)
      {
        v = f(x, y, z + c1);
        if (v < minV)minV = v;
        if (v > maxV)maxV = v;
      }
      z += voxelZLv2 - 1;
  #pragma unroll
      for (int c1(8); c1 > 0; c1 /= 2)
      {
        float t0, t1;
        t0 = KOKKOS_IF_ON_DEVICE(static_cast<float>(__shfl_down_sync(0xffffffffu, minV, c1)));
        t1 = KOKKOS_IF_ON_DEVICE(static_cast<float>(__shfl_down_sync(0xffffffffu, maxV, c1)));
        if (t0 < minV)minV = t0;
        if (t1 > maxV)maxV = t1;
      }
      if (tid == 0)
      {
        minMax[idx] = minV;
        minMax[idx + 1] = maxV;
        constexpr unsigned int offsetSize(2 * blockXLv2 * blockYLv2);
        idx += offsetSize;
      }
      minV = v;
      maxV = v;
    }
  });
}

void compactLv2(
  int countingBlockNumLv2,
  float isoValue,
  const float*__restrict minMax,
  const unsigned int*__restrict blockIndicesLv1,
  unsigned int*__restrict blockIndicesLv2,
  unsigned int counterBlockNumLv1,
  unsigned int*__restrict countedBlockNumLv2)
{
  const size_t lws = countingThreadNumLv2;
  const size_t gws = countingBlockNumLv2;
  Kokkos::parallel_for("compact2", 
  Kokkos::TeamPolicy<ExecSpace>(gws, lws)
  .set_scratch_size(0, Kokkos::PerTeam(32*sizeof(unsigned int))), 
  KOKKOS_LAMBDA(Kokkos::TeamPolicy<ExecSpace>::member_type memT){
    unsigned int* sums = (unsigned int*) memT.team_shmem().get_shmem(32*sizeof(unsigned int));
    constexpr unsigned int warpNum(countingThreadNumLv2 / 32);
    unsigned int tid(memT.team_rank());
    unsigned int laneid = tid % 32;
    unsigned int warpid(tid >> 5);
    unsigned int id0(tid + memT.league_rank() * countingThreadNumLv2);
    unsigned int id1(id0 / voxelNumLv2);
    unsigned int test;
    if (id1 < counterBlockNumLv1)
    {
      if (minMax[2 * id0] <= isoValue && minMax[2 * id0 + 1] >= isoValue)
        test = 1;
      else
        test = 0;
    }
    else test = 0;
    unsigned int testSum(test);
  #pragma unroll
    for (int c0(1); c0 < 32; c0 *= 2)
    {
      unsigned int tp(KOKKOS_IF_ON_DEVICE(__shfl_up_sync(0xffffffffu, testSum, c0)));
      if (laneid >= c0) testSum += tp;
    }
    if (laneid == 31) sums[warpid] = testSum;
    memT.team_barrier();
    if (warpid == 0)
    {
      unsigned int warpSum = sums[laneid];
  #pragma unroll
      for (int c0(1); c0 < warpNum; c0 *= 2)
      {
        unsigned int tp(KOKKOS_IF_ON_DEVICE(__shfl_up_sync(0xffffffffu, warpSum, c0)));
        if (laneid >= c0)warpSum += tp;
      }
      sums[laneid] = warpSum;
    }
    memT.team_barrier();
    if (warpid != 0) testSum += sums[warpid - 1];
    if (tid == countingThreadNumLv2 - 1) {
      sums[31] = Kokkos::atomic_fetch_add(countedBlockNumLv2, testSum);
    }
    memT.team_barrier();

    if (test)
    {
      unsigned int bIdx1(blockIndicesLv1[id1]);
      unsigned int bIdx2;
      unsigned int x1, y1, z1;
      unsigned int x2, y2, z2;
      unsigned int tp1(bIdx1);
      unsigned int tp2((tid + memT.league_rank() * countingThreadNumLv2) % voxelNumLv2);
      x1 = tp1 % gridXLv1;
      x2 = tp2 % blockXLv2;
      tp1 /= gridXLv1;
      tp2 /= blockXLv2;
      y1 = tp1 % gridYLv1;
      y2 = tp2 % blockYLv2;
      z1 = tp1 / gridYLv1;
      z2 = tp2 / blockYLv2;
      bIdx2 = x2 + blockXLv2 * (x1 + gridXLv1 * (y2 + blockYLv2 * (y1 + gridYLv1 * (z1 * blockZLv2 + z2))));
      blockIndicesLv2[testSum + sums[31] - 1] = bIdx2;
    }
  });
}

int main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  unsigned int repeat = atoi(argv[1]);

  std::uniform_real_distribution<float>rd(0, 1);
  std::mt19937 mt(123);

  float *minMaxLv1Device = (float*)Kokkos::kokkos_malloc<MemSpace>(blockNum * 2*sizeof(float));
  unsigned int *blockIndicesLv1Device = (unsigned int*)Kokkos::kokkos_malloc<MemSpace>(blockNum*sizeof(unsigned int));
  unsigned int *countedBlockNumLv1Device = (unsigned int*)Kokkos::kokkos_malloc<MemSpace>(1*sizeof(unsigned int));
  unsigned int *countedBlockNumLv2Device = (unsigned int*)Kokkos::kokkos_malloc<MemSpace>(1*sizeof(unsigned int));

  unsigned short *distinctEdgesTableDevice = (unsigned short*)Kokkos::kokkos_malloc<MemSpace>(256*sizeof(unsigned short));
  Kokkos::Impl::DeepCopy<MemSpace, host_memory>(distinctEdgesTableDevice, distinctEdgesTable, 256*sizeof(unsigned short));

  int *triTableDevice = (int*)Kokkos::kokkos_malloc<MemSpace>(256*16*sizeof(int));
  Kokkos::Impl::DeepCopy<MemSpace, host_memory>(triTableDevice, triTable, 256*16*sizeof(int));

  pixel* edgeIDTableDevice = (pixel*)Kokkos::kokkos_malloc<MemSpace>(12*sizeof(pixel));
  Kokkos::Impl::DeepCopy<MemSpace, host_memory>(edgeIDTableDevice, edgeIDTableP, 12*sizeof(pixel));

  unsigned int *countedVerticesNumDevice = (unsigned int*)Kokkos::kokkos_malloc<MemSpace>(sizeof(unsigned int));
  unsigned int *countedTrianglesNumDevice = (unsigned int*)Kokkos::kokkos_malloc<MemSpace>(sizeof(unsigned int));

  // simulate rendering without memory allocation for vertices and triangles
  unsigned long long *trianglesDevice = (unsigned long long*)Kokkos::kokkos_malloc<MemSpace>(sizeof(unsigned long long));
  float *coordXDevice = (float*)Kokkos::kokkos_malloc<MemSpace>(sizeof(float));
  float *coordYDevice = (float*)Kokkos::kokkos_malloc<MemSpace>(sizeof(float));
  float *coordZDevice = (float*)Kokkos::kokkos_malloc<MemSpace>(sizeof(float));
  float *coordZPDevice = (float*)Kokkos::kokkos_malloc<MemSpace>(sizeof(float));

  constexpr size_t BlockSizeGenerating = voxelZLv2*voxelYLv2*voxelXLv2;

  float isoValue(-0.9f);

  unsigned int countedBlockNumLv1;
  unsigned int countedBlockNumLv2;
  unsigned int countedVerticesNum;
  unsigned int countedTrianglesNum;
  constexpr int init{0};
  constexpr float initF{0.0f};

  float time(0.f);

  for (unsigned int c0(0); c0 < repeat; ++c0)
  {
    Kokkos::fence();

    Kokkos::Impl::DeepCopy<MemSpace, host_memory>(countedBlockNumLv1Device, &init, sizeof(unsigned int));
    Kokkos::Impl::DeepCopy<MemSpace, host_memory>(countedBlockNumLv2Device, &init, sizeof(unsigned int));
    Kokkos::Impl::DeepCopy<MemSpace, host_memory>(countedVerticesNumDevice, &init, sizeof(unsigned int));
    Kokkos::Impl::DeepCopy<MemSpace, host_memory>(countedTrianglesNumDevice, &init, sizeof(unsigned int));
    Kokkos::Impl::DeepCopy<MemSpace, host_memory>(trianglesDevice, &init, sizeof(unsigned long long));
    Kokkos::Impl::DeepCopy<MemSpace, host_memory>(coordXDevice, &initF, sizeof(float));
    Kokkos::Impl::DeepCopy<MemSpace, host_memory>(coordYDevice, &initF, sizeof(float));
    Kokkos::Impl::DeepCopy<MemSpace, host_memory>(coordZDevice, &initF, sizeof(float));
    Kokkos::Impl::DeepCopy<MemSpace, host_memory>(coordZPDevice, &initF, sizeof(float));
    Kokkos::fence();

    computeMinMaxLv1(minMaxLv1Device);
    compactLv1(isoValue, minMaxLv1Device, blockIndicesLv1Device, countedBlockNumLv1Device);

    Kokkos::Impl::DeepCopy<host_memory, MemSpace>(&countedBlockNumLv1, countedBlockNumLv1Device, sizeof(unsigned int));
    Kokkos::fence();
    float* minMaxLv2Device = (float*)Kokkos::kokkos_malloc<MemSpace>(countedBlockNumLv1 * voxelNumLv2 * 2*sizeof(float));

    computeMinMaxLv2(countedBlockNumLv1, blockIndicesLv1Device, minMaxLv2Device);

    unsigned int *blockIndicesLv2Device = (unsigned int*)Kokkos::kokkos_malloc<MemSpace>(countedBlockNumLv1 * voxelNumLv2*sizeof(unsigned int));
    unsigned int countingBlockNumLv2((countedBlockNumLv1 * voxelNumLv2 + countingThreadNumLv2 - 1) / countingThreadNumLv2);

    compactLv2(countingBlockNumLv2, isoValue, minMaxLv2Device, blockIndicesLv1Device, blockIndicesLv2Device, countedBlockNumLv1, countedBlockNumLv2Device);

    Kokkos::Impl::DeepCopy<host_memory, MemSpace>(&countedBlockNumLv2, countedBlockNumLv2Device, sizeof(unsigned int));
    Kokkos::fence();
    constexpr size_t shSizeTr = voxelZLv2*voxelYLv2*voxelXLv2*sizeof(unsigned short) + (voxelZLv2+1)*(voxelYLv2+1)*(voxelXLv2+1)*sizeof(float) + 32*sizeof(unsigned int) + 32*sizeof(unsigned int);

    auto start = std::chrono::steady_clock::now();

    Kokkos::parallel_for("triangles_gen", 
    Kokkos::TeamPolicy<ExecSpace>(countedBlockNumLv2, BlockSizeGenerating)
    .set_scratch_size(0, Kokkos::PerTeam(shSizeTr)), 
    KOKKOS_LAMBDA(Kokkos::TeamPolicy<ExecSpace>::member_type memT){
      unsigned short (*vertexIndices)[voxelYLv2][voxelXLv2] = (unsigned short (*)[voxelYLv2][voxelXLv2]) memT.team_shmem().get_shmem((voxelZLv2*voxelYLv2*voxelXLv2)*sizeof(unsigned short));
      float (*value)[voxelYLv2 + 1][voxelXLv2 + 1] = (float (*)[voxelYLv2 + 1][voxelXLv2 + 1]) memT.team_shmem().get_shmem(((voxelZLv2+1)*(voxelYLv2+1)*(voxelXLv2+1))*sizeof(float));
      unsigned int* sumsVertices = (unsigned int*) memT.team_shmem().get_shmem(32*sizeof(unsigned int));
      unsigned int* sumsTriangles = (unsigned int*) memT.team_shmem().get_shmem(32*sizeof(unsigned int));
      unsigned int threadIdx_x = memT.team_rank() % voxelXLv2;
      unsigned int threadIdx_y = (memT.team_rank() % (voxelYLv2*voxelXLv2)) / voxelXLv2;
      unsigned int threadIdx_z = memT.team_rank() / (voxelYLv2*voxelXLv2);

      unsigned int blockId(blockIndicesLv2Device[memT.league_rank()]);
      unsigned int tp(blockId);
      unsigned int x((tp % gridXLv2) * (voxelXLv2 - 1) + threadIdx_x);
      tp /= gridXLv2;
      unsigned int y((tp % gridYLv2) * (voxelYLv2 - 1) + threadIdx_y);
      unsigned int z((tp / gridYLv2) * (voxelZLv2 - 1) + threadIdx_z);
      unsigned int eds(7);
      // TODO: index value in 1D
      float v(value[threadIdx_z][threadIdx_y][threadIdx_x] = f(x, y, z));
      if (threadIdx_x == voxelXLv2 - 1)
      {
        eds &= 6;
        value[threadIdx_z][threadIdx_y][voxelXLv2] = f(x + 1, y, z);
        if (threadIdx_y == voxelYLv2 - 1)
          value[threadIdx_z][voxelYLv2][voxelXLv2] = f(x + 1, y + 1, z);
      }
      if (threadIdx_y == voxelYLv2 - 1)
      {
        eds &= 5;
        value[threadIdx_z][voxelYLv2][threadIdx_x] = f(x, y + 1, z);
        if (threadIdx_z == voxelZLv2 - 1)
          value[voxelZLv2][voxelYLv2][threadIdx_x] = f(x, y + 1, z + 1);
      }
      if (threadIdx_z == voxelZLv2 - 1)
      {
        eds &= 3;
        value[voxelZLv2][threadIdx_y][threadIdx_x] = f(x, y, z + 1);
        if (threadIdx_x == voxelXLv2 - 1)
          value[voxelZLv2][threadIdx_y][voxelXLv2] = f(x + 1, y, z + 1);
      }
      eds <<= 13;
      memT.team_barrier();
      unsigned int cubeCase(0);
      if (value[threadIdx_z][threadIdx_y][threadIdx_x] < isoValue) cubeCase |= 1;
      if (value[threadIdx_z][threadIdx_y][threadIdx_x + 1] < isoValue) cubeCase |= 2;
      if (value[threadIdx_z][threadIdx_y + 1][threadIdx_x + 1] < isoValue) cubeCase |= 4;
      if (value[threadIdx_z][threadIdx_y + 1][threadIdx_x] < isoValue) cubeCase |= 8;
      if (value[threadIdx_z + 1][threadIdx_y][threadIdx_x] < isoValue) cubeCase |= 16;
      if (value[threadIdx_z + 1][threadIdx_y][threadIdx_x + 1] < isoValue) cubeCase |= 32;
      if (value[threadIdx_z + 1][threadIdx_y + 1][threadIdx_x + 1] < isoValue) cubeCase |= 64;
      if (value[threadIdx_z + 1][threadIdx_y + 1][threadIdx_x] < isoValue) cubeCase |= 128;

      unsigned int distinctEdges(eds ? distinctEdgesTableDevice[cubeCase] : 0);
      unsigned int numTriangles(eds != 0xe000 ? 0 : distinctEdges & 7);
      unsigned int numVertices(Kokkos::popcount(distinctEdges &= eds));
      unsigned int laneid = (threadIdx_x + voxelXLv2 * (threadIdx_y + voxelYLv2 * threadIdx_z)) % 32;
      unsigned warpid((threadIdx_x + voxelXLv2 * (threadIdx_y + voxelYLv2 * threadIdx_z)) >> 5);
      constexpr unsigned int threadNum(voxelXLv2 * voxelYLv2 * voxelZLv2);
      constexpr unsigned int warpNum(threadNum / 32);
      unsigned int sumVertices(numVertices);
      unsigned int sumTriangles(numTriangles);

      #pragma unroll
      for (int c0(1); c0 < 32; c0 *= 2)
      {
        unsigned int tp0(KOKKOS_IF_ON_DEVICE(__shfl_up_sync(0xffffffffu, sumVertices, c0)));
        unsigned int tp1(KOKKOS_IF_ON_DEVICE(__shfl_up_sync(0xffffffffu, sumTriangles, c0)));
        if (laneid >= c0)
        {
          sumVertices += tp0;
          sumTriangles += tp1;
        }
      }
      if (laneid == 31)
      {
        sumsVertices[warpid] = sumVertices;
        sumsTriangles[warpid] = sumTriangles;
      }
      memT.team_barrier();
      if (warpid == 0)
      {
        unsigned warpSumVertices = sumsVertices[laneid];
        unsigned warpSumTriangles = sumsTriangles[laneid];
        #pragma unroll
        for (int c0(1); c0 < warpNum; c0 *= 2)
        {
          unsigned int tp0(KOKKOS_IF_ON_DEVICE(__shfl_up_sync(0xffffffffu, warpSumVertices, c0)));
          unsigned int tp1(KOKKOS_IF_ON_DEVICE(__shfl_up_sync(0xffffffffu, warpSumTriangles, c0)));
          if (laneid >= c0)
          {
            warpSumVertices += tp0;
            warpSumTriangles += tp1;
          }
        }
        sumsVertices[laneid] = warpSumVertices;
        sumsTriangles[laneid] = warpSumTriangles;
      }
      memT.team_barrier();
      if (warpid != 0)
      {
        sumVertices += sumsVertices[warpid - 1];
        sumTriangles += sumsTriangles[warpid - 1];
      }
      if (eds == 0)
      {
        sumsVertices[31] = Kokkos::atomic_fetch_add(countedVerticesNumDevice, sumVertices);
        sumsTriangles[31] = Kokkos::atomic_fetch_add(countedTrianglesNumDevice, sumTriangles);
      }

      unsigned int interOffsetVertices(sumVertices - numVertices);
      sumVertices = interOffsetVertices + sumsVertices[31];//exclusive offset
      sumTriangles = sumTriangles + sumsTriangles[31] - numTriangles;//exclusive offset
      vertexIndices[threadIdx_z][threadIdx_y][threadIdx_x] = interOffsetVertices | distinctEdges;
      memT.team_barrier();

      for (unsigned int c0(0); c0 < numTriangles; ++c0)
      {
        #pragma unroll
        for (unsigned int c1(0); c1 < 3; ++c1)
        {
          int edgeID(triTableDevice[16 * cubeCase + 3 * c0 + c1]);
          pixel edgePos(edgeIDTableDevice[edgeID]);
          unsigned short vertexIndex(
            vertexIndices[threadIdx_z + edgePos.z][threadIdx_y + edgePos.y][threadIdx_x + edgePos.x]);
          int popC = Kokkos::popcount(static_cast<unsigned short>(vertexIndex >> (16 - edgePos.w)));
          unsigned int tp(popC + (vertexIndex & 0x1fff));
          Kokkos::atomic_add(trianglesDevice, (unsigned long long)(sumsVertices[31] + tp));
        }
      }

      // sumVertices may be too large for a GPU memory
      float zp = 0.f, cx = 0.f, cy = 0.f, cz = 0.f;

      if (distinctEdges & (1 << 15))
      {
        zp = zeroPoint(x, v, value[threadIdx_z][threadIdx_y][threadIdx_x + 1], isoValue);
        cy = transformToCoord(y);
        cz = transformToCoord(z);
      }
      if (distinctEdges & (1 << 14))
      {
        cx = transformToCoord(x);
        zp += zeroPoint(y, v, value[threadIdx_z][threadIdx_y + 1][threadIdx_x], isoValue);
        cz += transformToCoord(z);
      }
      if (distinctEdges & (1 << 13))
      {
        cx += transformToCoord(x);
        cy += transformToCoord(y);
        zp += zeroPoint(z, v, value[threadIdx_z + 1][threadIdx_y][threadIdx_x], isoValue);
      }
      Kokkos::atomic_add(coordXDevice, cx);
      Kokkos::atomic_add(coordYDevice, cy);
      Kokkos::atomic_add(coordZDevice, cz);
      Kokkos::atomic_add(coordZPDevice, zp);
    });
    Kokkos::fence();

    auto end = std::chrono::steady_clock::now();
    auto ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    time += ktime;

    Kokkos::Impl::DeepCopy<host_memory, MemSpace>(&countedVerticesNum, countedVerticesNumDevice, sizeof(unsigned int));
    Kokkos::Impl::DeepCopy<host_memory, MemSpace>(&countedTrianglesNum, countedTrianglesNumDevice, sizeof(unsigned int));
    Kokkos::fence();

    Kokkos::kokkos_free(minMaxLv2Device);
    Kokkos::kokkos_free(blockIndicesLv2Device);
  }

  printf("Block Lv1: %u\nBlock Lv2: %u\n", countedBlockNumLv1, countedBlockNumLv2);
  printf("Vertices Size: %u\n", countedBlockNumLv2 * 304);
  printf("Triangles Size: %u\n", countedBlockNumLv2 * 315 * 3);
  printf("Vertices: %u\nTriangles: %u\n", countedVerticesNum, countedTrianglesNum);
  printf("Total kernel execution time (generatingTriangles): %f (s)\n", (time * 1e-9f));
  printf("Average kernel execution time (generatingTriangles): %f (s)\n", (time * 1e-9f) / repeat);

  // specific to the problem size
  bool ok = (countedBlockNumLv1 == 8296 && countedBlockNumLv2 == 240380 &&
             countedVerticesNum == 4856560 && countedTrianglesNum == 6101640);
  printf("%s\n", ok ? "PASS" : "FAIL");

  Kokkos::kokkos_free(minMaxLv1Device);
  Kokkos::kokkos_free(blockIndicesLv1Device);
  Kokkos::kokkos_free(countedBlockNumLv1Device);
  Kokkos::kokkos_free(countedBlockNumLv2Device);
  Kokkos::kokkos_free(distinctEdgesTableDevice);
  Kokkos::kokkos_free(triTableDevice);
  Kokkos::kokkos_free(edgeIDTableDevice);
  Kokkos::kokkos_free(countedVerticesNumDevice);
  Kokkos::kokkos_free(countedTrianglesNumDevice);
  Kokkos::kokkos_free(trianglesDevice);
  Kokkos::kokkos_free(coordXDevice);
  Kokkos::kokkos_free(coordYDevice);
  Kokkos::kokkos_free(coordZDevice);
  Kokkos::kokkos_free(coordZPDevice);
  }
  Kokkos::finalize();
  return 0;
}
