#include "morphology.h"

enum class MorphOpType {
    ERODE,
    DILATE,
};

// Forward declarations
template <MorphOpType opType>
class vert;

template <MorphOpType opType>
class horiz;

template <MorphOpType opType>
KOKKOS_INLINE_FUNCTION unsigned char elementOp(unsigned char lhs, unsigned char rhs)
{
}

template <>
KOKKOS_INLINE_FUNCTION unsigned char elementOp<MorphOpType::ERODE>(unsigned char lhs, unsigned char rhs)
{
    return Kokkos::min(lhs, rhs);
}

template <>
KOKKOS_INLINE_FUNCTION unsigned char elementOp<MorphOpType::DILATE>(unsigned char lhs, unsigned char rhs)
{
    return Kokkos::max(lhs, rhs);
}

template <MorphOpType opType>
KOKKOS_INLINE_FUNCTION unsigned char borderValue()
{
}

template <>
KOKKOS_INLINE_FUNCTION unsigned char borderValue<MorphOpType::ERODE>()
{
    return BLACK;
}

template <>
KOKKOS_INLINE_FUNCTION unsigned char borderValue<MorphOpType::DILATE>()
{
    return WHITE;
}

template <MorphOpType opType>
KOKKOS_IMPL_FUNCTION void twoWayScan(unsigned char* __restrict sMem,
                unsigned char* __restrict opArray,
                const int selSize,
                const int tid,
                Kokkos::TeamPolicy<ExecSpace>::member_type memT)
{
  opArray[tid] = sMem[tid];
  opArray[tid + selSize] = sMem[tid + selSize];
  memT.team_barrier();

  for (int offset = 1; offset < selSize; offset *= 2) {
    if (tid >= offset) {
        opArray[tid + selSize - 1] = 
            elementOp<opType>(opArray[tid + selSize - 1], opArray[tid + selSize - 1 - offset]);
    }
    if (tid <= selSize - 1 - offset) {
        opArray[tid] = elementOp<opType>(opArray[tid], opArray[tid + offset]);
    }
    memT.team_barrier();
  }
}

template <MorphOpType opType>
KOKKOS_IMPL_FUNCTION void vhgw_horiz(unsigned char* __restrict dst,
                const unsigned char* __restrict src,
                unsigned char* __restrict sMem,
                const int width,
                const int height,
                const int selSize,
                const int gridSizeX,
                const int localSizeX,
                Kokkos::TeamPolicy<ExecSpace>::member_type memT)
{
  unsigned char* opArray = sMem + 2 * selSize;
  const int gid = memT.league_rank() * memT.team_size() + memT.team_rank();
  const int tidx = gid % gridSizeX;
  const int tidy = gid / gridSizeX;
  const int lidx = memT.team_rank() % localSizeX;

  if (tidx >= width || tidy >= height) return;

  sMem[lidx] = src[tidy * width + tidx];
  if (tidx + selSize < width) {
    sMem[lidx + selSize] = src[tidy * width + tidx + selSize];
  }
  memT.team_barrier();

  opArray[lidx] = sMem[lidx];
  opArray[lidx + selSize] = sMem[lidx + selSize];
  memT.team_barrier();

  for (int offset = 1; offset < selSize; offset *= 2) {
    if (lidx >= offset) {
        opArray[lidx + selSize - 1] = 
            elementOp<opType>(opArray[lidx + selSize - 1], opArray[lidx + selSize - 1 - offset]);
    }
    if (lidx <= selSize - 1 - offset) {
        opArray[lidx] = elementOp<opType>(opArray[lidx], opArray[lidx + offset]);
    }
    memT.team_barrier();
  }

  if (tidx + selSize/2 < width - selSize/2) {
    dst[tidy * width + tidx + selSize/2] = 
      elementOp<opType>(opArray[lidx], opArray[lidx + selSize - 1]);
  }
}

template <MorphOpType opType>
KOKKOS_IMPL_FUNCTION void vhgw_vert(unsigned char* __restrict dst,
               const unsigned char* __restrict src,
               unsigned char* __restrict sMem,
               const int width,
               const int height,
               const int selSize,
               const int gridSizeX,
               const int localSizeX,
               Kokkos::TeamPolicy<ExecSpace>::member_type memT)
{
  unsigned char* opArray = sMem + 2 * selSize;
  const int gid = memT.league_rank() * memT.team_size() + memT.team_rank();
  const int tidx = gid % gridSizeX;
  const int tidy = gid / gridSizeX;
  const int lidy = memT.team_rank() / localSizeX;

  if (tidy >= height || tidx >= width) return;

  sMem[lidy] = src[tidy * width + tidx];
  if (tidy + selSize < height) {
    sMem[lidy + selSize] = src[(tidy + selSize) * width + tidx];
  }
  memT.team_barrier();

  opArray[lidy] = sMem[lidy];
  opArray[lidy + selSize] = sMem[lidy + selSize];
  memT.team_barrier();

  for (int offset = 1; offset < selSize; offset *= 2) {
    if (lidy >= offset) {
        opArray[lidy + selSize - 1] = 
            elementOp<opType>(opArray[lidy + selSize - 1], opArray[lidy + selSize - 1 - offset]);
    }
    if (lidy <= selSize - 1 - offset) {
        opArray[lidy] = elementOp<opType>(opArray[lidy], opArray[lidy + offset]);
    }
    memT.team_barrier();
  }

  if (tidy + selSize/2 < height - selSize/2) {
    dst[(tidy + selSize/2) * width + tidx] = 
      elementOp<opType>(opArray[lidy], opArray[lidy + selSize - 1]);
  }

  if (tidy < selSize/2 || tidy >= height - selSize/2) {
    dst[tidy * width + tidx] = borderValue<opType>();
  }
}

template <MorphOpType opType>
double morphology(
        unsigned char *img_d,
        unsigned char *tmp_d,
        unsigned char* initU,
        const int width,
        const int height,
        const int hsize,
        const int vsize)
{
  unsigned int memSize = width * height * sizeof(unsigned char);
  Kokkos::Impl::DeepCopy<MemSpace, host_memory>(tmp_d, initU, memSize);

  int blockSize_x = hsize;
  int blockSize_y = 1;
  int gridSize_x = roundUp(width, blockSize_x);
  int gridSize_y = roundUp(height, blockSize_y);
  size_t h_gws = gridSize_y * gridSize_x;
  size_t h_lws = blockSize_y * blockSize_x;

  blockSize_x = 1;
  blockSize_y = vsize;
  gridSize_x = roundUp(width, blockSize_x);
  gridSize_y = roundUp(height, blockSize_y);
  size_t v_gws = gridSize_y * gridSize_x;
  size_t v_lws = blockSize_y * blockSize_x;

  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();

  Kokkos::parallel_for("horiz", 
  Kokkos::TeamPolicy<ExecSpace>(h_gws, h_lws)
  .set_scratch_size(0, Kokkos::PerTeam(4*hsize*sizeof(unsigned char))), 
  KOKKOS_LAMBDA(Kokkos::TeamPolicy<ExecSpace>::member_type memT){
    unsigned char* sMem = (unsigned char*) memT.team_shmem().get_shmem(4*hsize*sizeof(unsigned char));
    vhgw_horiz<opType>(tmp_d, img_d, sMem,
                        width, height, hsize, gridSize_x, blockSize_x, memT);
  });

  Kokkos::parallel_for("vert", 
  Kokkos::TeamPolicy<ExecSpace>(v_gws, v_lws)
  .set_scratch_size(0, Kokkos::PerTeam(4*vsize*sizeof(unsigned char))), 
  KOKKOS_LAMBDA(Kokkos::TeamPolicy<ExecSpace>::member_type memT){
    unsigned char* sMem = (unsigned char*) memT.team_shmem().get_shmem(4*hsize*sizeof(unsigned char));
    vhgw_vert<opType>(tmp_d, img_d, sMem,
                        width, height, hsize, gridSize_x, blockSize_x, memT);
  });

  Kokkos::fence();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return time;
}

extern "C"
double erode(
        unsigned char *img_d,
        unsigned char *tmp_d,
        unsigned char* initU,
        const int width,
        const int height,
        const int hsize,
        const int vsize)
{
  return morphology<MorphOpType::ERODE>(img_d, tmp_d, initU, width, height, hsize, vsize);
}

extern "C"
double dilate(
        unsigned char *img_d,
        unsigned char *tmp_d,
        unsigned char* initU,
        const int width,
        const int height,
        const int hsize,
        const int vsize)
{
  return morphology<MorphOpType::DILATE>(img_d, tmp_d, initU, width, height, hsize, vsize);
}
