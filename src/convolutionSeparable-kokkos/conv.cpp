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

#include <assert.h>
#include <Kokkos_Core.hpp>
#include "conv.hpp"

#define ROWS_BLOCKDIM_X       16
#define COLUMNS_BLOCKDIM_X    16
#define ROWS_BLOCKDIM_Y       4
#define COLUMNS_BLOCKDIM_Y    8
#define ROWS_RESULT_STEPS     8
#define COLUMNS_RESULT_STEPS  8
#define ROWS_HALO_STEPS       1
#define COLUMNS_HALO_STEPS    1

void convolutionRows(
    Kokkos::View<float*, Layout, MemSpace> dst,
    Kokkos::View<float*, Layout, MemSpace> src,
    Kokkos::View<float*, Layout, MemSpace> kernel,
    const int imageW,
    const int imageH,
    const int pitch
)
{
    assert ( ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS );
    assert ( imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0 );
    assert ( imageH % ROWS_BLOCKDIM_Y == 0 );

    int lws = ROWS_BLOCKDIM_Y * ROWS_BLOCKDIM_X;
    int gws = ((imageH * imageW / ROWS_RESULT_STEPS) + (lws-1))/lws;
    size_t ssize1 = ROWS_BLOCKDIM_Y;
    size_t ssize2 = (ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X;
    size_t ssize = ssize1 * ssize2;
    size_t shared_size = shared_float::shmem_size(ssize);

    Kokkos::parallel_for("conv_rows", 
    Kokkos::TeamPolicy<ExecSpace>(gws, lws)
    .set_scratch_size(0, Kokkos::PerTeam(shared_size)), 
    KOKKOS_LAMBDA(Kokkos::TeamPolicy<ExecSpace>::member_type memT){
      const int localId = memT.team_rank();
      int lidX = localId % ROWS_BLOCKDIM_X;
      int lidY = localId / ROWS_BLOCKDIM_X;
      //Offset to the left halo edge
      const int gid = memT.league_rank() * memT.team_size() + memT.team_rank();
      int baseX = gid % imageW;
      int gidX = (baseX - lidX)/ROWS_BLOCKDIM_X;
      baseX = (gidX * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + lidX;
      const int baseY = gid / imageW;
      shared_float l_Data(memT.team_scratch(0), ssize);

      const size_t src_new = baseY * pitch + baseX;
      const size_t dst_new = baseY * pitch + baseX;

      //Load main data
      #pragma unroll
      for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
          l_Data[lidY * ssize2 + lidX + i * ROWS_BLOCKDIM_X] = src[src_new + i * ROWS_BLOCKDIM_X];

      //Load left halo
      #pragma unroll
      for(int i = 0; i < ROWS_HALO_STEPS; i++)
          l_Data[lidY * ssize2 + lidX + i * ROWS_BLOCKDIM_X] = (baseX + i * ROWS_BLOCKDIM_X >= 0) ? src[src_new + i * ROWS_BLOCKDIM_X] : 0;

      //Load right halo
      #pragma unroll
      for(int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
          l_Data[lidY * ssize2 + lidX + i * ROWS_BLOCKDIM_X] = (baseX + i * ROWS_BLOCKDIM_X < imageW) ? src[src_new + i * ROWS_BLOCKDIM_X] : 0;

      //Compute and store results
      memT.team_barrier();

      #pragma unroll
      for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
          float sum = 0;

          #pragma unroll
          for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
              sum += kernel[KERNEL_RADIUS - j] * l_Data[lidY * ssize2 + lidX + i * ROWS_BLOCKDIM_X + j];

          dst[dst_new + i * ROWS_BLOCKDIM_X] = sum;
      }
    });
}

void convolutionColumns(
    Kokkos::View<float*, Layout, MemSpace> dst,
    Kokkos::View<float*, Layout, MemSpace> src,
    Kokkos::View<float*, Layout, MemSpace> kernel,
    const int imageW,
    const int imageH,
    const int pitch
)
{
    assert ( COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS );
    assert ( imageW % COLUMNS_BLOCKDIM_X == 0 );
    assert ( imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0 );

    int lws = COLUMNS_BLOCKDIM_Y * COLUMNS_BLOCKDIM_X;
    int gws = ((imageH / COLUMNS_RESULT_STEPS * imageW) + (lws-1))/lws;
    size_t ssize1 = COLUMNS_BLOCKDIM_X;
    size_t ssize2 = (COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1;
    size_t ssize = ssize1 * ssize2;
    size_t shared_size = shared_float::shmem_size(ssize);

    Kokkos::parallel_for("conv_cols", 
    Kokkos::TeamPolicy<ExecSpace>(gws, lws)
    .set_scratch_size(0, Kokkos::PerTeam(shared_size)), 
    KOKKOS_LAMBDA(Kokkos::TeamPolicy<ExecSpace>::member_type memT){
      const int localId = memT.team_rank();
      int lidX = localId % COLUMNS_BLOCKDIM_X;
      int lidY = localId / COLUMNS_BLOCKDIM_X;

      //Offset to the upper halo edge
      const int gid = memT.league_rank() * memT.team_size() + memT.team_rank();
      const int baseX = gid % imageW;
      int baseY = gid / imageW;
      int gidY = (baseY - lidY)/COLUMNS_BLOCKDIM_Y;
      baseY = (gidY * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + lidY;
      shared_float l_Data(memT.team_scratch(0), ssize);

      const size_t src_new = baseY * pitch + baseX;
      const size_t dst_new = baseY * pitch + baseX;

      //Load main data
      #pragma unroll
      for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
          l_Data[lidX * ssize2 + lidY + i * COLUMNS_BLOCKDIM_Y] = src[src_new + i * COLUMNS_BLOCKDIM_Y * pitch];

      //Load upper halo
      #pragma unroll
      for(int i = 0; i < COLUMNS_HALO_STEPS; i++)
          l_Data[lidX * ssize2 + lidY + i * COLUMNS_BLOCKDIM_Y] = (baseY + i * COLUMNS_BLOCKDIM_Y >= 0) ? src[src_new + i * COLUMNS_BLOCKDIM_Y * pitch] : 0;

      //Load lower halo
      #pragma unroll
      for(int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
          l_Data[lidX * ssize2 + lidY + i * COLUMNS_BLOCKDIM_Y] = (baseY + i * COLUMNS_BLOCKDIM_Y < imageH) ? src[src_new + i * COLUMNS_BLOCKDIM_Y * pitch] : 0;

      //Compute and store results
      memT.team_barrier();

      #pragma unroll
      for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
          float sum = 0;

          #pragma unroll
          for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
              sum += kernel[KERNEL_RADIUS - j] * l_Data[lidX * ssize2 + lidY + i * COLUMNS_BLOCKDIM_Y + j];

          dst[dst_new + i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
      }
    });
}
