#include <Kokkos_Core.hpp>
#include "kernels.h"

/* ################# wrappers ################### */

void compute_costs(
    int current_w, int w, int h,
    pixel *d_pixels,
    short *d_costs_left,
    short *d_costs_up,
    short *d_costs_right)
{
#ifndef COMPUTE_COSTS_FULL

  size_t lws = COSTS_BLOCKSIZE_Y * COSTS_BLOCKSIZE_X;
  size_t gwsx = (current_w-1)/(COSTS_BLOCKSIZE_X-2) + 1;
  size_t gwsy = (h-1)/(COSTS_BLOCKSIZE_Y-1) + 1;

  Kokkos::parallel_for("compute_costs", 
  Kokkos::TeamPolicy<ExecSpace>(gwsx*gwsy, lws)
  .set_scratch_size(0, Kokkos::PerTeam(COSTS_BLOCKSIZE_Y * COSTS_BLOCKSIZE_X * sizeof(pixel))), 
  KOKKOS_LAMBDA(Kokkos::TeamPolicy<ExecSpace>::member_type memT){
    pixel* sm = (pixel*) memT.team_shmem().get_shmem(COSTS_BLOCKSIZE_Y * COSTS_BLOCKSIZE_X*sizeof(pixel));
    compute_costs_kernel(memT,
                        sm,
                        d_pixels,
                        d_costs_left,
                        d_costs_up,
                        d_costs_right,
                        w, h, current_w);
  });

#else

  size_t lws = COSTS_BLOCKSIZE_Y * COSTS_BLOCKSIZE_X;
  size_t gwsx = (current_w-1)/COSTS_BLOCKSIZE_X + 1;
  size_t gwsy = (h-1)/COSTS_BLOCKSIZE_Y + 1;

  Kokkos::parallel_for("compute_costs_full", 
  Kokkos::TeamPolicy<ExecSpace>(gwsx*gwsy, lws)
  .set_scratch_size(0, Kokkos::PerTeam((COSTS_BLOCKSIZE_Y+1) * (COSTS_BLOCKSIZE_X+2) * sizeof(pixel))), 
  KOKKOS_LAMBDA(Kokkos::TeamPolicy<ExecSpace>::member_type memT){
    pixel* sm = (pixel*) memT.team_shmem().get_shmem((COSTS_BLOCKSIZE_Y+1) * (COSTS_BLOCKSIZE_X+2)*sizeof(pixel));
    compute_costs_full_kernel(memT,
                              sm,
                              d_pixels,
                              d_costs_left,
                              d_costs_up,
                              d_costs_right,
                              w, h, current_w);
  });
#endif
}

void compute_M(
    int current_w, int w, int h,
    int *d_M,
    short *d_costs_left,
    short *d_costs_up,
    short *d_costs_right)
{
#if !defined(COMPUTE_M_SINGLE) && !defined(COMPUTE_M_ITERATE)

  if(current_w <= 256){
    //compute_M_kernel_small<<<num_blocks, threads_per_block, 2*current_w*sizeof(int)>>>(d_costs_left, d_costs_up, d_costs_right, d_M, w, h, current_w);
    Kokkos::parallel_for("compute_M_small", 
    Kokkos::TeamPolicy<ExecSpace>(1, current_w)
    .set_scratch_size(0, Kokkos::PerTeam(2*current_w * sizeof(int))), 
    KOKKOS_LAMBDA(Kokkos::TeamPolicy<ExecSpace>::member_type memT){
      int* sm = (int*) memT.team_shmem().get_shmem(2*current_w*sizeof(int));
      compute_M_kernel_small(memT,
                              sm,
                              d_costs_left,
                              d_costs_up,
                              d_costs_right,
                              d_M,
                              w, h, current_w);
    });
  }
  else{
    int num_iterations = (h-1)/(COMPUTE_M_BLOCKSIZE_X/2 - 1) + 1;

    int base_row = 0;
    for(int i = 0; i < num_iterations; i++){
      //compute_M_kernel_step1<<<num_blocks, threads_per_block>>>(d_costs_left, d_costs_up, d_costs_right, d_M, w, h, current_w, base_row);
      Kokkos::parallel_for("compute_M_step1", 
      Kokkos::TeamPolicy<ExecSpace>((current_w-1)/COMPUTE_M_BLOCKSIZE_X + 1, COMPUTE_M_BLOCKSIZE_X)
      .set_scratch_size(0, Kokkos::PerTeam(2*COMPUTE_M_BLOCKSIZE_X * sizeof(int))), 
      KOKKOS_LAMBDA(Kokkos::TeamPolicy<ExecSpace>::member_type memT){
        int* sm = (int*) memT.team_shmem().get_shmem(2*COMPUTE_M_BLOCKSIZE_X*sizeof(int));
        compute_M_kernel_step1(memT,
                          sm,
                          d_costs_left,
                          d_costs_up,
                          d_costs_right,
                          d_M,
                          w, h, current_w, base_row);
      });

      //compute_M_kernel_step2<<<num_blocks2, threads_per_block>>>(d_costs_left, d_costs_up, d_costs_right, d_M, w, h, current_w, base_row);
      Kokkos::parallel_for("compute_M_step2", 
      Kokkos::TeamPolicy<ExecSpace>((current_w-COMPUTE_M_BLOCKSIZE_X-1)/COMPUTE_M_BLOCKSIZE_X + 1, COMPUTE_M_BLOCKSIZE_X), 
      KOKKOS_LAMBDA(Kokkos::TeamPolicy<ExecSpace>::member_type memT){
        compute_M_kernel_step2(memT,
                                d_costs_left,
                                d_costs_up,
                                d_costs_right,
                                d_M,
                                w, h, current_w, base_row);
      });
      base_row = base_row + (COMPUTE_M_BLOCKSIZE_X/2) - 1;
    }
  }
#endif
#ifdef COMPUTE_M_SINGLE

  int block_size = std::min(256, next_pow2(current_w));
  size_t lws (block_size);
  size_t gws (1);

  int num_el = (current_w-1)/block_size + 1;
  //compute_M_kernel_single<<<num_blocks, threads_per_block, 2*current_w*sizeof(int)>>>(d_costs_left, d_costs_up, d_costs_right, d_M, w, h, current_w, num_el);
  Kokkos::parallel_for("compute_M_single", 
  Kokkos::TeamPolicy<ExecSpace>(gws, lws)
  .set_scratch_size(0, Kokkos::PerTeam(2*current_w * sizeof(int))), 
  KOKKOS_LAMBDA(Kokkos::TeamPolicy<ExecSpace>::member_type memT){
    int* sm = (int*) memT.team_shmem().get_shmem(2*current_w*sizeof(int));
    compute_M_kernel_single(memT,
                            sm.get_pointer(),
                            d_costs_left,
                            d_costs_up,
                            d_costs_right,
                            d_M,
                            w, h, current_w, num_el);
  });
#else
#ifdef COMPUTE_M_ITERATE

  size_t gws ((current_w-1)/COMPUTE_M_BLOCKSIZE_X + 1);
  size_t lws (COMPUTE_M_BLOCKSIZE_X);

  //compute_M_kernel_iterate0<<<num_blocks, threads_per_block>>>(d_costs_left, d_costs_up, d_costs_right, d_M, w, current_w);
  Kokkos::parallel_for("compute_M_iter0", 
  Kokkos::TeamPolicy<ExecSpace>(gws, lws), 
  KOKKOS_LAMBDA(Kokkos::TeamPolicy<ExecSpace>::member_type memT){
      compute_M_kernel_iterate0(memT,
                                d_costs_left,
                                d_costs_up,
                                d_costs_right,
                                d_M,
                                w, current_w);
  }); 

  for(int row = 1; row < h; row++){
  //  compute_M_kernel_iterate1<<<num_blocks, threads_per_block>>>(d_costs_left, d_costs_up, d_costs_right, d_M, w, current_w, row);
    Kokkos::parallel_for("compute_M_iter1", 
    Kokkos::TeamPolicy<ExecSpace>((current_w-1)/COMPUTE_M_BLOCKSIZE_X + 1, COMPUTE_M_BLOCKSIZE_X), 
    KOKKOS_LAMBDA(Kokkos::TeamPolicy<ExecSpace>::member_type memT){
        compute_M_kernel_iterate1(memT,
                                  d_costs_left,
                                  d_costs_up,
                                  d_costs_right,
                                  d_M,
                                  w, current_w, row);
    }); 
  }

#endif
#endif
}

void find_min_index(
    int current_w,
    int *d_indices_ref,
    int *d_indices,
    int *reduce_row)
{
  //set the reference index array
  Kokkos::Impl::DeepCopy<MemSpace, MemSpace>(d_indices, d_indices_ref, current_w*sizeof(int));
  Kokkos::fence();

  size_t lws (REDUCE_BLOCKSIZE_X);
  size_t gws (1);

  int reduce_num_elements = current_w;
  do{
    int num_blocks_x = (reduce_num_elements-1)/(REDUCE_BLOCKSIZE_X*REDUCE_ELEMENTS_PER_THREAD) + 1;
    gws = num_blocks_x;
    // min_reduce<<<num_blocks, threads_per_block>>>(reduce_row, d_indices, reduce_num_elements);
    Kokkos::parallel_for("reduce", 
    Kokkos::TeamPolicy<ExecSpace>(gws, lws)
    .set_scratch_size(0, Kokkos::PerTeam(REDUCE_BLOCKSIZE_X * sizeof(int) + REDUCE_BLOCKSIZE_X * sizeof(int))), 
    KOKKOS_LAMBDA(Kokkos::TeamPolicy<ExecSpace>::member_type memT){
      int* sm_val = (int*) memT.team_shmem().get_shmem(COMPUTE_M_BLOCKSIZE_X*sizeof(int));
      int* sm_ix = (int*) memT.team_shmem().get_shmem(COMPUTE_M_BLOCKSIZE_X*sizeof(int));
      min_reduce(memT,
                sm_val,
                sm_ix,
                reduce_row,
                d_indices,
                reduce_num_elements);
    });
    reduce_num_elements = num_blocks_x;
  }while(reduce_num_elements > 1);
}

void find_seam(
    int current_w, int w, int h,
    int *d_M,
    int *d_indices,
    int *d_seam )
{
  //find_seam_kernel<<<1, 1>>>(d_M, d_indices, d_seam, w, h, current_w);
  Kokkos::parallel_for("find_seam", 
  Kokkos::RangePolicy<ExecSpace>(0, 1),
  KOKKOS_LAMBDA(int x){
    find_seam_kernel(d_M, d_indices, d_seam, w, h, current_w);
  });
}

void remove_seam(
    int current_w, int w, int h,
    int *d_M,
    pixel *d_pixels,
    pixel *d_pixels_swap,
    int *d_seam )
{
  int num_blocks_x = (current_w-1)/REMOVE_BLOCKSIZE_X + 1;
  int num_blocks_y = (h-1)/REMOVE_BLOCKSIZE_Y + 1;
  size_t lws1 = REMOVE_BLOCKSIZE_Y;
  size_t lws2 = REMOVE_BLOCKSIZE_X;
  size_t gws1 = REMOVE_BLOCKSIZE_Y * num_blocks_y;
  size_t gws2 = REMOVE_BLOCKSIZE_X * num_blocks_x;

  //remove_seam_kernel<<<num_blocks, threads_per_block>>>(d_pixels, d_pixels_swap, d_seam, w, h, current_w);
  Kokkos::parallel_for("update_seam", 
  Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>> ({0,0}, {gws1, gws2}, {lws1,lws2}), 
  KOKKOS_LAMBDA(const int y, const int x){
    remove_seam_kernel(y, x,
                        d_pixels,
                        d_pixels_swap,
                        d_seam,
                        w, h, current_w);
  });
}

void update_costs(
    int current_w, int w, int h,
    int *d_M,
    pixel *d_pixels,
    short *d_costs_left,
    short *d_costs_up,
    short *d_costs_right,
    short *d_costs_swap_left,
    short *d_costs_swap_up,
    short *d_costs_swap_right,
    int *d_seam )
{
  int num_blocks_x = (current_w-1)/UPDATE_BLOCKSIZE_X + 1;
  int num_blocks_y = (h-1)/UPDATE_BLOCKSIZE_Y + 1;
  size_t lws1 = UPDATE_BLOCKSIZE_Y;
  size_t lws2 = UPDATE_BLOCKSIZE_X;
  size_t gws1 = UPDATE_BLOCKSIZE_Y * num_blocks_y;
  size_t gws2 = UPDATE_BLOCKSIZE_X * num_blocks_x;

  Kokkos::parallel_for("update_costs", 
  Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>> ({0,0}, {gws1, gws2}, {lws1,lws2}), 
  KOKKOS_LAMBDA(const int y, const int x){
      update_costs_kernel (y, x,
                           d_pixels,
                           d_costs_left,
                           d_costs_up,
                           d_costs_right,
                           d_costs_swap_left,
                           d_costs_swap_up,
                           d_costs_swap_right,
                           d_seam,
                           w, h, current_w);
  });
}

void approx_setup(
    int current_w, int w, int h,
    pixel *d_pixels,
    int *d_index_map,
    int *d_offset_map,
    int *d_M )
{
  int num_blocks_x = (current_w-1)/(APPROX_SETUP_BLOCKSIZE_X-4) + 1;
  int num_blocks_y = (h-2)/(APPROX_SETUP_BLOCKSIZE_Y-1) + 1;
  size_t lws (APPROX_SETUP_BLOCKSIZE_Y*APPROX_SETUP_BLOCKSIZE_X);
  size_t gws (num_blocks_y * num_blocks_x);

  const size_t sm_size (APPROX_SETUP_BLOCKSIZE_Y*APPROX_SETUP_BLOCKSIZE_X);

  Kokkos::parallel_for("reduce", 
  Kokkos::TeamPolicy<ExecSpace>(gws, lws)
  .set_scratch_size(0, Kokkos::PerTeam(sm_size * sizeof(short)*3 + sm_size * sizeof(pixel))), 
  KOKKOS_LAMBDA(Kokkos::TeamPolicy<ExecSpace>::member_type memT){
    pixel* p_sm = (pixel*) memT.team_shmem().get_shmem(sm_size*sizeof(pixel));
    short* l_sm = (short*) memT.team_shmem().get_shmem(sm_size*sizeof(short));
    short* u_sm = (short*) memT.team_shmem().get_shmem(sm_size*sizeof(short));
    short* r_sm = (short*) memT.team_shmem().get_shmem(sm_size*sizeof(short));
    approx_setup_kernel(memT,
                        p_sm,
                        l_sm,
                        u_sm,
                        r_sm,
                        d_pixels,
                        d_index_map,
                        d_offset_map,
                        d_M,
                        w, h, current_w);
  });
}

void approx_M(int current_w, int w, int h, int *d_offset_map, int *d_M) {
  int num_blocks_x = (current_w - 1) / APPROX_M_BLOCKSIZE_X + 1;
  int num_blocks_y = h / 2;

  int step = 1;
  while (num_blocks_y > 0) {
    Kokkos::parallel_for("approx_M", Kokkos::TeamPolicy<ExecSpace>(num_blocks_y*num_blocks_x, APPROX_M_BLOCKSIZE_X),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<ExecSpace>::member_type& team) {
        int row = team.league_rank() * 2 * step;
        int next_row = row + step;

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, APPROX_M_BLOCKSIZE_X), [=] (const int column) {
          int ix = row * w + column;
          
          if (next_row < h - 1 && column < current_w) {
            int offset = d_offset_map[ix];
            d_M[ix] += d_M[offset];
            d_offset_map[ix] = d_offset_map[offset];
          }
        });
      });

    num_blocks_y = num_blocks_y / 2;
    step = step * 2;
  }
}

void approx_seam(
    int w, int h,
    int *d_index_map,
    int *d_indices,
    int *d_seam )
{
  //approx_seam_kernel<<<1, 1>>>(d_index_map, d_indices, d_seam, w, h);
  Kokkos::parallel_for("approx_seam", 
  Kokkos::RangePolicy<ExecSpace>(0, 1),
  KOKKOS_LAMBDA(int x){
    approx_seam_kernel(d_index_map, d_indices, d_seam, w, h);
  });
}
