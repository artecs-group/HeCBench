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

const int COSTS_BLOCKSIZE_X = 32;
const int COSTS_BLOCKSIZE_Y = 8;

const int COMPUTE_M_BLOCKSIZE_X = 128; //must be divisible by 2

const int REDUCE_BLOCKSIZE_X = 128;
const int REDUCE_ELEMENTS_PER_THREAD = 8;

const int REMOVE_BLOCKSIZE_X = 32;
const int REMOVE_BLOCKSIZE_Y = 8;

const int UPDATE_BLOCKSIZE_X = 32;
const int UPDATE_BLOCKSIZE_Y = 8;

const int APPROX_SETUP_BLOCKSIZE_X = 32;
const int APPROX_SETUP_BLOCKSIZE_Y = 8;

const int APPROX_M_BLOCKSIZE_X = 128;

#define BORDER_PIXEL {0,0,0}

#define syncthreads() memT.team_barrier();

void pointer_swap(void **p1, void **p2){
  void *tmp;
  tmp = *p1;
  *p1 = *p2;
  *p2 = tmp;
}

KOKKOS_INLINE_FUNCTION void compute_costs_kernel(
    Kokkos::TeamPolicy<ExecSpace>::member_type memT,
    pixel *__restrict__ pix_cache,
    const pixel *__restrict__ d_pixels,
    short *__restrict__ d_costs_left,
    short *__restrict__ d_costs_up,
    short *__restrict__ d_costs_right,
    int w, int h, int current_w)
{
  //first row, first column and last column of shared memory are reserved for halo...
  //...and the global index in the image is computed accordingly to this
  int blockIdx_y = memT.league_rank() / COSTS_BLOCKSIZE_X;
  int blockIdx_x = memT.league_rank() % COSTS_BLOCKSIZE_X;
  int threadIdx_y = memT.team_rank() / (current_w-1);
  int threadIdx_x = memT.team_rank() % (current_w-1);
  int row = blockIdx_y*(COSTS_BLOCKSIZE_Y-1) + threadIdx_y -1 ;
  int column = blockIdx_x*(COSTS_BLOCKSIZE_X-2) + threadIdx_x -1;
  int ix = row*w + column;
  int cache_row = threadIdx_y;
  int cache_column = threadIdx_x;
  short active = 0;

  if(row >= 0 && row < h && column >= 0 && column < current_w){
    active = 1;
    pix_cache[cache_row * COSTS_BLOCKSIZE_X + cache_column] = d_pixels[ix];
  }
  else{
    pix_cache[cache_row * COSTS_BLOCKSIZE_X + cache_column] = BORDER_PIXEL;
  }

  //wait until each thread has initialized its portion of shared memory
  syncthreads();

  //all the threads that are NOT in halo positions can now compute costs, with fast access to shared memory
  if(active && cache_row != 0 && cache_column != 0 && cache_column != COSTS_BLOCKSIZE_X-1){
    int rdiff, gdiff, bdiff;
    int p_r, p_g, p_b;
    pixel pix1, pix2, pix3;

    pix1 = pix_cache[cache_row * COSTS_BLOCKSIZE_X + cache_column+1];
    pix2 = pix_cache[cache_row * COSTS_BLOCKSIZE_X + cache_column-1];
    pix3 = pix_cache[(cache_row-1) * COSTS_BLOCKSIZE_X + cache_column];

    //compute partials
    p_r = Kokkos::abs(pix1.r - pix2.r);
    p_g = Kokkos::abs(pix1.g - pix2.g);
    p_b = Kokkos::abs(pix1.b - pix2.b);

    //compute left cost
    rdiff = p_r + Kokkos::abs(pix3.r - pix2.r);
    gdiff = p_g + Kokkos::abs(pix3.g - pix2.g);
    bdiff = p_b + Kokkos::abs(pix3.b - pix2.b);
    d_costs_left[ix] = rdiff + gdiff + bdiff;

    //compute up cost
    d_costs_up[ix] = p_r + p_g + p_b;

    //compute right cost
    rdiff = p_r + Kokkos::abs(pix3.r - pix1.r);
    gdiff = p_g + Kokkos::abs(pix3.g - pix1.g);
    bdiff = p_b + Kokkos::abs(pix3.b - pix1.b);
    d_costs_right[ix] = rdiff + gdiff + bdiff;
  }
}

KOKKOS_INLINE_FUNCTION void compute_costs_full_kernel(
    Kokkos::TeamPolicy<ExecSpace>::member_type memT,
    pixel *__restrict__ pix_cache,
    const pixel* __restrict__ d_pixels,
    short *__restrict__ d_costs_left,
    short *__restrict__ d_costs_up,
    short *__restrict__ d_costs_right,
    int w, int h, int current_w)
{
  int blockIdx_y = memT.league_rank() / COSTS_BLOCKSIZE_X;
  int blockIdx_x = memT.league_rank() % COSTS_BLOCKSIZE_X;
  int threadIdx_y = memT.team_rank() / (current_w-1);
  int threadIdx_x = memT.team_rank() % (current_w-1);
  int row = blockIdx_y*COSTS_BLOCKSIZE_Y + threadIdx_y;
  int column = blockIdx_x*COSTS_BLOCKSIZE_X + threadIdx_x;
  int ix = row*w + column;
  int cache_row = threadIdx_y + 1;
  int cache_column = threadIdx_x + 1;
  short active = 0;

  if(row < h && column < current_w){
    active = 1;
    if(threadIdx_x == 0){
      if(column == 0)
        pix_cache[cache_row * (COSTS_BLOCKSIZE_X+2)] = BORDER_PIXEL;
      else
        pix_cache[cache_row * (COSTS_BLOCKSIZE_X+2)] = d_pixels[ix-1];
    }
    if(threadIdx_x == COSTS_BLOCKSIZE_X-1 || column == current_w-1){
      if(column == current_w-1)
        pix_cache[cache_row * (COSTS_BLOCKSIZE_X+2) + cache_column+1] = BORDER_PIXEL;
      else
        pix_cache[cache_row * (COSTS_BLOCKSIZE_X+2) + COSTS_BLOCKSIZE_X+1] = d_pixels[ix+1];
    }
    if(threadIdx_y == 0){
      if(row == 0)
        pix_cache[cache_column] = BORDER_PIXEL;
      else
        pix_cache[cache_column] = d_pixels[ix-w];
    }
    pix_cache[cache_row * (COSTS_BLOCKSIZE_X+2) + cache_column] = d_pixels[ix];
  }

  syncthreads();

  if(active){
    int rdiff, gdiff, bdiff;
    int p_r, p_g, p_b;
    pixel pix1, pix2, pix3;

    pix1 = pix_cache[cache_row * (COSTS_BLOCKSIZE_X+2) + cache_column+1];
    pix2 = pix_cache[cache_row * (COSTS_BLOCKSIZE_X+2) + cache_column-1];
    pix3 = pix_cache[(cache_row-1) * (COSTS_BLOCKSIZE_X+2) + cache_column];

    //compute partials
    p_r = Kokkos::abs(pix1.r - pix2.r);
    p_g = Kokkos::abs(pix1.g - pix2.g);
    p_b = Kokkos::abs(pix1.b - pix2.b);

    //compute left cost
    rdiff = p_r + Kokkos::abs(pix3.r - pix2.r);
    gdiff = p_g + Kokkos::abs(pix3.g - pix2.g);
    bdiff = p_b + Kokkos::abs(pix3.b - pix2.b);
    d_costs_left[ix] = rdiff + gdiff + bdiff;

    //compute up cost
    d_costs_up[ix] = p_r + p_g + p_b;

    //compute right cost
    rdiff = p_r + Kokkos::abs(pix3.r - pix1.r);
    gdiff = p_g + Kokkos::abs(pix3.g - pix1.g);
    bdiff = p_b + Kokkos::abs(pix3.b - pix1.b);
    d_costs_right[ix] = rdiff + gdiff + bdiff;
  }
}

KOKKOS_INLINE_FUNCTION void compute_M_kernel_step1(
    Kokkos::TeamPolicy<ExecSpace>::member_type memT,
    int *__restrict__  cache,
    const short *__restrict__ d_costs_left,
    const short *__restrict__ d_costs_up,
    const short *__restrict__ d_costs_right,
    int* __restrict__ d_M,
    int w, int h, int current_w, int base_row)
{
  int blockIdx_x = memT.league_rank();
  int threadIdx_x = memT.team_rank();
  int gridDim_x = memT.team_size();
  int *m_cache = cache;
  int *m_cache_swap = &(cache[COMPUTE_M_BLOCKSIZE_X]);
  int column = blockIdx_x*COMPUTE_M_BLOCKSIZE_X + threadIdx_x;
  int ix = base_row*w + column;
  int cache_column = threadIdx_x;
  int right, up, left;

  short is_first = blockIdx_x == 0;
  short is_last = blockIdx_x == gridDim_x-1;

  if(column < current_w){
    if(base_row == 0){
      left = Kokkos::min(d_costs_left[ix], Kokkos::min(d_costs_up[ix], d_costs_right[ix]));
      m_cache[cache_column] = left;
      d_M[ix] = left;
    }
    else{
      m_cache[cache_column] = d_M[ix];
    }
  }

  syncthreads();

  int max_row = base_row + COMPUTE_M_BLOCKSIZE_X/2;
  for(int row = base_row+1, inc = 1; row < max_row && row < h; row++, inc++){
    ix = ix + w;
    if(column < current_w && (is_first || inc <= threadIdx_x) && (is_last || threadIdx_x < COMPUTE_M_BLOCKSIZE_X - inc)){

      //with left
      if(column > 0)
        left = m_cache[cache_column - 1] + d_costs_left[ix];
      else
        left = INT_MAX;
      //with up
      up = m_cache[cache_column] + d_costs_up[ix];
      //with right
      if(column < current_w-1)
        right = m_cache[cache_column + 1] + d_costs_right[ix];
      else
        right = INT_MAX;

      left = Kokkos::min(left, Kokkos::min(up, right));
      d_M[ix] = left;
      //swap read/write shared memory
      pointer_swap((void**)&m_cache, (void**)&m_cache_swap);
      m_cache[cache_column] = left;
    }
    //wait until every thread has written shared memory
    syncthreads();
  }
}

KOKKOS_INLINE_FUNCTION void compute_M_kernel_step2(
    Kokkos::TeamPolicy<ExecSpace>::member_type memT,
    const short *__restrict__ d_costs_left,
    const short *__restrict__ d_costs_up,
    const short *__restrict__ d_costs_right,
    int* __restrict__ d_M,
    int w, int h, int current_w, int base_row)
{
  int blockIdx_x = memT.league_rank();
  int threadIdx_x = memT.team_rank();
  int column = blockIdx_x*COMPUTE_M_BLOCKSIZE_X + threadIdx_x + COMPUTE_M_BLOCKSIZE_X/2;
  int right, up, left;

  int ix;
  int prev_ix = base_row*w + column;
  int max_row = base_row + COMPUTE_M_BLOCKSIZE_X/2;
  for(int row = base_row+1, inc = 1; row < max_row && row < h; row++, inc++){
    ix = prev_ix + w;
    if(column < current_w && (COMPUTE_M_BLOCKSIZE_X/2 - inc <= threadIdx_x) && (threadIdx_x < COMPUTE_M_BLOCKSIZE_X/2 + inc)){
      //ix = row*w + column;
      //prev_ix = ix - w;

      //with left
      left = d_M[prev_ix - 1] + d_costs_left[ix];
      //with up
      up = d_M[prev_ix] + d_costs_up[ix];
      //with right
      if(column < current_w-1)
        right = d_M[prev_ix + 1] + d_costs_right[ix];
      else
        right = INT_MAX;

      left = Kokkos::min(left, Kokkos::min(up, right));
      d_M[ix] = left;
    }
    prev_ix = ix;
    syncthreads();
  }
}

KOKKOS_INLINE_FUNCTION void compute_M_kernel_small(
    Kokkos::TeamPolicy<ExecSpace>::member_type memT,
    int *__restrict__ cache,
    const short *__restrict__ d_costs_left,
    const short *__restrict__ d_costs_up,
    const short *__restrict__ d_costs_right,
    int* __restrict__ d_M,
    int w, int h, int current_w)
{
  int *m_cache = cache;
  int *m_cache_swap = &(cache[current_w]);
  int column = memT.team_rank();
  int ix = column;
  int left, up, right;

  //first row
  left = Kokkos::min(d_costs_left[ix], Kokkos::min(d_costs_up[ix], d_costs_right[ix]));
  d_M[ix] = left;
  m_cache[ix] = left;

  syncthreads();

  //other rows
  for(int row = 1; row < h; row++){
    if(column < current_w){
      ix = ix + w;//ix = row*w + column;

      //with left
      if(column > 0)
        left = m_cache[column - 1] + d_costs_left[ix];
      else
        left = INT_MAX;
      //with up
      up = m_cache[column] + d_costs_up[ix];
      //with right
      if(column < current_w-1)
        right = m_cache[column + 1] + d_costs_right[ix];
      else
        right = INT_MAX;

      left = Kokkos::min(left, Kokkos::min(up, right));
      d_M[ix] = left;
      //swap read/write shared memory
      pointer_swap((void**)&m_cache, (void**)&m_cache_swap);
      m_cache[column] = left;
    }
    syncthreads();
  }
}

KOKKOS_INLINE_FUNCTION void compute_M_kernel_single(
   Kokkos::TeamPolicy<ExecSpace>::member_type memT,
    int *__restrict__ cache,
    const short *__restrict__ d_costs_left,
    const short *__restrict__ d_costs_up,
    const short *__restrict__ d_costs_right,
    int* __restrict__ d_M,
    int w, int h, int current_w, int n_elem)
{
  int *m_cache = cache;
  int *m_cache_swap = &(cache[current_w]);
  int tid = memT.team_rank();
  int column;
  int ix;
  int left, up, right;

  //first row
  for(int i = 0; i < n_elem; i++){
    column = tid + i*memT.team_size();
    if(column < current_w){
      left = Kokkos::min(d_costs_left[column], Kokkos::min(d_costs_up[column], d_costs_right[column]));
      d_M[column] = left;
      m_cache[column] = left;
    }
  }

  syncthreads();

  //other rows
  for(int row = 1; row < h; row++){
    for(int i = 0; i < n_elem; i++){
      column = tid + i*memT.team_size();
      if(column < current_w){
        ix = row*w + column;

        //with left
        if(column > 0){
          left = m_cache[column - 1] + d_costs_left[ix];
        }
        else
          left = INT_MAX;
        //with up
        up = m_cache[column] + d_costs_up[ix];
        //with right
        if(column < current_w-1){
          right = m_cache[column + 1] + d_costs_right[ix];
        }
        else
          right = INT_MAX;

        left = Kokkos::min(left, Kokkos::min(up, right));
        d_M[ix] = left;
        m_cache_swap[column] = left;
      }
    }
    //swap read/write shared memory
    pointer_swap((void**)&m_cache, (void**)&m_cache_swap);
    syncthreads();
  }
}

//compute M one row at a time with multiple kernel calls for global synchronization
KOKKOS_INLINE_FUNCTION void compute_M_kernel_iterate0(
    Kokkos::TeamPolicy<ExecSpace>::member_type memT,
    const short *__restrict__ d_costs_left,
    const short *__restrict__ d_costs_up,
    const short *__restrict__ d_costs_right,
    int* __restrict__ d_M,
    int w, int current_w)
{
  int column = memT.league_rank() * memT.team_size() + memT.team_rank();
  if(column < current_w){
    d_M[column] = Kokkos::min(d_costs_left[column], Kokkos::min(d_costs_up[column], d_costs_right[column]));
  }
}

KOKKOS_INLINE_FUNCTION void compute_M_kernel_iterate1(
    Kokkos::TeamPolicy<ExecSpace>::member_type memT,
    const short *__restrict__ d_costs_left,
    const short *__restrict__ d_costs_up,
    const short *__restrict__ d_costs_right,
    int* __restrict__ d_M,
    int w, int current_w, int row)
{
  int column = memT.league_rank() * memT.team_size() + memT.team_rank();
  int ix = row*w + column;
  int prev_ix = ix - w;
  int left, up, right;

  if(column < current_w){
    //with left
    if(column > 0)
      left = d_M[prev_ix - 1] + d_costs_left[ix];
    else
      left = INT_MAX;
    //with up
    up = d_M[prev_ix] + d_costs_up[ix];
    //with right
    if(column < current_w-1)
      right = d_M[prev_ix + 1] + d_costs_right[ix];
    else
      right = INT_MAX;

    d_M[ix] = Kokkos::min(left, Kokkos::min(up, right));
  }
}

KOKKOS_INLINE_FUNCTION void min_reduce(
    Kokkos::TeamPolicy<ExecSpace>::member_type memT,
    int *__restrict__ val_cache,
    int *__restrict__ ix_cache,
    const int* __restrict__ d_values,
    int* __restrict__ d_indices,
    int size)
{
  int tid = memT.team_rank();
  int bid = memT.league_rank();
  int column = bid*REDUCE_BLOCKSIZE_X + tid;
  int grid_size = memT.team_size()*REDUCE_BLOCKSIZE_X;
  int min_v = INT_MAX;
  int min_i = 0;
  int new_i, new_v;

  for(int i = 0; i < REDUCE_ELEMENTS_PER_THREAD; i++){
    if(column < size){
      new_i = d_indices[column];
      new_v  = d_values[new_i];
      if(new_v < min_v){
        min_i = new_i;
        min_v = new_v;
      }
    }
    column = column + grid_size;
  }
  val_cache[tid] = min_v;
  ix_cache[tid] = min_i;

  syncthreads();

  for(int i = REDUCE_BLOCKSIZE_X/2; i > 0; i = i/2){
    if(tid < i){
      if(val_cache[tid + i] < val_cache[tid] || (val_cache[tid + i] == val_cache[tid] && ix_cache[tid + i] < ix_cache[tid])){
        val_cache[tid] = val_cache[tid + i];
        ix_cache[tid] = ix_cache[tid + i];
      }
    }
    syncthreads();
  }

  if(tid == 0){
    d_indices[bid] = ix_cache[0];
  }
}

KOKKOS_INLINE_FUNCTION void find_seam_kernel(
    const int *__restrict__ d_M,
    const int *__restrict__ d_indices,
    int *__restrict__ d_seam,
    int w, int h, int current_w)
{
  int base_row, mid;
  int min_index = d_indices[0];

  d_seam[h-1] = min_index;
  for(int row = h-2; row >= 0; row--){
    base_row = row*w;
    mid = min_index;
    if(mid != 0){
      if(d_M[base_row + mid - 1] < d_M[base_row + min_index])
        min_index = mid - 1;
    }
    if(mid != current_w){
      if(d_M[base_row + mid + 1] < d_M[base_row + min_index])
        min_index = mid + 1;
    }
    d_seam[row] = min_index;
  }
}

KOKKOS_INLINE_FUNCTION void remove_seam_kernel(
    size_t row, size_t column,
    const pixel *__restrict__ d_pixels,
          pixel *__restrict__ d_pixels_swap,
    const int *__restrict__ d_seam,
    int w, int h, int current_w)
{
  if(row < h && column < current_w-1){
    int seam_c = d_seam[row];
    int ix = row*w + column;
    d_pixels_swap[ix] = (column >= seam_c) ? d_pixels[ix + 1] : d_pixels[ix];
  }
}

KOKKOS_INLINE_FUNCTION void update_costs_kernel(
    size_t row, size_t column,
    const pixel *__restrict__ d_pixels,
    const short *__restrict__ d_costs_left,
    const short *__restrict__ d_costs_up,
    const short *__restrict__ d_costs_right,
          short *__restrict__ d_costs_swap_left,
          short *__restrict__ d_costs_swap_up,
          short *__restrict__ d_costs_swap_right,
    const int *__restrict__ d_seam,
    int w, int h, int current_w)
{

  if(row < h && column < current_w-1){
    int seam_c = d_seam[row];
    int ix = row*w + column;
    if(column >= seam_c-2 && column <= seam_c+1){
      //update costs near removed seam
      pixel pix1, pix2, pix3;
      int p_r, p_g, p_b;
      int rdiff, gdiff, bdiff;

      if(column == current_w-2)
        pix1 = BORDER_PIXEL;
      else
        pix1 = d_pixels[ix + 1];
      if(column == 0)
        pix2 = BORDER_PIXEL;
      else
        pix2 = d_pixels[ix - 1];
      if(row == 0)
        pix3 = BORDER_PIXEL;
      else
        pix3 = d_pixels[ix - w];

      //compute partials
      p_r = Kokkos::abs(pix1.r - pix2.r);
      p_g = Kokkos::abs(pix1.g - pix2.g);
      p_b = Kokkos::abs(pix1.b - pix2.b);

      //compute left cost
      rdiff = p_r + Kokkos::abs(pix3.r - pix2.r);
      gdiff = p_g + Kokkos::abs(pix3.g - pix2.g);
      bdiff = p_b + Kokkos::abs(pix3.b - pix2.b);
      d_costs_swap_left[ix] = rdiff + gdiff + bdiff;

      //compute up cost
      d_costs_swap_up[ix] = p_r + p_g + p_b;

      //compute right cost
      rdiff = p_r + Kokkos::abs(pix3.r - pix1.r);
      gdiff = p_g + Kokkos::abs(pix3.g - pix1.g);
      bdiff = p_b + Kokkos::abs(pix3.b - pix1.b);
      d_costs_swap_right[ix] = rdiff + gdiff + bdiff;
    }
    else if(column > seam_c+1){
      //shift costs to the left
      d_costs_swap_left[ix] = d_costs_left[ix + 1];
      d_costs_swap_up[ix] = d_costs_up[ix + 1];
      d_costs_swap_right[ix] = d_costs_right[ix + 1];
    }
    else{
      //copy remaining costs
      d_costs_swap_left[ix] = d_costs_left[ix];
      d_costs_swap_up[ix] = d_costs_up[ix];
      d_costs_swap_right[ix] = d_costs_right[ix];
    }
  }
}

KOKKOS_INLINE_FUNCTION void approx_setup_kernel(
    Kokkos::TeamPolicy<ExecSpace>::member_type memT,
    pixel *__restrict__ pix_cache,
    short *__restrict__ left_cache,
    short *__restrict__ up_cache,
    short *__restrict__ right_cache,
    const pixel *__restrict__ d_pixels,
    int *__restrict__ d_index_map,
    int *__restrict__ d_offset_map,
    int *__restrict__ d_M, int w, int h, int current_w){
  int blockIdx_y = memT.league_rank() / APPROX_SETUP_BLOCKSIZE_X;
  int blockIdx_x = memT.league_rank() % APPROX_SETUP_BLOCKSIZE_X;
  int threadIdx_y = memT.team_rank() / (current_w-1);
  int threadIdx_x = memT.team_rank() % (current_w-1);
  int row = blockIdx_y*(APPROX_SETUP_BLOCKSIZE_Y-1) + threadIdx_y -1 ;
  int column = blockIdx_x*(APPROX_SETUP_BLOCKSIZE_X-4) + threadIdx_x -2; //WE NEED MORE HORIZONTAL HALO...
  int ix = row*w + column;
  int cache_row = threadIdx_y;
  int cache_column = threadIdx_x;
  short active = 0;

  if(row >= 0 && row < h && column >= 0 && column < current_w){
    active = 1;
    pix_cache[cache_row * APPROX_SETUP_BLOCKSIZE_X + cache_column] = d_pixels[ix];
  }
  else{
    pix_cache[cache_row * APPROX_SETUP_BLOCKSIZE_X + cache_column] = BORDER_PIXEL;
  }

  //wait until each thread has initialized its portion of shared memory
  syncthreads();

  if(active && cache_row > 0){
    int rdiff, gdiff, bdiff;
    int p_r, p_g, p_b;
    pixel pix1, pix2, pix3;

    if(cache_column < APPROX_SETUP_BLOCKSIZE_X-1){
      pix1 = pix_cache[cache_row * APPROX_SETUP_BLOCKSIZE_X + cache_column+1];   //...OR ELSE WE CANNOT CALCULATE LEFT COST FOR THE LAST THREAD IN THE BLOCK (pix1 dependance)
    }

    if(cache_column > 0){
      pix2 = pix_cache[cache_row * APPROX_SETUP_BLOCKSIZE_X + cache_column-1];   //SAME THING WITH RIGHT COST FOR THE FIRST THREAD (pix2 dependance)
    }

    pix3 = pix_cache[(cache_row-1) * APPROX_SETUP_BLOCKSIZE_X + cache_column];

    //compute partials
    p_r = Kokkos::abs(pix1.r - pix2.r);
    p_g = Kokkos::abs(pix1.g - pix2.g);
    p_b = Kokkos::abs(pix1.b - pix2.b);

    //compute left cost
    rdiff = p_r + Kokkos::abs(pix3.r - pix2.r);
    gdiff = p_g + Kokkos::abs(pix3.g - pix2.g);
    bdiff = p_b + Kokkos::abs(pix3.b - pix2.b);
    left_cache[cache_row * APPROX_SETUP_BLOCKSIZE_X + cache_column] = rdiff + gdiff + bdiff;

    //compute up cost
    up_cache[cache_row * APPROX_SETUP_BLOCKSIZE_X + cache_column] = p_r + p_g + p_b;

    //compute right cost
    rdiff = p_r + Kokkos::abs(pix3.r - pix1.r);
    gdiff = p_g + Kokkos::abs(pix3.g - pix1.g);
    bdiff = p_b + Kokkos::abs(pix3.b - pix1.b);
    right_cache[cache_row * APPROX_SETUP_BLOCKSIZE_X + cache_column] = rdiff + gdiff + bdiff;
  }

  syncthreads();

  if(active && row < h-1 && cache_column > 1 && cache_column < APPROX_SETUP_BLOCKSIZE_X-2 &&
     cache_row != APPROX_SETUP_BLOCKSIZE_Y-1){
    int min_cost = INT_MAX;
    int map_ix;
    int cost;

    if(column > 0){
      min_cost = right_cache[(cache_row+1) * APPROX_SETUP_BLOCKSIZE_X + cache_column-1];
      map_ix = ix + w - 1;
    }

    cost = up_cache[(cache_row+1) * APPROX_SETUP_BLOCKSIZE_X + cache_column];
    if(cost < min_cost){
      min_cost = cost;
      map_ix = ix + w;
    }

    if(column < current_w-1){
      cost = left_cache[(cache_row+1) * APPROX_SETUP_BLOCKSIZE_X + cache_column+1];
      if(cost < min_cost){
        min_cost = cost;
        map_ix = ix + w + 1;
      }
    }

    d_index_map[ix] = map_ix;
    d_offset_map[ix] = map_ix;
    d_M[ix] = min_cost;
  }
}

KOKKOS_INLINE_FUNCTION void approx_seam_kernel(
    const int *__restrict__ d_index_map,
    const int *__restrict__ d_indices,
    int *__restrict__ d_seam,
    int w, int h)
{
  int ix;
  ix = d_indices[0]; //min index
  for(int i = 0; i < h; i++){
    d_seam[i] = ix - i*w;
    ix = d_index_map[ix];
  }
}
