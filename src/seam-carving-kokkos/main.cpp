#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdint.h>
#include <chrono>
#include <utility>  // std::swap
#include "utils.h"
#include "kernels.h"
#include "kernels_wrapper.h"
#include <Kokkos_Core.hpp>

//#define STBI_ONLY_BMP
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

template <typename T>
void swap(Kokkos::View<T*, Layout, MemSpace> a, Kokkos::View<T*, Layout, MemSpace> b, Kokkos::View<T*, Layout, MemSpace> swap){
  Kokkos::deep_copy(swap, a);
  Kokkos::deep_copy(a, b);
  Kokkos::deep_copy(b, swap);
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
  if(argc < 3){
    printf("Usage: %s <file> <number of seams to remove> [options]\n"
        "valid options:\n-u\tupdate costs instead of recomputing them\n"
        "-a\tapproximate computation\n", argv[0]);
    return 1;
  }

  char *check;
  long seams_to_remove = strtol(argv[2], &check, 10);  //10 specifies base-10
  if (check == argv[2]){   //if no characters were converted pointers are equal
    printf("ERROR: can't convert string to number, exiting.\n");
    return 1;
  }

  int w, h, ncomp;
  unsigned char* imgv = stbi_load(argv[1], &w, &h, &ncomp, 0);
  if(imgv == NULL){
    printf("ERROR: can't load image \"%s\" (maybe the file does not exist?), exiting.\n", argv[1]);
    return 1;
  }

  if(ncomp != 3){
    printf("ERROR: image does not have 3 components (RGB), exiting.\n");
    return 1;
  }

  if(seams_to_remove < 0 || seams_to_remove >= w){
    printf("ERROR: number of seams to remove is invalid, exiting.\n");
    return 1;
  }

  seam_carver_mode mode = SEAM_CARVER_STANDARD_MODE;

  if(argc >= 4){
    if(strcmp(argv[3],"-u") == 0){
      mode = SEAM_CARVER_UPDATE_MODE;
      printf("update mode selected.\n");
    }
    else if(strcmp(argv[3],"-a") == 0){
      mode = SEAM_CARVER_APPROX_MODE;
      printf("approximation mode selected.\n");
    }
    else{
      printf("an invalid option was specified and will be ignored. Valid options are: -u, -a.\n");
    }
  }

  printf("Image loaded. Resizing...\n");

  int current_w = w;
  uchar4 *h_pixels = build_pixels(imgv, w, h);
  const int img_size = w * h;
  const int img_bytes = img_size * sizeof(uchar4);
  const int w_bytes = w * sizeof(int);

  int* indices = (int*)malloc(w_bytes);
  for(int i = 0; i < w; i++) indices[i] = i;

  Kokkos::View<short*, Layout, MemSpace> d_costs_left;
  Kokkos::View<short*, Layout, MemSpace> d_costs_up;
  Kokkos::View<short*, Layout, MemSpace> d_costs_right;

  if(mode != SEAM_CARVER_APPROX_MODE) {
    d_costs_left = Kokkos::View<short*, Layout, MemSpace>("d_costs_left", img_size);
    d_costs_up = Kokkos::View<short*, Layout, MemSpace>("d_costs_up", img_size);
    d_costs_right = Kokkos::View<short*, Layout, MemSpace>("d_costs_right", img_size);
  }

  Kokkos::View<short*, Layout, MemSpace> d_costs_swap_left;
  Kokkos::View<short*, Layout, MemSpace> d_costs_swap_up;
  Kokkos::View<short*, Layout, MemSpace> d_costs_swap_right;
  if(mode == SEAM_CARVER_UPDATE_MODE) {
    d_costs_swap_left = Kokkos::View<short*, Layout, MemSpace>("d_costs_swap_left", img_size);
    d_costs_swap_up = Kokkos::View<short*, Layout, MemSpace>("d_costs_swap_up", img_size);
    d_costs_swap_right = Kokkos::View<short*, Layout, MemSpace>("d_costs_swap_right", img_size);
  }

  //sum map in approx mode
  Kokkos::View<int*, Layout, MemSpace> d_M = Kokkos::View<int*, Layout, MemSpace>("d_M", img_size);
  Kokkos::View<int*, Layout, MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> reduce_row;

  // rows to consider for reduce
  if(mode == SEAM_CARVER_APPROX_MODE)
    reduce_row = Kokkos::View<int*, Layout, MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(d_M.data(), h);
    //first row
  else
    reduce_row = Kokkos::View<int*, Layout, MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(d_M.data() + w*(h-1), h);
    //last row
  
  Kokkos::View<int*, Layout, MemSpace> d_index_map;
  Kokkos::View<int*, Layout, MemSpace> d_offset_map;

  if(mode == SEAM_CARVER_APPROX_MODE){
    d_index_map = Kokkos::View<int*, Layout, MemSpace>("d_index_map", img_size);
    d_offset_map = Kokkos::View<int*, Layout, MemSpace>("d_offset_map", img_size);
  }

  Kokkos::View<int*, Layout, MemSpace> d_indices = Kokkos::View<int*, Layout, MemSpace>("d_indices", w);
  Kokkos::View<int*, Layout, MemSpace> d_indices_ref = Kokkos::View<int*, Layout, MemSpace>("d_indices_ref", w);
  Kokkos::View<const int*, Layout, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> vindices(indices, w);
  Kokkos::deep_copy(d_indices_ref, vindices);

  Kokkos::View<int*, Layout, MemSpace> d_seam = Kokkos::View<int*, Layout, MemSpace>("d_seam", h);

  Kokkos::View<uchar4*, Layout, MemSpace> d_pixels = Kokkos::View<uchar4*, Layout, MemSpace>("d_pixels", img_size);
  Kokkos::View<const uchar4*, Layout, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> vh_pixels(h_pixels, img_size);
  Kokkos::deep_copy(d_pixels, vh_pixels);

  Kokkos::View<uchar4*, Layout, MemSpace> d_pixels_swap = Kokkos::View<uchar4*, Layout, MemSpace>("d_pixels_swap", img_size);
  Kokkos::View<uchar4*, Layout, MemSpace> d_swap = Kokkos::View<uchar4*, Layout, MemSpace>("d_swap", img_size);

  if(mode == SEAM_CARVER_UPDATE_MODE)
    compute_costs(current_w, w, h, d_pixels.data(), d_costs_left.data(), d_costs_up.data(), d_costs_right.data());

  int num_iterations = 0;

  auto start = std::chrono::steady_clock::now();

  while(num_iterations < (int)seams_to_remove){

    if(mode == SEAM_CARVER_STANDARD_MODE)
      compute_costs(current_w, w, h, d_pixels.data(), d_costs_left.data(), d_costs_up.data(), d_costs_right.data());

    if(mode != SEAM_CARVER_APPROX_MODE){
      compute_M(current_w, w, h, d_M.data(), d_costs_left.data(), d_costs_up.data(), d_costs_right.data());
      find_min_index(current_w, d_indices_ref.data(), d_indices.data(), reduce_row.data());
      find_seam(current_w, w, h, d_M.data(), d_indices.data(), d_seam.data());
    }
    else{
      approx_setup(current_w, w, h, d_pixels.data(), d_index_map.data(), d_offset_map.data(), d_M.data());
      approx_M(current_w, w, h, d_offset_map.data(),  d_M.data());
      find_min_index(current_w, d_indices_ref.data(), d_indices.data(), reduce_row.data());
      approx_seam(w, h, d_index_map.data(), d_indices.data(), d_seam.data());
    }

    remove_seam(current_w, w, h, d_M.data(), d_pixels.data(), d_pixels_swap.data(), d_seam.data());
    std::swap(d_pixels.data(), d_pixels_swap.data());

    if(mode == SEAM_CARVER_UPDATE_MODE){
      update_costs(current_w, w, h, d_M.data(), d_pixels.data(),
                   d_costs_left.data(), d_costs_up.data(), d_costs_right.data(),
                   d_costs_swap_left.data(), d_costs_swap_up.data(), d_costs_swap_right.data(), d_seam.data() );
      std::swap(d_costs_left.data(), d_costs_swap_left.data());
      std::swap(d_costs_up.data(), d_costs_swap_up.data());
      std::swap(d_costs_right.data(), d_costs_swap_right.data());
    }

    current_w--;
    num_iterations++;
  }

  Kokkos::fence();
  auto end = std::chrono::steady_clock::now();
  float time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Execution time of seam carver kernels: %f (ms)\n", time * 1e-6f);

  Kokkos::deep_copy(vh_pixels, d_pixels);
  Kokkos::fence();
  unsigned char* output = flatten_pixels(h_pixels, w, h, current_w);
  printf("Image resized\n");

  printf("Saving in resized.bmp...\n");
  int success = stbi_write_bmp("resized.bmp", current_w, h, 3, output);
  printf("%s\n", success ? "Success" : "Failed");
  free(h_pixels);
  free(output);
  free(indices);
  }
  Kokkos::finalize();
  return 0;
}
