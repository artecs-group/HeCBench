#include "morphology.h"

void display(unsigned char *img, const int height, const int width)
{
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++)
      printf("%d ", img[i*width+j]);
    printf("\n");
  }
  printf("\n");
}

int main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
  if (argc != 6) {
    printf("Usage: %s <kernel width> <kernel height> ", argv[0]);
    printf("<image width> <image height> <repeat>\n");
    return 1;
  }

  int hsize = atoi(argv[1]);  // kernel width
  int vsize = atoi(argv[2]);  // kernel height
  int width = atoi(argv[3]);  // image width
  int height = atoi(argv[4]); // image height
  int repeat = atoi(argv[5]);

  unsigned char* initU = new unsigned char[width * height]();
  unsigned int memSize = width * height * sizeof(unsigned char);
  unsigned char* srcImg = (unsigned char*) malloc (memSize);

  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++)
      srcImg[i*width+j] = (i == (height/2 - 1) &&
                           j == (width/2 - 1)) ? WHITE : BLACK;

  unsigned char *img_d = (unsigned char*)Kokkos::kokkos_malloc<MemSpace>(memSize);
  Kokkos::Impl::DeepCopy<MemSpace, host_memory>(img_d, srcImg, memSize);

  unsigned char *tmp_d = (unsigned char*)Kokkos::kokkos_malloc<MemSpace>(memSize);

  double dilate_time = 0.0, erode_time = 0.0;

  for (int n = 0; n < repeat; n++) {
    dilate_time += dilate(img_d, tmp_d, initU, width, height, hsize, vsize);
    erode_time += erode(img_d, tmp_d, initU, width, height, hsize, vsize);
  }

  printf("Average kernel execution time (dilate): %f (s)\n", (dilate_time * 1e-9f) / repeat);
  printf("Average kernel execution time (erode): %f (s)\n", (erode_time * 1e-9f) / repeat);

  Kokkos::Impl::DeepCopy<host_memory, MemSpace>(srcImg, img_d, memSize);
  Kokkos::fence();

  int s = 0;
  for (unsigned int i = 0; i < memSize; i++) s += srcImg[i];
  printf("%s\n", s == WHITE ? "PASS" : "FAIL");

  Kokkos::kokkos_free(img_d);
  Kokkos::kokkos_free(tmp_d);
  free(srcImg);
  free(initU);
  }
  Kokkos::finalize();
  return 0;
}