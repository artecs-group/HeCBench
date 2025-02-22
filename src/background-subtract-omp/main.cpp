#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <omp.h>

#define BLOCK_SIZE 256

void findMovingPixels(
  const size_t imgSize,
  const unsigned char *__restrict Img,
  const unsigned char *__restrict Img1,
  const unsigned char *__restrict Img2,
  const unsigned char *__restrict Tn,
        unsigned char *__restrict Mp) // moving pixel map
{
  #pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE)
  for (size_t i = 0; i < imgSize; i++) {
    if ( abs(Img[i] - Img1[i]) > Tn[i] || abs(Img[i] - Img2[i]) > Tn[i] )
      Mp[i] = 255;
    else 
      Mp[i] = 0;
  }
}

// alpha = 0.92 
void updateBackground(
  const size_t imgSize,
  const unsigned char *__restrict Img,
  const unsigned char *__restrict Mp,
        unsigned char *__restrict Bn)
{
  #pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE)
  for (size_t i = 0; i < imgSize; i++) {
    if ( Mp[i] == 0 ) Bn[i] = 0.92f * Bn[i] + 0.08f * Img[i];
  }
}

// alpha = 0.92, c = 3
void updateThreshold(
  const size_t imgSize,
  const unsigned char *__restrict Img,
  const unsigned char *__restrict Mp,
  const unsigned char *__restrict Bn,
        unsigned char *__restrict Tn)
{
  #pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE)
  for (size_t i = 0; i < imgSize; i++) {
    if (Mp[i] == 0) {
      float th = 0.92f * Tn[i] + 0.24f * (Img[i] - Bn[i]);
      Tn[i] = fmaxf(th, 20.f);
    }
  }
}

//
// merge three kernels into a single kernel
//
void merge(
  const size_t imgSize,
  const unsigned char *__restrict Img,
  const unsigned char *__restrict Img1,
  const unsigned char *__restrict Img2,
        unsigned char *__restrict Tn,
        unsigned char *__restrict Bn)
{
  #pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE)
  for (size_t i = 0; i < imgSize; i++) {
    if ( abs(Img[i] - Img1[i]) <= Tn[i] && abs(Img[i] - Img2[i]) <= Tn[i] ) {
      // update background
      Bn[i] = 0.92f * Bn[i] + 0.08f * Img[i];

      // update threshold
      float th = 0.92f * Tn[i] + 0.24f * (Img[i] - Bn[i]);
      Tn[i] = fmaxf(th, 20.f);
    }
  }
}

int main(int argc, char* argv[]) {
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
  unsigned char *Img1 = (unsigned char*) malloc (imgSize_bytes);
  unsigned char *Img2 = (unsigned char*) malloc (imgSize_bytes);
  unsigned char *Bn = (unsigned char*) malloc (imgSize_bytes);
  unsigned char *Mp = (unsigned char*) malloc (imgSize_bytes);
  unsigned char *Tn = (unsigned char*) malloc (imgSize_bytes);

  std::mt19937 generator( 123 );
  std::uniform_int_distribution<int> distribute( 0, 255 );

  for (int j = 0; j < imgSize; j++) {
    Bn[j] = distribute(generator);
    Tn[j] = 128;
  }

  long time = 0;

  #pragma omp target data map (to: Bn[0:imgSize]) \
                          map (tofrom: Tn[0:imgSize]) \
                          map (alloc: Mp[0:imgSize], \
                                      Img[0:imgSize], \
                                      Img1[0:imgSize], \
                                      Img2[0:imgSize])
  {
    for (int i = 0; i < repeat; i++) {

      for (int j = 0; j < imgSize; j++) {
        Img[j] = distribute(generator);
      }

      #pragma omp target update to (Img[0:imgSize])

    // Time t   : Image   | Image1   | Image2
    // Time t+1 : Image2  | Image    | Image1
    // Time t+2 : Image1  | Image2   | Image
      unsigned char *t = Img2;
      Img2 = Img1;
      Img1 = Img;
      Img = t;

      if (i >= 2) {
        if (merged) {
          auto start = std::chrono::steady_clock::now();
          merge ( imgSize, Img, Img1, Img2, Tn, Bn );
          auto end = std::chrono::steady_clock::now();
          time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        }
        else {
          auto start = std::chrono::steady_clock::now();
          findMovingPixels ( imgSize, Img, Img1, Img2, Tn, Mp );
          updateBackground ( imgSize, Img, Mp, Bn );
          updateThreshold ( imgSize, Img, Mp, Bn, Tn );
          auto end = std::chrono::steady_clock::now();
          time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        }
      }
    }

    float kernel_time = (repeat <= 2) ? 0 : (time * 1e-9f) / (repeat - 2);
    printf("Average kernel execution time: %f (s)\n", kernel_time);
    printf("Total kernel execution time: %f (s)\n", time * 1e-9f);
  }
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
  free(Img1);
  free(Img2);
  free(Tn);
  free(Bn);
  free(Mp);

  return 0;
}
