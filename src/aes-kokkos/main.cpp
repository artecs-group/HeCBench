#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#include "SDKBitMap.h"
#include "aes.h"
#include "kernels.hpp"
#include "reference.hpp"
#include "utils.hpp"


int main(int argc, char * argv[])
{
  Kokkos::initialize(argc, argv);
  {
  if (argc != 4) {
    printf("Usage: %s <iterations> <0 or 1> <path to bitmap image file>\n", argv[0]);
    printf("0=encrypt, 1=decrypt\n");
    return 1;
  }

  const unsigned int keySizeBits = 128;
  const unsigned int rounds = 10;
  const unsigned int seed = 123;

  const int iterations = atoi(argv[1]);
  const bool decrypt = atoi(argv[2]);
  const char* filePath = argv[3];

  SDKBitMap image;
  image.load(filePath);
  const int width  = image.getWidth();
  const int height = image.getHeight();

  /* check condition for the bitmap to be initialized */
  if (width <= 0 || height <= 0) return 1;

  std::cout << "Image width and height: " 
            << width << " " << height << std::endl;

  uchar4S *pixels = image.getPixels();

  unsigned int sizeBytes = width*height*sizeof(unsigned char);
  unsigned char *input = (unsigned char*)malloc(sizeBytes); 

  /* initialize the input array, do NOTHING but assignment when decrypt*/
  if (decrypt)
    convertGrayToGray(pixels, input, height, width);
  else
    convertColorToGray(pixels, input, height, width);

  unsigned int keySize = keySizeBits/8; // 1 Byte = 8 bits

  unsigned int keySizeBytes = keySize*sizeof(unsigned char);

  unsigned char *key = (unsigned char*)malloc(keySizeBytes);

  fillRandom<unsigned char>(key, keySize, 1, 0, 255, seed); 

  // expand the key
  unsigned int explandedKeySize = (rounds+1)*keySize;
  unsigned char *expandedKey = (unsigned char*)malloc(explandedKeySize*sizeof(unsigned char));
  unsigned char *roundKey    = (unsigned char*)malloc(explandedKeySize*sizeof(unsigned char));

  keyExpansion(key, expandedKey, keySize, explandedKeySize);
  for(unsigned int i = 0; i < rounds+1; ++i)
  {
    createRoundKey(expandedKey + keySize*i, roundKey + keySize*i);
  }

  // save device result
  unsigned char* output = (unsigned char*)malloc(sizeBytes);

  uchar4S *inputBuffer = (uchar4S*)Kokkos::kokkos_malloc<MemSpace>(sizeBytes);
  Kokkos::Impl::DeepCopy<MemSpace, host_memory>(inputBuffer, input, sizeBytes);

  uchar4S *outputBuffer = (uchar4S*)Kokkos::kokkos_malloc<MemSpace>(sizeBytes);

  uchar4S *rKeyBuffer = (uchar4S*)Kokkos::kokkos_malloc<MemSpace>(explandedKeySize);
  Kokkos::Impl::DeepCopy<MemSpace, host_memory>(rKeyBuffer, roundKey, explandedKeySize);

  ucharS *sBoxBuffer = (ucharS*)Kokkos::kokkos_malloc<MemSpace>(256*sizeof(ucharS));
  Kokkos::Impl::DeepCopy<MemSpace, host_memory>(sBoxBuffer, sbox, 256*sizeof(ucharS));

  ucharS *rsBoxBuffer = (ucharS*)Kokkos::kokkos_malloc<MemSpace>(256*sizeof(ucharS));
  Kokkos::Impl::DeepCopy<MemSpace, host_memory>(rsBoxBuffer, rsbox, 256*sizeof(ucharS));

  std::cout << "Executing kernel for " << iterations 
            << " iterations" << std::endl;
  std::cout << "-------------------------------------------" << std::endl;

  size_t gws (height/4 * width/4);
  size_t lws (4);

  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();

  for(int i = 0; i < iterations; i++)
  {
    if (decrypt)
      Kokkos::parallel_for("dec", 
      Kokkos::TeamPolicy<ExecSpace>(gws, lws)
      .set_scratch_size(0, Kokkos::PerTeam(keySize/4*sizeof(uchar4S)*2)), 
      KOKKOS_LAMBDA(Kokkos::TeamPolicy<ExecSpace>::member_type memT){
        uchar4S* block0 = (uchar4S*) memT.team_shmem().get_shmem(keySize/4*sizeof(uchar4S));
        uchar4S* block1 = (uchar4S*) memT.team_shmem().get_shmem(keySize/4*sizeof(uchar4S));
        AESDecrypt(outputBuffer,
                    inputBuffer,
                    rKeyBuffer,
                    rsBoxBuffer,
                    block0,
                    block1,
                    width, rounds, memT);
      });

    else
      Kokkos::parallel_for("enc", 
      Kokkos::TeamPolicy<ExecSpace>(gws, lws)
      .set_scratch_size(0, Kokkos::PerTeam(keySize/4*sizeof(uchar4S)*2)), 
      KOKKOS_LAMBDA(Kokkos::TeamPolicy<ExecSpace>::member_type memT){
        uchar4S* block0 = (uchar4S*) memT.team_shmem().get_shmem(keySize/4*sizeof(uchar4S));
        uchar4S* block1 = (uchar4S*) memT.team_shmem().get_shmem(keySize/4*sizeof(uchar4S));
        AESEncrypt(outputBuffer,
                    inputBuffer,
                    rKeyBuffer,
                    sBoxBuffer,
                    block0,
                    block1,
                    width, rounds, memT);
      });
  }

  Kokkos::fence();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time " << (time * 1e-9f) / iterations << " (s)\n";

  Kokkos::Impl::DeepCopy<host_memory, MemSpace>(output, outputBuffer, sizeBytes);
  Kokkos::fence();

  // Verify
  unsigned char *verificationOutput = (unsigned char *) malloc(sizeBytes);

  reference(verificationOutput, input, roundKey, explandedKeySize, 
      width, height, decrypt, rounds, keySize);

  /* compare the results and see if they match */
  if(memcmp(output, verificationOutput, sizeBytes) == 0)
    std::cout<<"Pass\n";
  else
    std::cout<<"Fail\n";

  Kokkos::kokkos_free(inputBuffer);
  Kokkos::kokkos_free(outputBuffer);
  Kokkos::kokkos_free(rKeyBuffer);
  Kokkos::kokkos_free(sBoxBuffer);
  Kokkos::kokkos_free(rsBoxBuffer);

  /* release program resources (input memory etc.) */
  if(input) free(input);

  if(key) free(key);

  if(expandedKey) free(expandedKey);

  if(roundKey) free(roundKey);

  if(output) free(output);

  if(verificationOutput) free(verificationOutput);
  }
  Kokkos::finalize();
  return 0;
}
