#include <Kokkos_Core.hpp>
#include "SDKBitMap.h"

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

// called by host and device
KOKKOS_INLINE_FUNCTION unsigned char galoisMultiplication(unsigned char a, unsigned char b)
{
    unsigned char p = 0; 
    for(unsigned int i=0; i < 8; ++i)
    {
        if((b&1) == 1)
        {
            p^=a;
        }
        unsigned char hiBitSet = (a & 0x80);
        a <<= 1;
        if(hiBitSet == 0x80)
        {
            a ^= 0x1b;
        }
        b >>= 1;
    }
    return p;
}

KOKKOS_INLINE_FUNCTION uchar4S sboxRead(const ucharS * SBox, uchar4S block)
{
    return {SBox[block.x], SBox[block.y], SBox[block.z], SBox[block.w]};
}

KOKKOS_INLINE_FUNCTION uchar4S mixColumns(const uchar4S * block, const uchar4S * galiosCoeff, unsigned int j)
{
    unsigned int bw = 4;

    ucharS x, y, z, w;

    x = galoisMultiplication(block[0].x, galiosCoeff[(bw-j)%bw].x);
    y = galoisMultiplication(block[0].y, galiosCoeff[(bw-j)%bw].x);
    z = galoisMultiplication(block[0].z, galiosCoeff[(bw-j)%bw].x);
    w = galoisMultiplication(block[0].w, galiosCoeff[(bw-j)%bw].x);
   
    for(unsigned int k=1; k< 4; ++k)
    {
        x ^= galoisMultiplication(block[k].x, galiosCoeff[(k+bw-j)%bw].x);
        y ^= galoisMultiplication(block[k].y, galiosCoeff[(k+bw-j)%bw].x);
        z ^= galoisMultiplication(block[k].z, galiosCoeff[(k+bw-j)%bw].x);
        w ^= galoisMultiplication(block[k].w, galiosCoeff[(k+bw-j)%bw].x);
    }
    
    return {x, y, z, w};
}

KOKKOS_INLINE_FUNCTION uchar4S shiftRows(uchar4S row, unsigned int j)
{
    uchar4S r = row;
    for(uint i=0; i < j; ++i)  
    {
        //r.xyzw() = r.yzwx();
        ucharS x = r.x;
        ucharS y = r.y;
        ucharS z = r.z;
        ucharS w = r.w;
        r = {y,z,w,x};
    }
    return r;
}

KOKKOS_INLINE_FUNCTION void AESEncrypt(      uchar4S  *__restrict output  ,
                const uchar4S  *__restrict input   ,
                const uchar4S  *__restrict roundKey,
                const ucharS   *__restrict SBox    ,
                      uchar4S  *__restrict block0  ,  // lmem
                      uchar4S  *__restrict block1  ,  // lmem
                const uint     width , 
                const uint     rounds,
                Kokkos::TeamPolicy<ExecSpace>::member_type memT)
                                
{
    unsigned int blockIdx = memT.league_rank() % (width/4);
    unsigned int blockIdy = memT.league_rank() / (width/4);
 
    //unsigned int localIdx = item.get_local_id(1);
    unsigned int localIdy = memT.team_rank();
    
    unsigned int globalIndex = (((blockIdy * width/4) + blockIdx) * 4 )+ (localIdy);
    unsigned int localIndex  = localIdy;

    uchar4S galiosCoeff[4];
    galiosCoeff[0] = {2, 0, 0, 0};
    galiosCoeff[1] = {3, 0, 0, 0};
    galiosCoeff[2] = {1, 0, 0, 0};
    galiosCoeff[3] = {1, 0, 0, 0};

    block0[localIndex]  = input[globalIndex];
    
    block0[localIndex] = block0[localIndex] ^ roundKey[localIndex];

    for(unsigned int r=1; r < rounds; ++r)
    {
        block0[localIndex] = sboxRead(SBox, block0[localIndex]);

        block0[localIndex] = shiftRows(block0[localIndex], localIndex); 
       
        memT.team_barrier();
        block1[localIndex]  = mixColumns(block0, galiosCoeff, localIndex); 
        
        memT.team_barrier();
        block0[localIndex] = block1[localIndex]^roundKey[r*4 + localIndex];
    }  
    block0[localIndex] = sboxRead(SBox, block0[localIndex]);
  
    block0[localIndex] = shiftRows(block0[localIndex], localIndex); 

    output[globalIndex] =  block0[localIndex]^roundKey[(rounds)*4 + localIndex];
}

KOKKOS_INLINE_FUNCTION uchar4S shiftRowsInv(uchar4S row, unsigned int j)
{
    uchar4S r = row;
    for(uint i=0; i < j; ++i)  
    {
        // r = r.wxyz();
        ucharS x = r.x;
        ucharS y = r.y;
        ucharS z = r.z;
        ucharS w = r.w;
        r = {w,x,y,z};
    }
    return r;
}

KOKKOS_INLINE_FUNCTION void AESDecrypt(       uchar4S  *__restrict output    ,
                const  uchar4S  *__restrict input     ,
                const  uchar4S  *__restrict roundKey  ,
                const  ucharS   *__restrict SBox      ,
                       uchar4S  *__restrict block0    ,
                       uchar4S  *__restrict block1    ,
                const  uint    width , 
                const  uint    rounds,
                Kokkos::TeamPolicy<ExecSpace>::member_type memT)
                                
{
    unsigned int blockIdx = memT.league_rank() % (width/4);
    unsigned int blockIdy = memT.league_rank() / (width/4);
 
    //unsigned int localIdx = item.get_local_id(1);
    unsigned int localIdy = memT.team_rank();
    
    unsigned int globalIndex = (((blockIdy * width/4) + blockIdx) * 4 )+ (localIdy);
    unsigned int localIndex  = localIdy;

    uchar4S galiosCoeff[4];
    galiosCoeff[0] = {14, 0, 0, 0};
    galiosCoeff[1] = {11, 0, 0, 0};
    galiosCoeff[2] = {13, 0, 0, 0};
    galiosCoeff[3] = { 9, 0, 0, 0};

    block0[localIndex]  = input[globalIndex];
    
    block0[localIndex] = block0[localIndex] ^ roundKey[4*rounds + localIndex];

    for(unsigned int r=rounds -1 ; r > 0; --r)
    {
        block0[localIndex] = shiftRowsInv(block0[localIndex], localIndex); 
    
        block0[localIndex] = sboxRead(SBox, block0[localIndex]);
        
        memT.team_barrier();
        block1[localIndex] = block0[localIndex]^roundKey[r*4 + localIndex];

        memT.team_barrier();
        block0[localIndex]  = mixColumns(block1, galiosCoeff, localIndex); 
    }  

    block0[localIndex] = shiftRowsInv(block0[localIndex], localIndex); 

    block0[localIndex] = sboxRead(SBox, block0[localIndex]);

    output[globalIndex] =  block0[localIndex]^roundKey[localIndex];
}
