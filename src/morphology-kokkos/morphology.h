#ifndef MORPHOLOGY_H
#define MORPHOLOGY_H

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <Kokkos_Core.hpp>

#define BLACK 0
#define WHITE 255

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

/*!
 * \file morphology.h
 *
 * We use the van Herk/Gil-Werman (vHGW) algorithm, [van Herk,
 * Patt. Recog. Let. 13, pp. 517-521, 1992; Gil and Werman,
 * IEEE Trans PAMI 15(5), pp. 504-507, 1993.]
 *
 * Please refer to the leptonica documents for more details:
 * http://www.leptonica.com/binary-morphology.html
 *
 */

inline int roundUp(const int x, const int y)
{
    return (x + y - 1) / y; 
}


/*!
 * \brief erode()
 *
 * \param[in/out]   img_d: device memory pointer to source image
 * \param[in]       width: image width
 * \param[in]       height: image height
 * \param[in]       hsize: horizontal size of Sel; must be odd; origin implicitly in center
 * \param[in]       vsize: ditto
 */
extern "C"
double erode(
        unsigned char *img_d,
        unsigned char *tmp_d,
        unsigned char* initU,
        const int width,
        const int height,
        const int hsize,
        const int vsize);


/*!
 * \brief dilate()
 *
 * \param[in/out]   img_d: device memory pointer to source image
 * \param[in]       width: image width
 * \param[in]       height: image height
 * \param[in]       hsize: horizontal size of Sel; must be odd; origin implicitly in center
 * \param[in]       vsize: ditto
 */
extern "C"
double dilate(
        unsigned char *img_d,
        unsigned char *tmp_d,
        unsigned char* initU,
        const int width,
        const int height,
        const int hsize,
        const int vsize);

#endif /* MORPHOLOGY_H */
