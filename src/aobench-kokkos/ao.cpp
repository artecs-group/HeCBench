#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <Kokkos_Core.hpp>

#define WIDTH        256
#define HEIGHT       256
#define NSUBSAMPLES  2
#define NAO_SAMPLES  8
#define BLOCK_SIZE   16

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

typedef Kokkos::HostSpace::memory_space host_memory;
typedef Kokkos::LayoutRight Layout;

typedef struct _vec
{
  float x;
  float y;
  float z;
} Vec;


typedef struct _Isect
{
  float t;
  Vec    p;
  Vec    n;
  int    hit; 
} Isect;

typedef struct _Sphere
{
  Vec    center;
  float radius;

} Sphere;

typedef struct _Plane
{
  Vec    p;
  Vec    n;

} Plane;

typedef struct _Ray
{
  Vec    org;
  Vec    dir;
} Ray;


static KOKKOS_INLINE_FUNCTION float vdot(Vec v0, Vec v1)
{
  return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}

static KOKKOS_INLINE_FUNCTION void vcross(Vec *c, Vec v0, Vec v1)
{

  c->x = v0.y * v1.z - v0.z * v1.y;
  c->y = v0.z * v1.x - v0.x * v1.z;
  c->z = v0.x * v1.y - v0.y * v1.x;
}

static KOKKOS_INLINE_FUNCTION void vnormalize(Vec *c)
{
  float length = Kokkos::sqrt(vdot((*c), (*c)));

  if (Kokkos::fabs(length) > 1.0e-17f) {
    c->x /= length;
    c->y /= length;
    c->z /= length;
  }
}

KOKKOS_INLINE_FUNCTION void ray_sphere_intersect(Isect *isect, const Ray *ray, const Sphere *sphere)
{
  Vec rs;

  rs.x = ray->org.x - sphere->center.x;
  rs.y = ray->org.y - sphere->center.y;
  rs.z = ray->org.z - sphere->center.z;

  float B = vdot(rs, ray->dir);
  float C = vdot(rs, rs) - sphere->radius * sphere->radius;
  float D = B * B - C;

  if (D > 0.0) {
    float t = -B - Kokkos::sqrt(D);

    if ((t > 0.0) && (t < isect->t)) {
      isect->t = t;
      isect->hit = 1;

      isect->p.x = ray->org.x + ray->dir.x * t;
      isect->p.y = ray->org.y + ray->dir.y * t;
      isect->p.z = ray->org.z + ray->dir.z * t;

      isect->n.x = isect->p.x - sphere->center.x;
      isect->n.y = isect->p.y - sphere->center.y;
      isect->n.z = isect->p.z - sphere->center.z;

      vnormalize(&(isect->n));
    }
  }
}

  
KOKKOS_INLINE_FUNCTION void ray_plane_intersect(Isect *isect, const Ray *ray, const Plane *plane)
{
  float d = -vdot(plane->p, plane->n);
  float v = vdot(ray->dir, plane->n);

  if (Kokkos::fabs(v) < 1.0e-17f) return;

  float t = -(vdot(ray->org, plane->n) + d) / v;

  if ((t > 0.f) && (t < isect->t)) {
    isect->t = t;
    isect->hit = 1;

    isect->p.x = ray->org.x + ray->dir.x * t;
    isect->p.y = ray->org.y + ray->dir.y * t;
    isect->p.z = ray->org.z + ray->dir.z * t;

    isect->n = plane->n;
  }
}

KOKKOS_INLINE_FUNCTION void orthoBasis(Vec *basis, Vec n)
{
  basis[2] = n;
  basis[1].x = 0.f; basis[1].y = 0.f; basis[1].z = 0.f;

  if ((n.x < 0.6f) && (n.x > -0.6f)) {
    basis[1].x = 1.0f;
  } else if ((n.y < 0.6f) && (n.y > -0.6f)) {
    basis[1].y = 1.0f;
  } else if ((n.z < 0.6f) && (n.z > -0.6f)) {
    basis[1].z = 1.0f;
  } else {
    basis[1].x = 1.0f;
  }

  vcross(&basis[0], basis[1], basis[2]);
  vnormalize(&basis[0]);

  vcross(&basis[1], basis[2], basis[0]);
  vnormalize(&basis[1]);
}

class RNG {
  public:
    unsigned int x;
    const int fmask = (1 << 23) - 1;   
      KOKKOS_INLINE_FUNCTION RNG(const unsigned int seed) { x = seed; }   
      KOKKOS_INLINE_FUNCTION int next() {     
        x ^= x >> 6;
        x ^= x << 17;     
        x ^= x >> 9;
        return int(x);
      }
      KOKKOS_INLINE_FUNCTION float operator()(void) {
        union {
          float f;
          int i;
        } u;
        u.i = (next() & fmask) | 0x3f800000;
        return u.f - 1.f;
      }
};


KOKKOS_INLINE_FUNCTION void ambient_occlusion(Vec *col, const Isect *isect, 
		       Sphere* spheres, const Plane *plane, RNG &rng)
{
  int    i, j;
  int    ntheta = NAO_SAMPLES;
  int    nphi   = NAO_SAMPLES;
  float eps = 0.0001f;

  Vec p;

  p.x = isect->p.x + eps * isect->n.x;
  p.y = isect->p.y + eps * isect->n.y;
  p.z = isect->p.z + eps * isect->n.z;

  Vec basis[3];
  orthoBasis(basis, isect->n);


  float occlusion = 0.f;

  for (j = 0; j < ntheta; j++) {
    for (i = 0; i < nphi; i++) {
      float theta = Kokkos::sqrt(rng());
      float phi = 2.0f * (float)M_PI * rng();
      float x = Kokkos::cos(phi) * theta;
      float y = Kokkos::sin(phi) * theta;
      float z = Kokkos::sqrt(1.0f - theta * theta);

      // local -> global
      float rx = x * basis[0].x + y * basis[1].x + z * basis[2].x;
      float ry = x * basis[0].y + y * basis[1].y + z * basis[2].y;
      float rz = x * basis[0].z + y * basis[1].z + z * basis[2].z;

      Ray ray;

      ray.org = p;
      ray.dir.x = rx;
      ray.dir.y = ry;
      ray.dir.z = rz;

      Isect occIsect;
      occIsect.t   = 1.0e+17f;
      occIsect.hit = 0;

      ray_sphere_intersect(&occIsect, &ray, &spheres[0]);
      ray_sphere_intersect(&occIsect, &ray, &spheres[1]); 
      ray_sphere_intersect(&occIsect, &ray, &spheres[2]); 
      ray_plane_intersect (&occIsect, &ray, plane); 

      if (occIsect.hit) occlusion += 1.f;

    }
  }

  occlusion = (ntheta * nphi - occlusion) / (float)(ntheta * nphi);

  col->x = occlusion;
  col->y = occlusion;
  col->z = occlusion;
}

  
KOKKOS_INLINE_FUNCTION unsigned char my_clamp(float f)
{
  int i = (int)(f * 255.5f);

  if (i < 0) i = 0;
  if (i > 255) i = 255;

  return (unsigned char)i;
}


void init_scene(Sphere* spheres, Plane &plane)
{
  spheres[0].center.x = -2.0f;
  spheres[0].center.y =  0.0f;
  spheres[0].center.z = -3.5f;
  spheres[0].radius = 0.5f;

  spheres[1].center.x = -0.5f;
  spheres[1].center.y =  0.0f;
  spheres[1].center.z = -3.0f;
  spheres[1].radius = 0.5f;

  spheres[2].center.x =  1.0f;
  spheres[2].center.y =  0.0f;
  spheres[2].center.z = -2.2f;
  spheres[2].radius = 0.5f;

  plane.p.x = 0.0f;
  plane.p.y = -0.5f;
  plane.p.z = 0.0f;

  plane.n.x = 0.0f;
  plane.n.y = 1.0f;
  plane.n.z = 0.0f;

}

void saveppm(const char *fname, int w, int h, unsigned char *img)
{
  FILE *fp;

  fp = fopen(fname, "wb");
  assert(fp);

  fprintf(fp, "P6\n");
  fprintf(fp, "%d %d\n", w, h);
  fprintf(fp, "255\n");
  fwrite(img, w * h * 3, 1, fp);
  fclose(fp);
}


void render(unsigned char *img, int w, int h, int nsubsamples, 
            const Sphere* spheres, const Plane &plane)
{
  Kokkos::View<unsigned char*, Layout, MemSpace> vd_img("d_img", WIDTH*HEIGHT*3);
  Kokkos::View<Sphere*, Layout, MemSpace> vd_spheres("d_spheres", 3);
  auto d_img = vd_img.data();
  auto d_spheres = vd_spheres.data();
  Kokkos::Impl::DeepCopy<MemSpace, host_memory>(d_img, img, w*h*3*sizeof(unsigned char));
  Kokkos::Impl::DeepCopy<MemSpace, host_memory>(d_spheres, spheres, 3*sizeof(Sphere));

  Kokkos::parallel_for("render_kernel", 
  Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>> ({0,0}, {h,w}, {BLOCK_SIZE,BLOCK_SIZE}), 
  KOKKOS_LAMBDA(const int y, const int x){
    if (y < h && x < w) {

      RNG rng(y * w + x);
      float s0 = 0.f;
      float s1 = 0.f;
      float s2 = 0.f;

      for(int  v = 0; v < nsubsamples; v++ ) {
        for(int  u = 0; u < nsubsamples; u++ ) {
          float px = ( x + ( u / ( float )nsubsamples ) - ( w / 2.0f ) ) / ( w / 2.0f );
          float py = -( y + ( v / ( float )nsubsamples ) - ( h / 2.0f ) ) / ( h / 2.0f );

          Ray ray;
          ray.org.x = 0.f;
          ray.org.y = 0.f;
          ray.org.z = 0.f;
          ray.dir.x = px;
          ray.dir.y = py;
          ray.dir.z = -1.f;
          vnormalize( &( ray.dir ) );

          Isect isect;
          isect.t = 1.0e+17f;
          isect.hit = 0;

          ray_sphere_intersect( &isect, &ray, &d_spheres[0]);
          ray_sphere_intersect( &isect, &ray, &d_spheres[1]);
          ray_sphere_intersect( &isect, &ray, &d_spheres[2]);
          ray_plane_intersect ( &isect, &ray, &plane);

          if( isect.hit ) {
            Vec col;
            ambient_occlusion( &col, &isect, d_spheres, &plane, rng );
            s0 += col.x;
            s1 += col.y;
            s2 += col.z;
          }
        }
      }
      d_img[ 3 * ( y * w + x ) + 0 ] = my_clamp ( s0 / ( float )( nsubsamples * nsubsamples ) );
      d_img[ 3 * ( y * w + x ) + 1 ] = my_clamp ( s1 / ( float )( nsubsamples * nsubsamples ) );
      d_img[ 3 * ( y * w + x ) + 2 ] = my_clamp ( s2 / ( float )( nsubsamples * nsubsamples ) );
    }
  });
  Kokkos::Impl::DeepCopy<host_memory, MemSpace>(img, d_img, w*h*3*sizeof(unsigned char));
}

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
  if (argc != 2) {
    printf("Usage: %s <iterations>\n", argv[0]);
    return 1;
  }
  const int LOOPMAX = atoi(argv[1]);

  // three spheres in the image
  Sphere spheres[3];
  Plane plane;

  init_scene(spheres, plane);

  unsigned char *img = ( unsigned char * )malloc( WIDTH * HEIGHT * 3 );
  // Kokkos::View<unsigned char*, Layout, MemSpace> d_img("d_img", WIDTH*HEIGHT*3);
  // Kokkos::View<Sphere*, Layout, MemSpace> d_spheres("d_spheres", 3);

  clock_t start;
  start = clock();
  for( int i = 0; i < LOOPMAX; ++i ){
    render(img, WIDTH, HEIGHT, NSUBSAMPLES, spheres, plane);
  }
  clock_t end = clock();
  float delta = ( float )end - ( float )start;
  float msec = delta * 1000.0 / ( float )CLOCKS_PER_SEC;

  printf( "Total render time (%d iterations): %f sec.\n", LOOPMAX, msec / 1000.0 );
  printf( "Average render time: %f sec.\n", msec / 1000.0 / (float)LOOPMAX );

  saveppm( "ao.ppm", WIDTH, HEIGHT, img );
  free( img );
  }
  Kokkos::finalize();
  return 0;
}
