typedef enum {
    SEAM_CARVER_STANDARD_MODE,
    SEAM_CARVER_UPDATE_MODE,
    SEAM_CARVER_APPROX_MODE
} seam_carver_mode;

typedef struct { int r; int g; int b; } pixel;

int next_pow2(int n){
  int res = 1;
  while(res < n)
    res = res*2;
  return res;
}

pixel *build_pixels(const unsigned char *imgv, int w, int h){
  pixel *pixels = (pixel*)malloc(w*h*sizeof(pixel));
  pixel pix;
  for(int i = 0; i < h; i++){
    for(int j = 0; j < w; j++){
      pix.r = imgv[i*3*w + 3*j];
      pix.g = imgv[i*3*w + 3*j + 1];
      pix.b = imgv[i*3*w + 3*j + 2];
      pixels[i*w + j] = pix;
    }
  }
  return pixels;
}

unsigned char *flatten_pixels(pixel *pixels, int w, int h, int new_w){
  unsigned char *flattened = (unsigned char*)malloc(3*new_w*h*sizeof(unsigned char));
  for(int i = 0; i < h; i++){
    for(int j = 0; j < new_w; j++){
      pixel pix = pixels[i*w + j];
      flattened[3*i*new_w + 3*j] = pix.r;
      flattened[3*i*new_w + 3*j + 1] = pix.g;
      flattened[3*i*new_w + 3*j + 2] = pix.b;
    }
  }
  return flattened;
}
