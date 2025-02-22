/**********************************************************************
  Copyright �2013 Advanced Micro Devices, Inc. All rights reserved.

  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

  �   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  �   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ********************************************************************/
#ifndef SDKBITMAP_H_
#define SDKBITMAP_H_

/**
 * Headers
 */
#include <cstdlib>
#include <iostream>
#include <string.h>
#include <stdio.h>

#define SDK_SUCCESS 0
#define SDK_FAILURE 1

static const short bitMapID = 19778;


/**
 * fixme this needs to be moved to common types header?
 */
#pragma pack(push,1)

/**
 * uchar4S
 * struct implements a vector of chars
 */
typedef struct __attribute__((__aligned__(4)))
{
  unsigned char x;
  unsigned char y;
  unsigned char z;
  unsigned char w;
} uchar4S;

KOKKOS_INLINE_FUNCTION uchar4S operator^(const uchar4S &a, const uchar4S &b) {
  uchar4S result;
  result.x = a.x ^ b.x;
  result.y = a.y ^ b.y;
  result.z = a.z ^ b.z;
  result.w = a.w ^ b.w;
  return result;
}

typedef unsigned char ucharS;

/**
 * ColorPelette of type uchar4S
 */
typedef uchar4S ColorPalette;

/**
 * struct Bitmap header
 */
typedef struct
{
  short id;
  int size;
  short reserved1;
  short reserved2;
  int offset;
} BitMapHeader;


/**
 * struct Bitmap info header
 */
typedef struct
{
  int sizeInfo;
  int width;
  int height;
  short planes;
  short bitsPerPixel;
  unsigned compression;
  unsigned imageSize;
  int xPelsPerMeter;
  int yPelsPerMeter;
  int clrUsed;
  int clrImportant;
} BitMapInfoHeader;

/**
 *class Bitmap used to load a bitmap image from a file.
 */
class SDKBitMap : public BitMapHeader, public BitMapInfoHeader
{
  private:
    uchar4S * pixels_;               /**< Pixel Data */
    int numColors_;                 /**< Number of colors */
    ColorPalette * colors_;         /**< Color Data */
    bool isLoaded_;                 /**< If Bitmap loaded */
    void releaseResources(void)     /**< Release Resources */
    {
      if (pixels_ != NULL)
      {
        delete[] pixels_;
      }
      if (colors_ != NULL)
      {
        delete[] colors_;
      }
      pixels_    = NULL;
      colors_    = NULL;
      isLoaded_  = false;
    }
    int colorIndex(uchar4S color)    /**< get a color index */
    {
      for (int i = 0; i < numColors_; i++)
      {
#if defined(SYCL_LANGUAGE_VERSION)
        if (colors_[i].x == color.x && colors_[i].y == color.y &&
            colors_[i].z == color.z && colors_[i].w == color.w)
#else
        if (colors_[i].x == color.x && colors_[i].y == color.y &&
            colors_[i].z == color.z && colors_[i].w == color.w)
#endif
        {
          return i;
        }
      }
      return SDK_SUCCESS;
    }
  public:

    /**
     * brief Default constructor
     */
    SDKBitMap() :
      pixels_(NULL), numColors_(0), colors_(NULL), isLoaded_(false) {}

    /**
     * brief Constructor
     * Tries to load bitmap image from filename provided.
     *
     * @param filename pointer to null terminated string that is the path and
     * filename to the bitmap image to be loaded.
     *
     * In the case of an error, e.g. the bitmap file could not be loaded for
     * some reason, then a following call to isLoaded will return false.
     */
    SDKBitMap(const char * filename) :
      pixels_(NULL), numColors_(0), colors_(NULL), isLoaded_(false) 
    {
      load(filename);
    }

    /**
     * Copy constructor
     *
     * @param rhs is the bitmap to be copied (cloned).
     */
    SDKBitMap(const SDKBitMap& rhs)
    {
      *this = rhs;
    }

    /**
     * Destructor
     */
    ~SDKBitMap()
    {
      releaseResources();
    }

    /**
     * Assignment operator
     * @param rhs is the bitmap to be assigned (cloned).
     */
    SDKBitMap& operator=(const SDKBitMap& rhs)
    {
      if (this == &rhs)
      {
        return *this;
      }
      // header
      id             = rhs.id;
      size           = rhs.size;
      reserved1      = rhs.reserved1;
      reserved2      = rhs.reserved2;
      offset         = rhs.offset;
      // header info
      sizeInfo       = rhs.sizeInfo;
      width          = rhs.width;
      height         = rhs.height;
      planes         = rhs.planes;
      bitsPerPixel   = rhs.bitsPerPixel;
      compression    = rhs.compression;
      imageSize      = rhs.imageSize;
      xPelsPerMeter  = rhs.xPelsPerMeter;
      yPelsPerMeter  = rhs.yPelsPerMeter;
      clrUsed        = rhs.clrUsed;
      clrImportant   = rhs.clrImportant;
      numColors_     = rhs.numColors_;
      isLoaded_      = rhs.isLoaded_;
      pixels_        = NULL;
      colors_        = NULL;
      if (isLoaded_)
      {
        if (rhs.colors_ != NULL)
        {
          colors_ = new ColorPalette[numColors_];
          if (colors_ == NULL)
          {
            isLoaded_ = false;
            return *this;
          }
          memcpy(colors_, rhs.colors_, numColors_ * sizeof(ColorPalette));
        }
        pixels_ = new uchar4S[width * height];
        if (pixels_ == NULL)
        {
          delete[] colors_;
          colors_   = NULL;
          isLoaded_ = false;
          return *this;
        }
        memcpy(pixels_, rhs.pixels_, width * height * sizeof(uchar4S));
      }
      return *this;
    }

    /**
     * Load Bitmap image
     *
     * @param filename is a pointer to a null terminated string that is the
     * path and filename name to the the bitmap file to be loaded.
     *
     * @return In the case of an error, e.g. the bitmap file could not be loaded for
     * some reason, then a following call to isLoaded will return false.
     */
    void
      load(const char * filename)
      {
        size_t val;
        // Release any existing resources
        releaseResources();
        // Open BMP file
        FILE * fd = fopen(filename, "rb");
        // FILE *fd;
        //fopen_s(&fd, filename, "rb");
        // Opened OK
        if (fd != NULL)
        {
          // Read header
          val = fread((BitMapHeader *)this, sizeof(BitMapHeader), 1, fd);
          // Failed to read header
          if (val != 1) 
          {
            fclose(fd);
            return;
          }
          // Confirm that we have a bitmap file
          if (id != bitMapID)
          {
            fclose(fd);
            return;
          }
          // Read map info header
          val = fread((BitMapInfoHeader *)this, sizeof(BitMapInfoHeader), 1, fd);
          // Failed to read map info header
          if (val != 1) 
          {
            fclose(fd);
            return;
          }

          // No support for compressed images
          if (compression)
          {
            fclose(fd);
            return;
          }
          // Support only 8 or 24 bits images
          if (bitsPerPixel < 8)
          {
            fclose(fd);
            return;
          }
          // Store number of colors
          numColors_ = 1 << bitsPerPixel;
          //load the palate for 8 bits per pixel
          if(bitsPerPixel == 8)
          {
            colors_ = new ColorPalette[numColors_];
            if (colors_ == NULL)
            {
              fclose(fd);
              return;
            }
            val  = fread(
                (char *)colors_,
                numColors_ * sizeof(ColorPalette),
                1,
                fd);

            // Failed to read colors
            if (val != 1) 
            {
              fclose(fd);
              return;
            }

          }
          // Allocate buffer to hold all pixels
          unsigned int sizeBuffer = size - offset;
          unsigned char * tmpPixels = new unsigned char[sizeBuffer];
          if (tmpPixels == NULL)
          {
            delete colors_;
            colors_ = NULL;
            fclose(fd);
            return;
          }
          // Read pixels from file, including any padding
          val = fread(tmpPixels, sizeBuffer * sizeof(unsigned char), 1, fd);
          // Failed to read pixel data
          if (val != 1) 
          {
            delete colors_;
            colors_ = NULL;
            delete[] tmpPixels;
            fclose(fd);
            return;
          }
          // Allocate image
          pixels_ = new uchar4S[width * height];
          if (pixels_ == NULL)
          {
            delete colors_;
            colors_ = NULL;
            delete[] tmpPixels;
            fclose(fd);
            return;
          }
          // Set image, including w component (white)
          memset(pixels_, 0xff, width * height * sizeof(uchar4S));
          unsigned int index = 0;
          for(int y = 0; y < height; y++)
          {
            for(int x = 0; x < width; x++)
            {
              // Read RGB values
              if (bitsPerPixel == 8)
              {
                pixels_[(y * width + x)] = colors_[tmpPixels[index++]];
              }
              else   // 24 bit
              {
#if defined(SYCL_LANGUAGE_VERSION)
                pixels_[(y * width + x)].z = tmpPixels[index++];
                pixels_[(y * width + x)].y = tmpPixels[index++];
                pixels_[(y * width + x)].x = tmpPixels[index++];
#else
                pixels_[(y * width + x)].z = tmpPixels[index++];
                pixels_[(y * width + x)].y = tmpPixels[index++];
                pixels_[(y * width + x)].x = tmpPixels[index++];
#endif
              }
            }
            // Handle padding
            for(int x = 0; x < (4 - (3 * width) % 4) % 4; x++)
            {
              index++;
            }
          }
          // Loaded file so we can close the file.
          fclose(fd);
          delete[] tmpPixels;
          // Loaded file so record this fact
          isLoaded_  = true;
        }
        else 
        {
          fprintf(stderr, "Failed to load file %s\n", filename);
        }
      }

    /**
     * Write Bitmap image
     *
     * @param filename is a pointer to a null terminated string that is the
     * path and filename name to the the bitmap file to be written.
     *
     * @return In the case that the bitmap is written true is returned. In
     * the case that a bitmap image is not already loaded or the write fails
     * for some reason false is returned.
     */
    bool
      write(const char * filename)
      {
        if (!isLoaded_)
        {
          return false;
        }
        // Open BMP file
        FILE * fd = fopen(filename, "wb");
        //FILE * fd;
        //fopen_s(&fd, filename, "wb");
        // Opened OK
        if (fd != NULL)
        {
          // Write header
          fwrite((BitMapHeader *)this, sizeof(BitMapHeader), 1, fd);
          // Failed to write header
          if (ferror(fd))
          {
            fclose(fd);
            return false;
          }
          // Write map info header
          fwrite((BitMapInfoHeader *)this, sizeof(BitMapInfoHeader), 1, fd);
          // Failed to write map info header
          if (ferror(fd))
          {
            fclose(fd);
            return false;
          }
          // Write palate for 8 bits per pixel
          if(bitsPerPixel == 8)
          {
            fwrite(
                (char *)colors_,
                numColors_ * sizeof(ColorPalette),
                1,
                fd);
            // Failed to write colors
            if (ferror(fd))
            {
              fclose(fd);
              return false;
            }
          }
          for(int y = 0; y < height; y++)
          {
            for(int x = 0; x < width; x++)
            {
              // Read RGB values
              if (bitsPerPixel == 8)
              {
                fputc(
                    colorIndex(
                      pixels_[(y * width + x)]),
                    fd);
              }
              else   // 24 bit
              {
#if defined(SYCL_LANGUAGE_VERSION)
                fputc(pixels_[(y * width + x)].z, fd);
                fputc(pixels_[(y * width + x)].y, fd);
                fputc(pixels_[(y * width + x)].x, fd);
#else
                fputc(pixels_[(y * width + x)].z, fd);
                fputc(pixels_[(y * width + x)].y, fd);
                fputc(pixels_[(y * width + x)].x, fd);
#endif
                if (ferror(fd))
                {
                  fclose(fd);
                  return false;
                }
              }
            }
            // Add padding
            for(int x = 0; x < (4 - (3 * width) % 4) % 4; x++)
            {
              fputc(0, fd);
            }
          }
          return true;
        }
        return false;
      }

    bool
      write(const char * filename, int width, int height, unsigned int *ptr)
      {
        // Open BMP file
        FILE * fd = fopen(filename, "wb");
        int alignSize  = width * 4;
        alignSize ^= 0x03;
        alignSize ++;
        alignSize &= 0x03;
        int rowLength = width * 4 + alignSize;
        // Opened OK
        if (fd != NULL)
        {
          BitMapHeader *bitMapHeader = new BitMapHeader;
          bitMapHeader->id = bitMapID;
          bitMapHeader->offset = sizeof(BitMapHeader) + sizeof(BitMapInfoHeader);
          bitMapHeader->reserved1 = 0x0000;
          bitMapHeader->reserved2 = 0x0000;
          bitMapHeader->size = sizeof(BitMapHeader) + sizeof(BitMapInfoHeader) + rowLength
            * height;
          // Write header
          fwrite(bitMapHeader, sizeof(BitMapHeader), 1, fd);
          // Failed to write header
          if (ferror(fd))
          {
            fclose(fd);
            return false;
          }
          BitMapInfoHeader *bitMapInfoHeader = new BitMapInfoHeader;
          bitMapInfoHeader->bitsPerPixel = 32;
          bitMapInfoHeader->clrImportant = 0;
          bitMapInfoHeader->clrUsed = 0;
          bitMapInfoHeader->compression = 0;
          bitMapInfoHeader->height = height;
          bitMapInfoHeader->imageSize = rowLength * height;
          bitMapInfoHeader->planes = 1;
          bitMapInfoHeader->sizeInfo = sizeof(BitMapInfoHeader);
          bitMapInfoHeader->width = width;
          bitMapInfoHeader->xPelsPerMeter = 0;
          bitMapInfoHeader->yPelsPerMeter = 0;
          // Write map info header
          fwrite(bitMapInfoHeader, sizeof(BitMapInfoHeader), 1, fd);
          // Failed to write map info header
          if (ferror(fd))
          {
            fclose(fd);
            return false;
          }
          unsigned char buffer[4];
          int x, y;
          for (y = 0; y < height; y++)
          {
            for (x = 0; x < width; x++, ptr++)
            {
              if( 4 != fwrite(ptr, 1, 4, fd))
              {
                fclose(fd);
                return false;
              }
            }
            memset( buffer, 0x00, 4 );
            fwrite( buffer, 1, alignSize, fd );
          }
          fclose( fd );
          return true;
        }
        return false;
      }

    /**
     * Get image width
     *
     * @return If a bitmap image has been successfully loaded, then the width
     * image is returned, otherwise -1;
     */
    int
      getWidth(void) const
      {
        if (isLoaded_)
        {
          return width;
        }
        else
        {
          return -1;
        }
      }


    /**
     * Get Number of Channels
     *
     * @return the number of channels used in image, otherwise -1.
     */
    int getNumChannels()
    {
      if (isLoaded_)
      {
        return bitsPerPixel / 8;
      }
      else
      {
        return SDK_FAILURE;
      }
    }

    /**
     * Get image height
     *
     * @return If a bitmap image has been successfully loaded, then the height
     * image is returned, otherwise -1.
     */
    int
      getHeight(void) const
      {
        if (isLoaded_)
        {
          return height;
        }
        else
        {
          return -1;
        }
      }

    /**
     * Get image width
     *
     * @return If a bitmap image has been successfully loaded, then returns
     * a pointer to image's pixels, otherwise NULL.
     */
    uchar4S * getPixels(void) const
    {
      return pixels_;
    }

    /**
     * Is an image currently loaded
     *
     * @return If a bitmap image has been successfully loaded, then returns
     * true, otherwise if an image could not be loaded or an image has yet
     * to be loaded false is returned.
     */
    bool
      isLoaded(void) const
      {
        return isLoaded_;
      }

};
#pragma pack(pop)
#endif //CL_BITMAP
