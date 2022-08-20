#ifndef FASTCORNER_H_
#define FASTCORNER_H_

#include <memory>

typedef unsigned char uchar;
std::tuple<u_int8_t *, int, int> reduceImage(uint8_t *img, int width, int height, float scaleX, float scaleY);
void fastCornerDetection(uchar *input, float *output, int width, int height, int T1=200, int T2=300, int window_size=5, float scaleX=2.0, float scaleY=2.0);

#endif