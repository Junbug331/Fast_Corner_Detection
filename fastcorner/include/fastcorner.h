#ifndef FASTCORNER_H_
#define FASTCORNER_H_

#include <memory>

typedef unsigned char uchar;
std::tuple<u_int8_t *, int, int> reduceImage(uint8_t *img, int width, int height, float scaleX, float scaleY);
void fastCornerDetection(uchar *input, float *output, int width, int height, int T1, int T2);

#endif