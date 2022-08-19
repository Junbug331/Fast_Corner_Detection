#include <iostream>
#include <cmath>
#include <tuple>
#include <vector>
#include "fastcorner.h"

std::tuple<u_int8_t *, int, int> reduceImage(uint8_t *img, int width, int height, float scaleX, float scaleY)
{
    /*
     * ex) scaleX = 2, scaleY = 2
     * new_width = width/2, new_height = height/2
     */
    if (!img)
        return {nullptr, 0, 0};

    int new_width = int(float(width)/scaleX + 0.5);
    int new_height = int(float(height) / scaleY+ 0.5) ;
    int num_size = int(scaleX * scaleY);
    uint8_t *output = new uchar[new_width*new_height];

    for (int y=0; y<new_height; ++y)
    {
        for (int x=0; x<new_width; ++x)
        {
            int x_input = float(x) * scaleX;
            int y_input = float(y) * scaleY;
            float val = 0;

            for (int i=y_input; i<y_input+int(scaleY+0.5); ++i)
                for (int j=x_input; j<x_input+int(scaleX+0.5); ++j)
                    if (i * width + j < width*height)
                        val += img[i*width+j];

            val /= num_size;
            output[y*new_width+x] = val;
        }
    }

    return {output, new_width, new_height};
}

void fastCornerDetection(uchar *input, float *output, int width, int height, int T1, int T2, int window_size)
{
    /**
     * 3 x 3 window example
     * input : pointer to gray image data
     * output : keypoint location
     */
    if (!input)
        return;

    // Get a low resolution image
    auto[low_res, low_width, low_height] = reduceImage(input, width, height, 2.0, 2.0);
    int scaleX = width / low_width;
    int scaleY = height / low_height;
    int padding = window_size/2;

    int IC, IA1, IB1, IA2, IB2;
    int rA, rB, C_sim;

    for (int r=padding; r<low_height-padding; ++r)
    {
        uchar* preRow = &low_res[(r-1) * low_width];
        uchar* currRow = &low_res[r * low_width];
        uchar* nextRow = &low_res[(r+1) * low_width];
        for (int c=1; c<low_width-1; ++c)
        {
            // Simple cornerness measure on low resoluion image
            // vertical
            IC = currRow[c];
            IA1 = currRow[c+padding];
            IA2 = currRow[c-padding];

            // horizontal
            IB1 = preRow[c];
            IB2 = nextRow[c];
            rA = (IA1 - IC)*(IA1 - IC) + (IA2 - IC)*(IA2 - IC);
            rB = (IB1 - IC)*(IB1 - IC) + (IB2 - IC)*(IB2 - IC);
            C_sim = std::min(rA, rB);

            // greater than T1 means it is a potential corner
            if (C_sim > T1)
            {
                int y = r*scaleY;
                int x = c*scaleX;
                if (y > height || x > width) continue;

                // Simple cornerness measure on input image(orignal size)
                IC = input[y*width + x];
                IA1 = input[y*width + x+padding];
                IA2 = input[y*width + x-padding];

                IB1 = input[(y-1)*width + x];
                IB2 = input[(y+1)*width + x];
                rA = (IA1 - IC)*(IA1 - IC) + (IA2 - IC)*(IA2 - IC);
                rB = (IB1 - IC)*(IB1 - IC) + (IB2 - IC)*(IB2 - IC);
                C_sim = std::min(rA, rB);

                if (C_sim > T2)
                {
                    // interpixel approximation cornerness measure
                    float B1, B2, A, B, C, C_interpixel;
                    C = rA;
                    B1 = (IB1 - IA1)*(IA1-IC) + (IB2-IA2)*(IA2-IC);
                    B2 = (IB1 - IA2)*(IA2-IC) + (IB2-IA1)*(IA1-IC);
                    B = std::min(B1, B2);
                    A = rB - rA - 2*B;
                    C_interpixel = (B < 0 && (A+B) > 0) ? C-(B*B/A) : C_sim;

                    if (C_interpixel > T2)
                        output[y*width + x] = C_interpixel;

                }
            }
        }
    }

    delete[] low_res;
}
