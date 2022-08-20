#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

#include "fastcorner.h"

using namespace std;
using namespace cv;

int main()
{
    string filename = string(RES_DIR) + "/" + string("test4.jpg");
    Mat img = imread(filename);
    Mat imgGray;
    cvtColor(img, imgGray, COLOR_BGR2GRAY);
    int width = img.cols;
    int height = img.rows;
    size_t size = width*height;

    shared_ptr<uint8_t> input_data(new uint8_t [size]);
    memcpy(input_data.get(), imgGray.data, sizeof(uchar)*size);

    shared_ptr<float> output_data(new float[size]{0, });

    fastCornerDetection(input_data.get(), output_data.get(), width, height, 200, 300, 7);

    for (int r=0; r<height; r++)
    {
        for (int c=0; c<width; c++)
        {
            if (output_data.get()[r*width + c] != 0.0)
            {
                cv::circle(img, Point(c, r), 2, Scalar(255, 255, 0));
            }
        }
    }

    imwrite(string(ROOT_DIR)+"/"+string("result.jpg"), img);
    cout << "Finished" << endl;
}

