#include <cstdlib>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <colmap/sensor/bitmap.h>
#include <ctime>

using namespace std;
/**
 * This function converts OpenCV Mat type to the Bitmap type used in COLMAP.
 * 
 * @param src Input OpenCV Mat to be converted, representing the image. By default, it is assumed to have BGR channel order.
 * @param dst Output COLMAP Bitmap. It can be initialized or left uninitialized; the function checks internally.
*/
void Mat2Bitmap(cv::Mat &src, colmap::Bitmap &dst)
{
    int width = src.cols;
    int height = src.rows;
    int channels = src.channels();
    bool is_rgb = channels == 1 ? false : true;
    int step = is_rgb ? 3 : 1;

    if (dst.Data() == nullptr) {
        dst.Allocate(width, height, is_rgb);
    } else if (width == dst.Width() && height == dst.Height()) {
        throw "dst.size() != src.size()";
    }

    for (int row = 0; row < height; ++row) {
        uchar *uc_pixel = src.data + row * src.step;
        for (int col = 0; col < width; ++col) {
            colmap::BitmapColor<uint8_t> color = is_rgb ? 
                    colmap::BitmapColor<uint8_t>(uc_pixel[2], uc_pixel[1], uc_pixel[0]) : 
                    colmap::BitmapColor<uint8_t>(uc_pixel[0]);
            dst.SetPixel(col, row, color);
            uc_pixel += step;
        }
    }
}

int main(int argc, char** argv) {
    cv::Mat lena = cv::imread("./Lenna.png", -1);

    // if you want to convert cv::mat gray image to bitmap
    cv::Mat lena_gray;
    cv::cvtColor(lena, lena_gray, cv::COLOR_RGB2GRAY);
    colmap::Bitmap bitmap_gray;
    clock_t start_gray = clock();
    Mat2Bitmap(lena_gray, bitmap_gray);
    clock_t end_gray = clock();
    cout << "time of gray = " << double(end_gray - start_gray) / CLOCKS_PER_SEC << endl;

    // or you want to convert cv::mat bgr image to bitmap
    colmap::Bitmap bitmap_color;
    clock_t start_bgr = clock();
    Mat2Bitmap(lena, bitmap_color);
    clock_t end_bgr = clock();
    cout << "time of bgr = " << double(end_bgr - start_bgr) / CLOCKS_PER_SEC << endl;

    // save image
    bitmap_gray.Write("Lenna_bitmap_gray.jpg");
    bitmap_color.Write("Lenna_bitmap_bgr.jpg");

    return 0;
}