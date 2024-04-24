#include <cstdlib>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <colmap/sensor/bitmap.h>
#include <colmap/feature/sift.h>
#include <colmap/scene/two_view_geometry.h>
#include "colmap/ui/model_viewer_widget.h"
#include "colmap/util/misc.h"
#include "colmap/ui/image_viewer_widget.h"
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

cv::Scalar generateRandomColor() {
    int r = rand() % 256;
    int g = rand() % 256;
    int b = rand() % 256;
    return cv::Scalar(b, g, r);
}

int main(int argc, char** argv) {
    cv::Mat left_gray_image, right_gray_image;
    cv::Mat left_image = cv::imread("../assets/1.png", -1);
    cv::Mat right_image = cv::imread("../assets/2.png", -1);
    cv::cvtColor(left_image, left_gray_image, cv::COLOR_RGB2GRAY);
    cv::cvtColor(right_image, right_gray_image, cv::COLOR_RGB2GRAY);

    colmap::Bitmap left_bitmap_gray, right_bitmap_gray;
    Mat2Bitmap(left_gray_image, left_bitmap_gray);
    Mat2Bitmap(right_gray_image, right_bitmap_gray);

    colmap::SiftExtractionOptions sifit_options;
    sifit_options.use_gpu = true;
    sifit_options.estimate_affine_shape = false;
    sifit_options.domain_size_pooling = false;
    sifit_options.force_covariant_extractor = false;
    auto extractor = CreateSiftFeatureExtractor(sifit_options);

    colmap::FeatureKeypoints keypoints_left, keypoints_right;
    colmap::FeatureDescriptors descriptors_left, descriptors_right;

    

    clock_t start_sift = clock();
    extractor->Extract(left_bitmap_gray, &keypoints_left, &descriptors_left);
    extractor->Extract(right_bitmap_gray, &keypoints_right, &descriptors_right);
    clock_t end_sift = clock();
    cout << "time of extract = " << double(end_sift - start_sift) / CLOCKS_PER_SEC * 1e3 << " ms" << endl;
    
    auto keypoints_left_ptr = std::make_shared<colmap::FeatureKeypoints>(keypoints_left);
    auto keypoints_right_ptr = std::make_shared<colmap::FeatureKeypoints>(keypoints_right);
    const auto descriptors_left_ptr = std::make_shared<colmap::FeatureDescriptors>(descriptors_left);
    const auto descriptors_right_ptr = std::make_shared<colmap::FeatureDescriptors>(descriptors_right);

    colmap::SiftMatchingOptions matcher_options;
    matcher_options.use_gpu = true;
    matcher_options.max_num_matches = 3000;
    auto matcher = colmap::CreateSiftFeatureMatcher(matcher_options);

    constexpr double kMaxError = 4.0;
    colmap::FeatureMatches matches_gpu;
    clock_t start_match = clock();
    colmap::TwoViewGeometry two_view_geometry;
    two_view_geometry.config = colmap::TwoViewGeometry::UNCALIBRATED;
    two_view_geometry.H = Eigen::Matrix3d::Identity();
    if (0) {
        matcher->Match(descriptors_left_ptr, descriptors_right_ptr, &matches_gpu);
    } else {
        matcher->MatchGuided(kMaxError, keypoints_left_ptr, keypoints_right_ptr, descriptors_left_ptr, descriptors_right_ptr, &two_view_geometry);
    }

    cv::Mat match_img;
    cv::hconcat(left_image, right_image, match_img);
    srand(time(0));
    cout << "inlier_matches = " << two_view_geometry.inlier_matches.size() << endl;
    for (int i = 0; i < two_view_geometry.inlier_matches.size(); ++i) {
        int left_idx = two_view_geometry.inlier_matches[i].point2D_idx1;
        int right_idx = two_view_geometry.inlier_matches[i].point2D_idx2;
        cv::Point p0(keypoints_left[left_idx].x, keypoints_left[left_idx].y);
        cv::Point p1(keypoints_right[right_idx].x, keypoints_right[right_idx].y);
        p1.x += left_image.cols;
        cv::circle(match_img, p0, 2, cv::Scalar(0, 255, 0), -1);
        cv::circle(match_img, p1, 2, cv::Scalar(0, 255, 0), -1);
        cv::line(match_img, p0, p1, generateRandomColor());
    }

    cv::imshow("match_image", match_img);
    cv::waitKey();

    return 0;
}
