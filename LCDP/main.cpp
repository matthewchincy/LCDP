#include <iostream>
#include "BackgroundSubtractorLCDP.h"
#include <opencv2\opencv.hpp>

int main() {
	
	int a = 1;
	cv::Mat img = cv::imread("test.png");
	cv::imshow("Test", img);
	cv::waitKey();
	return 0;
}