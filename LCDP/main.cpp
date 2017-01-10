#include <iostream>

#include <opencv2\opencv.hpp>
#include "BackgroundSubtractorLCDP.h"

int main() {

	std::string filename = "../../fall.avi";
	cv::VideoCapture videoCapture;
	videoCapture.open(filename);
	// Checking video whether successful be opened
	if (!videoCapture.isOpened()) {
		std::cout << "Video having problem. Cannot open the video file." << std::endl;
		return -1;
	}
	// Getting frames per second (FPS) of the input video
	double FPS = videoCapture.get(cv::CAP_PROP_FPS);
	double FRAME_COUNT = videoCapture.get(cv::CAP_PROP_FRAME_COUNT);

	// Display windows
	cv::namedWindow("Input Video");

	// Background Subtractor Initialize
	BackgroundSubtractorLCDP backgroundSubtractorLCDP;

	// Input frame
	cv::Mat inputFrame;
	for (int frameIndex = 1;frameIndex <= FRAME_COUNT;frameIndex++) {
		bool inputCheck = videoCapture.read(inputFrame);
		if (!inputCheck) {
			std::cout << "Video having problem. Cannot read the frame from video file." << std::endl;
		}



		cv::imshow("Input Video", inputFrame);
		// If 'esc' key is pressed, break loop
		if (cv::waitKey(1) == 27)
		{
			std::cout << "Program ended by users." << std::endl;
			break;
		}
	}
	return 0;
}