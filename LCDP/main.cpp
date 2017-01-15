#include <iostream>

#include <opencv2\opencv.hpp>
#include "BackgroundSubtractorLCDP.h"

int main() {

	//std::string filename = "../../fall.avi";
	std::string filename = "C:/Users/mattc/Desktop/Background Subtraction/Results/fall/fall.avi";
	cv::VideoCapture videoCapture;
	videoCapture.open(filename);
	// Input frame
	cv::Mat inputFrame;
	// FG Mask
	cv::Mat fgMask;
	videoCapture >> inputFrame;
	videoCapture.set(cv::CAP_PROP_POS_FRAMES, 0);

	// Checking video whether successful be opened
	if (!videoCapture.isOpened() || inputFrame.empty()) {
		std::cout << "Video having problem. Cannot open the video file." << std::endl;
		return -1;
	}
	// Getting frames per second (FPS) of the input video
	double FPS = videoCapture.get(cv::CAP_PROP_FPS);
	double FRAME_COUNT = videoCapture.get(cv::CAP_PROP_FRAME_COUNT);
	int FRAME_HEIGHT = videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT);
	int FRAME_WIDTH = videoCapture.get(cv::CAP_PROP_FRAME_WIDTH);
	// Display windows
	cv::namedWindow("Input Video");
	size_t maxWordsNo = 50;
	// Background Subtractor Initialize
	BackgroundSubtractorLCDP backgroundSubtractorLCDP(FRAME_HEIGHT,FRAME_WIDTH, maxWordsNo);
	backgroundSubtractorLCDP.Initialize(inputFrame,inputFrame);
	
	for (int frameIndex = 1;frameIndex <= FRAME_COUNT;frameIndex++) {
		bool inputCheck = videoCapture.read(inputFrame);
		if (!inputCheck) {
			std::cout << "Video having problem. Cannot read the frame from video file." << std::endl;
		}


		backgroundSubtractorLCDP.Process(inputFrame, fgMask);
		cv::imshow("Input Video", inputFrame);
		cv::imshow("Results", fgMask);
		// If 'esc' key is pressed, break loop
		if (cv::waitKey(1) == 27)
		{
			std::cout << "Program ended by users." << std::endl;
			break;
		}
	}
	return 0;
}