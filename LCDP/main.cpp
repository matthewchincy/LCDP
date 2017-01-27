#include <iostream>
#include <direct.h>
#include <opencv2\opencv.hpp>
#include "BackgroundSubtractorLCDP.h"
#include <time.h>

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string currentDateTime() {
	time_t     now = time(0);
	struct tm  tstruct;
	char       buf[80];
	tstruct = *localtime(&now);
	strftime(buf, sizeof(buf), "%d%m%y-%H%M%S", &tstruct);

	return buf;
}
int main() {

	//std::string filename = "../../fall.avi";
	//std::string filename = "C:/Users/mattc/Desktop/Background Subtraction/Results/bungalows/bungalows.avi";
	std::string filename;
	std::cout << "Video name? ";
	getline(std::cin, filename);
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
	else {
		std::cout << "Video successful loaded!" << std::endl;
	}
	// Getting frames per second (FPS) of the input video
	const double FPS = videoCapture.get(cv::CAP_PROP_FPS);
	const double FRAME_COUNT = videoCapture.get(cv::CAP_PROP_FRAME_COUNT);
	const cv::Size FRAME_SIZE = cv::Size(videoCapture.get(cv::CAP_PROP_FRAME_WIDTH), videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT));
	cv::Mat ROI_FRAME;
	ROI_FRAME.create(FRAME_SIZE, CV_8UC1);
	ROI_FRAME = cv::Scalar_<uchar>(255);
	// Display windows
	cv::namedWindow("Input Video");
	size_t maxWordsNo = 25;
	// Background Subtractor Initialize
	BackgroundSubtractorLCDP backgroundSubtractorLCDP(FRAME_SIZE, ROI_FRAME, maxWordsNo);
	backgroundSubtractorLCDP.Initialize(inputFrame,inputFrame);
	// current date/time based on current system
	time_t now = time(0);
	tm *ltm = localtime(&now);
	std::string folderName = "LCD Sample Consensus-" + currentDateTime();
	const char *s1;
	s1 = folderName.c_str();
	_mkdir(s1);
	folderName += "/results";
	s1 = folderName.c_str();
	_mkdir(s1);
	char s[25];
	for (int currFrameIndex = 1;currFrameIndex <= FRAME_COUNT;currFrameIndex++) {
		bool inputCheck = videoCapture.read(inputFrame);
		if (!inputCheck) {
			std::cout << "Video having problem. Cannot read the frame from video file." << std::endl;
		}
		backgroundSubtractorLCDP.Process(inputFrame, fgMask);
		cv::imshow("Input Video", inputFrame);
		cv::imshow("Results", fgMask);
		std::string saveFolder;

		sprintf(s, "/bin%06d.png", (currFrameIndex));
		saveFolder = folderName +s;
		cv::imwrite(saveFolder, fgMask);
		// If 'esc' key is pressed, break loop
		if (cv::waitKey(1) == 27)
		{
			std::cout << "Program ended by users." << std::endl;
			break;
		}
	}
	return 0;
}