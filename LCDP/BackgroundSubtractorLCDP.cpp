#include "BackgroundSubtractorLCDP.h"


BackgroundSubtractorLCDP::BackgroundSubtractorLCDP(int FRAME_HEIGHT, int FRAME_WIDTH) {
	// Initialize the parameters and background model
	Initialize();
}
BackgroundSubtractorLCDP::~BackgroundSubtractorLCDP() {

}

void BackgroundSubtractorLCDP::Process(const cv::Mat & img_input, cv::Mat & img_output)
{
	if (!modelInitCheck) {
		Initialize(img_input);
	}
}

void BackgroundSubtractorLCDP::Initialize(cv::Mat img_input)
{
	int totalPixel = img_input.cols*img_input.rows;
	modelLut = cv::Mat(img_input.size(), CV_64F, cv::Scalar_<uchar>(0));
	std::vector<std::vector<descriptor>> model(totalPixel);
	int counting = 1;
	for (int rowIndex = 0;rowIndex < img_input.rows;rowIndex++) {
		for (int colIndex = 0;colIndex < img_input.cols;colIndex++) {
			modelLut.at<double>(rowIndex, colIndex) = counting;
			model.at(counting) = std::vector<descriptor>(wordsNo);
				counting++;
		}
	}

}

void BackgroundSubtractorLCDP::DescriptorGenerator()
{
}
