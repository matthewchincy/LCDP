#pragma once

#ifndef __BackgroundSubtractorLCDP_H_INCLUDED
#define __BackgroundSubtractorLCDP_H_INCLUDED
#include <opencv2\opencv.hpp>
#include <vector>


class BackgroundSubtractorLCDP {
public:
	// Constructer
	BackgroundSubtractorLCDP(int FRAME_HEIGHT, int FRAME_WIDTH);

	// Destructer
	~BackgroundSubtractorLCDP();

	struct descriptor {
		int rgb[3];
		int frameCount;
		int p;
		int q;
		std::vector<int> LCDP;
	};
	
	// Program processing
	void Process(const cv::Mat &img_input, cv::Mat &img_output);
	// Parameters and Model initialize
	void Initialize(cv::Mat img_input);
	// Descriptor Generator
	void DescriptorGenerator();

protected:
	// Model initialization check
	bool modelInitCheck = false;

	// Model lookup table
	cv::Mat modelLut;
	// Total number of words per pixel
	int wordsNo;

};
#endif