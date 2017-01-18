#pragma once

#ifndef __BackgroundSubtractorLCDP_H_INCLUDED
#define __BackgroundSubtractorLCDP_H_INCLUDED
#include <opencv2\opencv.hpp>
#include <vector>
//! defines the default value for BackgroundSubtractorPAWCS::m_nMaxLocalWords and m_nMaxGlobalWords
#define BG_DEFAULT_MAX_NO_WORDS (50)
#define PRE_DEFAULT_GAUSSIAN_SIZE cv::Size(3,3);

class BackgroundSubtractorLCDP {
public:
	// Constructer
	BackgroundSubtractorLCDP(cv::Size inputFrameSize, cv::Mat inputROI, size_t inputWordsNo);

	// Destructer
	~BackgroundSubtractorLCDP();
	
	// Program processing
	void Process(const cv::Mat INPUT_IMG, cv::Mat &OUTPUT_IMG);

	// Parameters and Model initialize
	void Initialize(const cv::Mat INPUT_IMG,const cv::Mat INPUT_ROI);

	// Refreshes all samples based on the last analyzed frame
	void RefreshModel(const float REFRESH_FRACTION,const bool FORCE_UPDATE_SWITCH = false);
	

protected:
	// Neighbourhood's offset value
	cv::Point nbOffset[16];

	// Descriptor structure for each pixels
	struct Descriptor {
		// Store the pixel's RGB values
		uint rgb[3];
		// Store the number of frames that having same descriptor
		int frameCount;
		// Store the frame index of the first occurences/initial of this descriptor
		int p;
		// Store the frame index of the last occurences of this descriptor
		int q;
		// Store the pixel's LCDP values
		std::vector<uint> LCDP;
	};
	
	// Store the background's word and it's iterator
	Descriptor * bgWordPtr, * bgWordPtrIter;
	// Store the currect frame's word and it's iterator
	Descriptor * currWordPtr, *currWordPtrIter;
	// Model initialization check
	bool modelInitCheck = false;
	// Total number of words per pixel
	const size_t WORDS_NO;
	// Frame index
	size_t frameIndex;

	/*=====PRE-PROCESS Parameters=====*/
	// Size of gaussian filter
	const cv::Size preGaussianSize = PRE_DEFAULT_GAUSSIAN_SIZE;

	/*=====DESCRIPTOR Parameters=====*/
	// Size of neighbourhood 3(3x3)/5(5x5)
	const size_t descNbSize;
	// Total number of LCD's neighbour 8(3x3)/16(5x5)
	const size_t descNbNo;
	// LCD colour differences ratio
	const double descColorDiffRatio;
	// Total number of LCD differences per pixel
	const size_t descDiffNo;
	// Persistence's offset value;
	const size_t descOffsetValue;

	/*=====CLASSIFIER Parameters=====*/
	// RGB differences threshold
	const double clsRGBThreshold;
	// RGB detection switch
	const bool clsRGBDiffSwitch;
	// RGB bright pixel switch
	const bool clsRGBBrightPxSwitch;
	// LCDP differences threshold
	const double clsLCDPThreshold;
	// Maximum number of LCDP differences threshold
	const double clsLCDPMaxThreshold;
	// LCDP detection switch
	const bool clsLCDPDiffSwitch;
	// LCDP detection AND (true) OR (false) switch
	const bool clsAndOrSwitch;
	// Neighbourhood matching switch
	const bool clsNbMatchSwitch;
	// Total number of neighbour 8(3x3)/16(5x5)
	size_t clsNbNo;

	/*=====POST-PROCESS Parameters=====*/
	// Size of median filter
	size_t postMedianFilterSize;

	/*=====FRAME Parameters=====*/
	// ROI frame
	const cv::Mat frameRoi;
	// Size of region of interest
	const cv::Size frameRoiSize;
	// Size of input frame
	const cv::Size frameInitSize;
	// Total number of pixel of region of interest
	const int frameRoiTotalPixel;
	// Total number of pixel of input frame
	const int frameInitTotalPixel;

	/*=====UPDATE Parameters=====*/
	// Specifies whether Tmin/Tmax scaling is enabled or not
	bool upLearningRateScalingSwitch;
	// Specifies the px update spread range
	bool upUse3x3Spread;
	// Current learning rate caps
	float upLearningRateLowerCap;
	float upLearningRateUpperCap;
	// Initial blinking accumulate level
	cv::Mat upBlinkAccLevel;
	// Random replace model switch
	const bool upRandomReplaceSwitch;
	// Random update neighbourhood model switch
	const bool upRandomUpdateNbSwitch;
	// Total number of neighbour undergo neighbourhood updates 8(3X3)/16(5X5)
	size_t upNbNo;
	// Feedback loop switch
	const bool upFeedbackSwitch;

	size_t upSamplesForMovingAvgs;

	/*=====OTHERS Parameters=====*/
	// Local define used to specify the color dist threshold offset used for unstable regions
	double STAB_COLOR_DIST_OFFSET;

	/*=====RESULTS=====*/
	// Per-pixel distance thresholds ('R(x)', but used as a relative value to determine both 
	// intensity and descriptor variation thresholds)
	cv::Mat resDistThreshold;
	// A lookup map used to keep track of unstable regions (based on segm. noise & local dist. thresholds)
	cv::Mat resUnstableRegionMask;
	// Per-pixel mean raw segmentation results (used to detect unstable segmentation regions)
	cv::Mat resMeanRawSegmRes_LT;
	cv::Mat resMeanRawSegmRes_ST;
	// Per-pixel mean raw segmentation results (used to detect unstable segmentation regions)
	cv::Mat resMeanFinalSegmRes_LT;
	cv::Mat resMeanFinalSegmRes_ST;
	// Per-pixel update rates('T(x)', which contains pixel - level 'sigmas')
	cv::Mat resUpdateRate;
	// Per-pixel mean minimal distances from the model ('D_min(x)', used to control 
	// variation magnitude and direction of 'T(x)' and 'R(x)')
	cv::Mat resMeanMinDist_LT;
	cv::Mat resMeanMinDist_ST;
	// Per-pixel distance variation modulators ('v(x)', relative value used to modulate 'R(x)' 
	// and 'T(x)' variations)
	cv::Mat resVariationModulator;
	// Per-pixel blink detection map ('Z(x)')
	cv::Mat resBlinksFrame;
	// Minimum RGB distance
	cv::Mat resMinRGBDistance;
	// Minimum LCD distance
	cv::Mat resMinLCDPDistance;
	// Current foreground mask
	cv::Mat resCurrFGMask;
	// Previous foreground mask
	cv::Mat resLastFGMask;
	// The foreground mask generated by the method at [t-1] (without post-proc, used for blinking px detection)
	cv::Mat resLastRawFGMask;
	// The blink mask generated by the method at [t-1] (without post-proc, used for blinking px detection)
	cv::Mat resLastRawFGBlinkMask;
	// t-1 foreground mask
	cv::Mat resT_1FGMask;
	// t-2 foreground mask
	cv::Mat resT_2FGMask;
	cv::Mat resLastFGMaskDilated;
	cv::Mat resLastFGMaskDilatedInverted;
	cv::Mat resFGMaskFloodedHoles;
	cv::Mat resFGMaskPreFlood;
	cv::Mat resCurrRawFGBlinkMask;

	struct PxInfoBase {
		int imgCoord_Y;
		int imgCoord_X;
		size_t nModelIdx;
	};
	struct RGBPxIndex {
		size_t R;
		size_t G;
		size_t B;
	};
	struct PxNbBase {
		RGBPxIndex pxIndex[16];
	};
	// Internal pixel info LUT for all possible pixel indexes
	PxInfoBase* pxInfoLUT;
	PxNbBase* pxNbLUT;

	// Descriptor Generator
	void DescriptorGenerator(const cv::Mat inputFrame, const size_t nPxRGBIter, const PxInfoBase *pxInfoLUT, Descriptor *tempWord);
	// Border line reconstruct
	cv::Mat BorderLineReconst();
	// Compensation with Motion Hist
	cv::Mat CompensationMotionHist();
	// Local color difference generator
	std::vector<uint> LCDGenerator(const cv::Mat inputFrame, const PxInfoBase *pxInfoLUT);
	// Descriptor matching
	bool DescriptorMatching();
	// Generate neighbourhood offset value
	void GenerateNbOffset(const PxInfoBase pxInfoLUT, PxNbBase* pxNbLUT);
};
#endif