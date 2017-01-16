#pragma once

#ifndef __BackgroundSubtractorLCDP_H_INCLUDED
#define __BackgroundSubtractorLCDP_H_INCLUDED
#include <opencv2\opencv.hpp>
#include <vector>
//! defines the default value for BackgroundSubtractorPAWCS::m_nMaxLocalWords and m_nMaxGlobalWords
#define BG_DEFAULT_MAX_NO_WORDS (50)

class BackgroundSubtractorLCDP {
public:
	// Constructer
	BackgroundSubtractorLCDP(int FRAME_HEIGHT, int FRAME_WIDTH, size_t maxWordsNo);

	// Destructer
	~BackgroundSubtractorLCDP();


	// Program processing
	void Process(const cv::Mat &img_input, cv::Mat &img_output);
	// Parameters and Model initialize
	void Initialize(cv::Mat img_input, cv::Mat roi_input);
	// Refreshes all samples based on the last analyzed frame
	void RefreshModel(float refreshFraction, bool forceFGUpdateSwitch = false);
	

protected:

	struct Descriptor {
		cv::Vec3b rgb;
		int frameCount;
		int p;
		int q;
		std::vector<std::vector<int>> LCDP;
	};


	Descriptor* bgWord, * bgWordListIter;
	Descriptor* lastWord, * lastWordListIter;
	// Model initialization check
	bool modelInitCheck = false;

	// Total number of words per pixel
	size_t wordsNo;

	// Frame index
	size_t frameIndex;

	// PRE-PROCESS Parameters
	struct PreProcess {
		// Size of gaussian filter
		cv::Size gaussianSize;
	};

	// GENERATE DESCRIPTOR Parameters
	// LCDP Parameters
	struct GenLCDP {
		// Size of neighbourhood 3(3x3)/5(5x5)
		size_t nbSize;
		// Total number of LCD's neighbour 8(3x3)/16(5x5)
		size_t nbNo;
		// Size of the extention on the frame
		cv::Size2d extendSize;
		// LCD colour differences ratio
		double T_colorDiffRatio;
		// Total number of LCD differences per pixel
		size_t diffNo;
	};
	struct GenDesc {
		GenLCDP LCDP;
		// Persistence's offset value;
		size_t offsetValue;
	};

	// CLASSIFIER Parameters
	// RGB Classifier Parameters
	struct ClsRGB {
		// RGB differences threshold
		double T_RGB;
		// RGB detection switch
		bool RGBDiffSwitch;
		// RGB bright pixel switch
		bool RGBBrightPxSwitch;
	};
	// LCDP Classifier Parameters
	struct ClsLCDP {
		// LCDP differences threshold
		double T_LCDP;
		// Maximum number of LCDP differences threshold
		double Max_T_LCDP;
		// LCDP detection switch
		bool LCDPDiffSwitch;
		// LCDP detection AND (true) OR (false) switch
		bool AndOrSwitch;
	};
	// Classifier Parameters
	struct Classifier {
		ClsRGB rgb;
		ClsLCDP LCDP;
		// Neighbourhood matching switch
		bool nbMatchSwitch;
		// Total number of neighbour 8(3x3)/16(5x5)
		size_t nbNo;
	};

	// POST-PROCESS Parameters
	struct PostProcess {
		// Size of median filter
		size_t medianFilterSize;
	};

	// FRAME parameters
	struct Frame {
		// Size of region of interest
		cv::Size roiSize;
		// Size of input frame
		cv::Size initFrameSize;
		// Total number of pixel of input frame
		double initTotalPixel;
		// Total number of pixel of region of interest
		double roiTotalPixel;
	};

	// UPDATE parameters
	struct Update {
		// Specifies whether Tmin/Tmax scaling is enabled or not
		bool learningRateScalingSwitch;
		// Specifies the px update spread range
		bool use3x3Spread;
		// Current learning rate caps
		float learningRateLowerCap;
		float learningRateUpperCap;
		// Initial blinking accumulate level
		cv::Mat blinkAccLevel;
		// Random replace model switch
		bool randomReplaceSwitch;
		// Random update neighbourhood model switch
		bool randomUpdateNbSwitch;
		// Total number of neighbour undergo neighbourhood updates 8(3X3)/16(5X5)
		size_t nbNo;
		// Feedback loop switch
		bool feedbackSwitch;

		size_t samplesForMovingAvgs;
	};

	// Method Parameters
	struct MethodParam {
		PreProcess preProcess;
		GenDesc descriptor;
		Classifier classifier;
		PostProcess postProcess;
		Frame frame;
		Update update;
		// Local define used to specify the color dist threshold offset used for unstable regions
		double STAB_COLOR_DIST_OFFSET;
	};

	// Results
	struct ResModel {
		// Per-pixel distance thresholds ('R(x)', but used as a relative value to determine both 
		// intensity and descriptor variation thresholds)
		cv::Mat distThreshold;
		// A lookup map used to keep track of unstable regions (based on segm. noise & local dist. thresholds)
		cv::Mat unstableRegionMask;
		// Per-pixel mean raw segmentation results (used to detect unstable segmentation regions)
		cv::Mat meanRawSegmRes_LT;
		cv::Mat meanRawSegmRes_ST;
		// Per-pixel mean raw segmentation results (used to detect unstable segmentation regions)
		cv::Mat meanFinalSegmRes_LT;
		cv::Mat meanFinalSegmRes_ST;
		// Per-pixel update rates('T(x)', which contains pixel - level 'sigmas')
		cv::Mat updateRate;
		// Per-pixel mean minimal distances from the model ('D_min(x)', used to control 
		// variation magnitude and direction of 'T(x)' and 'R(x)')
		cv::Mat meanMinDist_LT;
		cv::Mat meanMinDist_ST;
		// Per-pixel distance variation modulators ('v(x)', relative value used to modulate 'R(x)' 
		// and 'T(x)' variations)
		cv::Mat variationModulator;
		// Per-pixel blink detection map ('Z(x)')
		cv::Mat blinksFrame;
	};
	struct Result {
		ResModel model;
		// Minimum RGB distance
		cv::Mat minRGBDistance;
		// Minimum LCD distance
		cv::Mat minLCDPDistance;
		// Current foreground mask
		cv::Mat currFGMask;
		// Previous foreground mask
		cv::Mat lastFGMask;
		// The foreground mask generated by the method at [t-1] (without post-proc, used for blinking px detection)
		cv::Mat lastRawFGMask;
		// The blink mask generated by the method at [t-1] (without post-proc, used for blinking px detection)
		cv::Mat lastRawFGBlinkMask;
		// t-1 foreground mask
		cv::Mat t_1FGMask;
		// t-2 foreground mask
		cv::Mat t_2FGMask;
		cv::Mat lastFGMaskDilated;
		cv::Mat lastFGMaskDilatedInverted;
		cv::Mat FGMaskFloodedHoles;
		cv::Mat FGMaskPreFlood;
		cv::Mat currRawFGBlinkMask;
	};
	MethodParam methodParam;
	Result result;
	struct PxInfoBase {
		int imgCoord_Y;
		int imgCoord_X;
		size_t nModelIdx;
	};
	// Internal pixel info LUT for all possible pixel indexes
	PxInfoBase* pxInfoLUT;
	// Descriptor Generator
	void DescriptorGenerator(cv::Mat inputFrame, cv::Point2d coor, Descriptor &tempWord);
	// Border line reconstruct
	cv::Mat BorderLineReconst();
	// Compensation with Motion Hist
	cv::Mat CompensationMotionHist();
	// Local color difference generator
	std::vector<std::vector<int>> LCDGenerator(cv::Mat inputFrame,cv::Point2d coor);
};
#endif