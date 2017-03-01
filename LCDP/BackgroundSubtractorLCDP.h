#pragma once

#ifndef __BackgroundSubtractorLCDP_H_INCLUDED
#define __BackgroundSubtractorLCDP_H_INCLUDED
#include <opencv2\opencv.hpp>
#include <vector>

class BackgroundSubtractorLCDP {
public:
	/*******CONSTRUCTOR*******/ // Checked
	BackgroundSubtractorLCDP(cv::Size inputFrameSize, cv::Mat inputROI, size_t inputWordsNo,
		bool inputRGBDiffSwitch, double inputRGBThreshold, bool inputRGBBrightPxSwitch, bool inputLCDPDiffSwitch,
		double inputLCDPThreshold, double inputLCDPMaxThreshold, bool inputAndOrSwitch, bool inputNbMatchSwitch,
		bool inputRandomReplaceSwitch, bool inputRandomUpdateNbSwitch, bool inputFeedbackSwitch);

	/*******DESTRUCTOR*******/ // Checked
	~BackgroundSubtractorLCDP();

	/*******INITIALIZATION*******/
	void Initialize(const cv::Mat inputImg, const cv::Mat inputROI);

	// Program processing
	void Process(const cv::Mat inputImg, cv::Mat &outputImg);

	/*=====OTHERS Methods=====*/
	// Save parameters
	void SaveParameter(std::string folderName);
	
	/*=====DEBUG=====*/
	
	bool debugSwitch = false;
	void DebugPxLocation(int x, int y);
protected:

	// PRE-DEFINED STRUCTURE
	// Descriptor structure - Checked
	struct DescriptorStruct {
		// Store the pixel's RGB values
		unsigned rgb[3];
		// Store the number of frames that having same descriptor
		unsigned frameCount;
		// Store the frame index of the first occurences/initial of this descriptor
		unsigned p;
		// Store the frame index of the last occurences of this descriptor
		unsigned q;
		// Store the pixel's LCDP values
		unsigned LCDP[16];
	};

	// Pixel info structure - Checked
	struct PxInfoStruct {
		// Coordinate Y value for current pixel
		int coor_y;
		// Coordinate X value for current pixel
		int coor_x;
		// Data index for pointer pixel
		size_t dataIndex;
		// Data index for BGR data
		size_t bgrDataIndex;
		// Start index for pixel's model
		size_t startModelIndex;
	};

	// Pixel info structure - Checked
	struct PxInfo:PxInfoStruct {
		// 16 neighbour pixels' info
		PxInfoStruct nbIndex[16];
	};

	/*=====LOOK-UP TABLE=====*/
	// Neighbourhood's offset value
	const cv::Point nbOffset[16] = {
		cv::Point(-1, -1), cv::Point(0, -1), cv::Point(1, -1),
		cv::Point(-1, 0),  cv::Point(1, 0),
		cv::Point(-1, 1),  cv::Point(0, 1),  cv::Point(1, 1),
		cv::Point(-2, -2), cv::Point(0, -2), cv::Point(2, -2),
		cv::Point(-2, 0),  cv::Point(2, 0),
		cv::Point(-2, 2),  cv::Point(0, 2),  cv::Point(2, 2) };
	// Internal pixel info LUT for all possible pixel indexes
	PxInfo * pxInfoLUTPtr;
	// LCD differences LUT
	float * LCDDiffLUTPtr;

	/*=====MODEL Parameters=====*/
	// Store the background's words and it's iterator
	DescriptorStruct * bgWordPtr, *bgWordPtrIter;
	// Store the currect frame's words and it's iterator
	DescriptorStruct * currWordPtr, *currWordPtrIter;
	// Total number of words to represent a pixel
	const size_t WORDS_NO;
	// Frame index
	size_t frameIndex;

	/*=====PRE-PROCESS Parameters=====*/
	// Size of gaussian filter
	cv::Size preGaussianSize;

	/*=====DESCRIPTOR Parameters=====*/
	// Size of neighbourhood 3(3x3)/5(5x5)
	cv::Mat descNbSize;
	// Total number of neighbourhood pixel 8(3x3)/16(5x5)
	cv::Mat descNbNo;
	// LCD colour differences ratio
	cv::Mat descColorDiffRatio;
	// Total number of differences per descriptor
	const size_t descDiffNo;
	// Persistence's offset value;
	const size_t descOffsetValue;

	/*=====CLASSIFIER Parameters=====*/
	// RGB detection switch
	const bool clsRGBDiffSwitch;
	// RGB differences threshold
	const double clsRGBThreshold;
	// RGB bright pixel switch
	const bool clsRGBBrightPxSwitch;
	// LCDP detection switch
	const bool clsLCDPDiffSwitch;
	// LCDP differences threshold
	double clsLCDPThreshold;
	// Maximum number of LCDP differences threshold
	const double clsLCDPMaxThreshold;
	// LCDP detection AND (true) OR (false) switch
	const bool clsAndOrSwitch;
	// Neighbourhood matching switch
	const bool clsNbMatchSwitch;
	// Total number of neighbour 0(0)/8(3x3)/16(5x5)
	cv::Mat clsNbNo;
	// Matched persistence threshold value
	cv::Mat clsPersistenceThreshold;
	// Minimum persistence threhsold value
	float clsMinPersistenceThreshold;
	
	/*=====POST-PROCESS Parameters=====*/
	// Size of median filter
	size_t postMedianFilterSize;
	// The compensation motion history threshold
	cv::Mat postCompensationThreshold;

	/*=====FRAME Parameters=====*/
	// ROI frame
	const cv::Mat frameRoi;
	// Size of input frame
	const cv::Size frameSize;
	// Size of input frame starting from 0
	const cv::Size frameSizeZero;
	// Total number of pixel of region of interest
	const size_t frameRoiTotalPixel;
	// Total number of pixel of input frame
	const size_t frameInitTotalPixel;

	/*=====UPDATE Parameters=====*/
	// Specifies the px update spread range
	bool upUse3x3Spread;
	// Current learning rate caps
	float upLearningRateLowerCap;
	float upLearningRateUpperCap;
	// Random replace model switch
	const bool upRandomReplaceSwitch;
	// Random update neighbourhood model switch
	const bool upRandomUpdateNbSwitch;
	// Total number of neighbour undergo neighbourhood updates 8(3X3)/16(5X5)
	cv::Mat upNbNo;
	// Feedback loop switch
	const bool upFeedbackSwitch;
	
	/*=====RESULTS=====*/
	// Per-pixel distance thresholds ('R(x)', but used as a relative value to determine both 
	// intensity and descriptor variation thresholds)
	cv::Mat resDistThreshold;
	// Per-pixel update rates('T(x)', which contains pixel - level 'sigmas')
	cv::Mat resUpdateRate;
	// Minimum match distance
	cv::Mat resMinMatchDistance;
	// Current foreground mask
	cv::Mat resCurrFGMask;
	// Previous foreground mask
	cv::Mat resLastFGMask;
	// The foreground mask generated by the method at [t-1] (without post-proc, used for blinking px detection)
	cv::Mat resLastRawFGMask;
	// t-1 foreground mask
	cv::Mat resT_1FGMask;
	// t-2 foreground mask
	cv::Mat resT_2FGMask;
	// Flooded holes foreground mask
	cv::Mat resFGMaskFloodedHoles;
	// Pre flooded holes foreground mask
	cv::Mat resFGMaskPreFlood;
	// Current pixel distance
	cv::Mat resCurrPxDistance;

	/*=====METHODS=====*/
	/*=====DEFAULT methods=====*/
	// Refreshes all samples based on the last analyzed frame - checked
	void RefreshModel(const float refreshFraction, const bool forceUpdateSwitch = false);

	/*=====DESCRIPTOR Methods=====*/
	// DescriptorStruct Generator-Generate pixels' descriptor (RGB+LCDP) - checked
	void DescriptorGenerator(const cv::Mat inputFrame, const PxInfo *pxInfoPtr, DescriptorStruct *wordPtr);
	// Generate LCD Descriptor - checked
	void LCDGenerator(const cv::Mat inputFrame, const PxInfo *pxInfoPtr, DescriptorStruct *wordPtr);
	// Calculate word persistence value - checked
	float GetLocalWordWeight(const DescriptorStruct* wordPtr, const size_t currFrameIndex, const size_t offsetValue);

	/*=====LUT Methods=====*/
	// Generate neighbourhood pixel offset value - checked
	void GenerateNbOffset(PxInfo * pxInfoPtr);
	// Generate LCD differences Lookup table (0: 100% Same -> 1: 100% Different) - checked
	void GenerateLCDDiffLUT();

	/*=====MATCHING Methods=====*/
	// Descriptor matching (RETURN-True:Not match, False: Match) - checked
	bool DescriptorMatching(DescriptorStruct *pxWordPtr, DescriptorStruct *currPxWordPtr, const size_t pxPointer,
		const double LCDPThreshold, const double RGBThreshold, float &tempMatchDistance);
	// LCD Matching (RETURN-True:Not match, False: Match) - checked
	bool LCDPMatching(DescriptorStruct *bgWord, DescriptorStruct *currWord, const size_t pxPointer, const double LCDPThreshold, float &minDistance);
	// RGB Matching (RETURN-True:Not match, False: Match) - checked
	bool RGBMatching(const unsigned bgRGB[], const unsigned currRGB[], const double RGBThreshold, float &minDistance);
	// Bright Pixel (RETURN-True:Not match, False: Match) - checked
	bool BrightRGBMatching(const unsigned bgRGB[], const unsigned currRGB[], const double BrightThreshold);

	/*=====POST-PROCESSING Methods=====*/
	// Compensation with Motion Hist - checked
	cv::Mat CompensationMotionHist(const cv::Mat T_1FGMask, const cv::Mat T_2FGMask, const cv::Mat currFGMask, const cv::Mat postCompensationThreshold);
	// Contour filling the empty holes - checked
	cv::Mat ContourFill(const cv::Mat inputImg);
	
	/*=====DEBUG=====*/
	cv::Point debPxLocation;
	size_t debPxPointer;
};
#endif