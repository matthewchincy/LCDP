#pragma once

#ifndef __BackgroundSubtractorLCDP_H_INCLUDED
#define __BackgroundSubtractorLCDP_H_INCLUDED
#include <opencv2\opencv.hpp>
#include <vector>

class BackgroundSubtractorLCDP {
public:
	/*******CONSTRUCTOR*******/ // Checked
	BackgroundSubtractorLCDP(std::string folderName, size_t inputWordsNo, bool inputPreSwitch,
		double inputDescColourDiffRatioInit, bool inputDescRatioCalculationMethod, bool inputRGBDiffSwitch,
		double inputRGBThreshold,bool inputLCDPDiffSwitch, double inputLCDPThreshold, double inputUpLCDPThreshold,
		double inputLCDPMaxThreshold, bool inputMatchingMethod, int inputMatchThreshold, bool inputNbMatchSwitch,
		cv::Mat inputROI, cv::Size inputFrameSize,
		bool inputRandomReplaceSwitch, bool inputRandomUpdateNbSwitch, bool inputFeedbackSwitch,
		float inputDynamicRateIncrease, float inputDynamicRateDecrease, float inputUpdateRateIncrease, float inputUpdateRateDecrease,
		float inputUpdateRateLowest, float inputUpdateRateHighest, bool inputPostSwitch);

	/*******DESTRUCTOR*******/ // Checked
	~BackgroundSubtractorLCDP();

	/*******INITIALIZATION*******/
	void Initialize(const cv::Mat inputImg, const cv::Mat inputROI);

	// Program processing
	void Process(const cv::Mat inputImg, cv::Mat &outputImg);

	/*=====OTHERS Methods=====*/
	// Save parameters
	void SaveParameter(std::string filename, std::string folderName);

	/*=====DEBUG=====*/
	bool debugSwitch = false;
	void DebugPxLocation(int x, int y);
protected:
	const std::string folderName;
	// PRE-DEFINED STRUCTURE
	// Descriptor structure - Checked
	struct DescriptorStruct {
		// Store the pixel's RGB values
		int rgb[3];
		// Store the number of frames that having same descriptor
		int frameCount;
		// Store the frame index of the first occurences/initial of this descriptor
		int p;
		// Store the frame index of the last occurences of this descriptor
		int q;
		// Store the pixel's LCDP values
		int LCDP[16];
	};

	// Pixel info structure - Checked
	struct PxInfoStruct {
		// Coordinate Y value for current pixel
		int coor_y;
		// Coordinate X value for current pixel
		int coor_x;
		// Data index for current pixel's pointer
		size_t dataIndex;
		// Data index for current pixel's BGR pointer
		size_t bgrDataIndex;
		// Model index for current pixel
		size_t modelIndex;
	};

	// Pixel info structure - Checked
	struct PxInfo :PxInfoStruct {
		// 16 neighbour pixels' info
		PxInfoStruct nbIndex[48];
	};

	/*=====LOOK-UP TABLE=====*/
	// Neighbourhood's offset value
	const cv::Point nbOffset[48] = {
		cv::Point(0, -1),	cv::Point(1, 0),	cv::Point(0, 1),	cv::Point(-1, 0),
		cv::Point(-1, -1),  cv::Point(1, -1),	cv::Point(1, 1),	cv::Point(-1, 1),
		cv::Point(-2, -2), cv::Point(0, -2), cv::Point(2, -2),
		cv::Point(-2, 0),  cv::Point(2, 0),
		cv::Point(-2, 2),  cv::Point(0, 2),  cv::Point(2, 2),
		cv::Point(-1, -2),  cv::Point(1, -2),
		cv::Point(-2, -1),  cv::Point(2, -1),
		cv::Point(-2, 1),  cv::Point(2, 1),
		cv::Point(-1, 2),  cv::Point(1, 2),
		cv::Point(-3, -3),  cv::Point(0, -3),cv::Point(3, -3),
		cv::Point(-3, 0),  cv::Point(3, 0),
		cv::Point(-3, 3),  cv::Point(0, 3),cv::Point(3, 3),
		cv::Point(-2, -3),  cv::Point(-1, -3),cv::Point(1, -3),cv::Point(2, -3),
		cv::Point(-3, -2),  cv::Point(3, -2),
		cv::Point(-3, -1),  cv::Point(3, -1),
		cv::Point(-3, 2),  cv::Point(3, 2),
		cv::Point(-3, 1),  cv::Point(3, 1),
		cv::Point(-2, 3),  cv::Point(-1, 3),cv::Point(1, 3),cv::Point(2, 3)
	};
	// Internal pixel info LUT for all possible pixel indexes
	PxInfo * pxInfoLUTPtr;
	// LCD differences 
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
	// Pre process switch
	const bool preSwitch;
	// Size of gaussian filter
	cv::Size preGaussianSize;	

	/*=====DESCRIPTOR Parameters=====*/
	// Size of neighbourhood 3(3x3)/5(5x5)
	cv::Mat descNbSize;
	// Total number of neighbourhood pixel 8(3x3)/16(5x5)
	cv::Mat descNbNo;
	// LCD colour differences ratio
	const double descColourDiffRatio;
	// LCD colour differences ratio
	cv::Mat descColorDiffRatio;
	// Total number of differences per descriptor
	const size_t descDiffNo;
	// Persistence's offset value;
	const size_t descOffsetValue;
	// Ratio calculation method
	const bool descRatioCalculationMethod;

	/*=====CLASSIFIER Parameters=====*/
	// RGB detection switch
	const bool clsRGBDiffSwitch;
	// RGB differences threshold
	const double clsRGBThreshold;
	// LCDP detection switch
	const bool clsLCDPDiffSwitch;
	// LCDP differences threshold
	double clsLCDPThreshold;
	// Up LCDP differences threshold
	double clsUpLCDPThreshold;
	// Maximum number of LCDP differences threshold
	const double clsLCDPMaxThreshold;
	// Neighbourhood matching switch
	const bool clsNbMatchSwitch;
	// Total number of neighbour 0(0)/8(3x3)/16(5x5)
	cv::Mat clsNbNo;
	// Matched persistence threshold value
	cv::Mat clsPersistenceThreshold;
	// Minimum persistence threhsold value
	float clsMinPersistenceThreshold;
	// Classify method
	const bool clsMatchingMethod;
	// Matching threshold
	const int clsMatchThreshold;

	/*=====POST-PROCESS Parameters=====*/
	// Size of median filter
	size_t postMedianFilterSize;
	// The compensation motion history threshold
	cv::Mat postCompensationThreshold;
	// Post processing switch
	const bool postSwitch;

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
	// Feedback V(x) Increment
	float upDynamicRateIncrease;
	// Feedback V(x) Decrement
	float upDynamicRateDecrease;
	// Feedback T(x) Increment
	float upUpdateRateIncrease;
	// Feedback T(x) Decrement
	float upUpdateRateDecrease;
	// Feedback T(x) Lowest
	float upUpdateRateLowerCap;
	// Feedback T(x) Highest
	float upUpdateRateUpperCap;

	/*=====RESULTS=====*/
	// Per-pixel distance thresholds ('R(x)', but used as a relative value to determine both 
	// intensity and descriptor variation thresholds)
	cv::Mat resLCDPDistThreshold;
	cv::Mat resRGBDistThreshold;
	// Per-pixel dynamic learning rate ('V(x)')
	cv::Mat resLCDPDynamicRate;
	cv::Mat resRGBDynamicRate;
	// Per-pixel update rates('T(x)', which contains pixel - level 'sigmas')
	cv::Mat resLCDPUpdateRate;
	cv::Mat resRGBUpdateRate;

	// Minimum LCDP distance
	cv::Mat resMinLCDPDistance;
	// Minimum RGB distance
	cv::Mat resMinRGBDistance;
	// Current LCDP pixel distance
	cv::Mat resCurrLCDPPxDistance;
	// Current LCDP pixel distance
	cv::Mat resCurrRGBPxDistance;

	// Current foreground mask
	cv::Mat resCurrFGMask;
	// Current LCDP foreground mask
	cv::Mat resCurrLCDPFGMask;
	// Current Up LCDP foreground mask
	cv::Mat resCurrUpLCDPFGMask;
	// Current RGB foreground mask
	cv::Mat resCurrRGBFGMask;
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
	// Dark Pixel
	cv::Mat resDarkPixel;
	// Last image frame
	cv::Mat resLastImg;
	// Last Grayscale image frame
	cv::Mat resLastGrayImg;

	/*=====METHODS=====*/
	/*=====DEFAULT methods=====*/
	// Refreshes all samples based on the last analyzed frame - checked
	void RefreshModel(const float refreshFraction, const bool forceUpdateSwitch = false);

	/*=====DESCRIPTOR Methods=====*/
	// DescriptorStruct Generator-Generate pixels' descriptor (RGB+LCDP) - checked
	void DescriptorGenerator(const cv::Mat inputFrame, const PxInfo &pxInfoPtr, DescriptorStruct &wordPtr);
	// Generate LCD Descriptor - checked
	void LCDGenerator(const cv::Mat inputFrame, const PxInfo &pxInfoPtr, DescriptorStruct &wordPtr);
	// Calculate word persistence value
	void GetLocalWordPersistence(DescriptorStruct &wordPtr, const size_t &currFrameIndex,
		const size_t offsetValue, float &persistenceValue);

	/*=====LUT Methods=====*/
	// Generate neighbourhood pixel offset value - checked
	void BackgroundSubtractorLCDP::GenerateNbOffset(PxInfo &pxInfoPtr);
	// Generate LCD differences Lookup table (0: 100% Same -> 1: 100% Different) - checked
	void GenerateLCDDiffLUT();

	/*=====MATCHING Methods=====*/
	// Descriptor matching (RETURN: LCDPResult-1:Not match, 0: Match)
	void DescriptorMatching(DescriptorStruct &bgWord, DescriptorStruct &currWord
		, const size_t &descNeighNo, const double LCDPThreshold, const double upLCDPThreshold, const double RGBThreshold,
		float &LCDPDistance, float &upLCDPDistance, float &RGBDistance, bool &LCDPResult, bool &upLCDPResult, bool &RGBResult);
	// LCD Matching (RETURN-1:Not match, 0: Match)
	void LCDPMatching(DescriptorStruct &bgWord, DescriptorStruct &currWord,
		const size_t &descNeighNo, const double &LCDPThreshold, float &minDistance, bool &result);
	// RGB Matching (RETURN-1:Not match, 0: Match)
	void RGBMatching(DescriptorStruct &bgWord, DescriptorStruct &currWord,
		const double &RGBThreshold, float &minDistance, bool &result);
	// Dark Pixel generator (RETURN-1: Not dark pixel, 0: dark pixel)
	void DarkPixelGenerator(const cv::Mat &inputGrayImg, const cv::Mat &inputRGBImg,
		const cv::Mat &lastGrayImg, const cv::Mat &lastRGBImg, cv::Mat &darkPixel);

	/*=====POST-PROCESSING Methods=====*/
	// Compensation with Motion Hist - checked
	cv::Mat CompensationMotionHist(const cv::Mat T_1FGMask, const cv::Mat T_2FGMask, const cv::Mat currFGMask, const cv::Mat postCompensationThreshold);
	// Contour filling the empty holes - checked
	cv::Mat ContourFill(const cv::Mat inputImg);
	cv::Mat BorderLineReconst(const cv::Mat inputMask);

	/*=====DEBUG=====*/
	cv::Point debPxLocation;
	size_t debPxPointer;
};
#endif