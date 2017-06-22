#pragma once

#ifndef __BackgroundSubtractorLCDP_H_INCLUDED
#define __BackgroundSubtractorLCDP_H_INCLUDED
#include <opencv2\opencv.hpp>
#include <vector>
#include <bitset>

class BackgroundSubtractorLCDP {
public:
	/*******CONSTRUCTOR*******/ // Checked
	BackgroundSubtractorLCDP(size_t inputWordsNo, bool inputPreSwitch,
		double inputDescColourDiffRatio, bool inputClsRGBDiffSwitch, double inputClsRGBThreshold, bool inputClsLCDPDiffSwitch,
		double inputClsLCDPThreshold, double inputClsUpLCDPThreshold, double inputClsLCDPMaxThreshold, int inputClsMatchThreshold,
		bool inputClsNbMatchSwitch, cv::Mat inputROI, cv::Size inputFrameSize,int inputFrameNo, bool inputUpRandomReplaceSwitch, bool inputUpRandomUpdateNbSwitch,
		bool inputUpFeedbackSwitch, float inputUpDynamicRateIncrease, float inputUpDynamicRateDecrease, float inputUpMinDynamicRate, float inputUpUpdateRateIncrease,
		float inputUpUpdateRateDecrease, float inputUpUpdateRateLowest, float inputUpUpdateRateHighest,
		float inputDarkMinIntensityRatio, float inputDarkMaxIntensityRatio, float inputDarkRDiffRatioMin, float inputDarkRDiffRatioMax,
		float inputDarkGDiffRatioMin, float inputDarkGDiffRatioMax,
		bool inputPostSwitch);

	/*******DESTRUCTOR*******/ // Checked
	~BackgroundSubtractorLCDP();

	std::string folderName;

	/*******INITIALIZATION*******/
	void Initialize(cv::Mat inputImg, cv::Mat inputROI);

	// Program processing
	void Process(cv::Mat inputImg, cv::Mat &outputImg);

	/*=====OTHERS Methods=====*/
	// Save parameters
	void SaveParameter(std::string versionFolderName, std::string saveFolderName);
protected:

	// PRE-DEFINED STRUCTURE
	// Descriptor structure
	struct DescriptorStruct {
		// Store the pixel's RGB values
		int rgb[3];
		// Store the number of frames that having same descriptor
		int frameCount;
		// Store the frame index of the first occurrences / initial of this descriptor
		int p;
		// Store the frame index of the last occurrences of this descriptor
		int q;
		// Store the pixel's LCDP values
		//int LCDP[16];
		std::bitset<96> LCDPColour[2];
		std::bitset<48> LCDPTexture[2];
		int lastF;
		float gray;
		int nowF;
	};

	// Pixel info structure
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

	// Pixel info structure
	struct PxInfo :PxInfoStruct {
		// 16 neighbor pixels' info
		PxInfoStruct nbIndex[48];
	};

	/*=====LOOK-UP TABLE=====*/
	// neighborhood's offset value
	const cv::Point nbOffset[48] = {
		/*cv::Point(0, -1),	cv::Point(1, 0),	cv::Point(0, 1),	cv::Point(-1, 0),
		cv::Point(-1, -1),  cv::Point(1, -1),	cv::Point(1, 1),	cv::Point(-1, 1),

		cv::Point(0, -2),   cv::Point(2, 0),    cv::Point(0, 2),    cv::Point(-2, 0),
		cv::Point(-2, -2),  cv::Point(2, -2),	cv::Point(2, 2),    cv::Point(-2, 2),

		cv::Point(-1, -3),	cv::Point(1, -3),	cv::Point(1, 3),	cv::Point(-1, 3),
		cv::Point(-3, -1),  cv::Point(3, -1),	cv::Point(3, 1),	cv::Point(-3, 1),
		cv::Point(-3, -3),  cv::Point(3, -3),   cv::Point(3, 3),	cv::Point(-3, 3),


		cv::Point(-2, 1),	cv::Point(2, 1),  	cv::Point(-1, 2),   cv::Point(1, 2),
		cv::Point(0, -3),	cv::Point(3, 0),	cv::Point(0, 3),	cv::Point(-3, 0),				  
		cv::Point(-2, -3),  cv::Point(2, -3),	cv::Point(2, 3),	cv::Point(-2, 3),
		cv::Point(-3, -2),  cv::Point(3, -2),	cv::Point(3, 2),	cv::Point(-3, 2),		
		cv::Point(-1, -2),  cv::Point(1, -2),	cv::Point(-2, -1),  cv::Point(2, -1)*/
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
	float LCDDiffLUTPtr[97][49];

	/*=====MODEL Parameters=====*/
	// Store the background's words and it's iterator
	DescriptorStruct * bgWordPtr, *bgWordPtrIter;
	// Store the current frame's words and it's iterator
	DescriptorStruct * currWordPtr, *currWordPtrIter;
	// Total number of words to represent a pixel
	const size_t WORDS_NO;
	// Frame index
	size_t frameIndex;

	/*=====PRE-PROCESS Parameters=====*/
	// Pre process switch
	const bool preSwitch;
	// Size of Gaussian filter
	cv::Size preGaussianSize;

	/*=====DESCRIPTOR Parameters=====*/
	// Size of neighborhood 3(3x3)/5(5x5)
	const int descNbSize;
	// Total number of neighborhood pixel 8(3x3)/16(5x5)
	const int descNbNo;
	// LCD color differences ratio
	const double descColourDiffRatio;
	// Total number of differences per descriptor
	const size_t descDiffNo;
	// Persistence's offset value;
	const size_t descOffsetValue;
	// Total length of descriptor pattern
	const int descPatternLength;

	/*=====CLASSIFIER Parameters=====*/
	// RGB detection switch
	const bool clsRGBDiffSwitch;
	// RGB differences threshold
	double clsRGBThreshold;
	// LCDP detection switch
	const bool clsLCDPDiffSwitch;
	// LCDP differences threshold
	double clsLCDPThreshold;
	// Up LCDP differences threshold
	double clsUpLCDPThreshold;
	// Maximum number of LCDP differences threshold
	const double clsLCDPMaxThreshold;
	// neighborhood matching switch
	const bool clsNbMatchSwitch;
	// Matched persistence threshold value
	cv::Mat clsPersistenceThreshold;
	// Minimum persistence threshold value
	float clsMinPersistenceThreshold;
	// Matching threshold
	const int clsMatchThreshold;

	/*=====POST-PROCESS Parameters=====*/
	// Size of median filter
	size_t postMedianFilterSize;
	// The compensation motion history threshold
	const float postCompensationThreshold;
	// Post processing switch
	const bool postSwitch;
	// Compensation Results
	cv::Mat postCompensationResult;

	/*=====FRAME Parameters=====*/
	// ROI frame
	const cv::Mat frameRoi;
	// Size of input frame
	const cv::Size frameSize;
	// Total number of input frame
	const int frameNo;
	// Size of input frame starting from 0
	const cv::Size frameSizeZero;
	// Total number of pixel of region of interest
	const size_t frameRoiTotalPixel;
	// Total number of pixel of input frame
	const size_t frameInitTotalPixel;

	/*=====UPDATE Parameters=====*/
	// Specifies the PX update spread range
	bool upUse3x3Spread;
	// Current learning rate caps
	float upLearningRateLowerCap;
	float upLearningRateUpperCap;
	// Random replace model switch
	const bool upRandomReplaceSwitch;
	// Random update neighborhood model switch
	const bool upRandomUpdateNbSwitch;
	// Feedback loop switch
	const bool upFeedbackSwitch;
	// Feedback V(x) Increment
	float upDynamicRateIncrease;
	// Feedback V(x) Decrement
	float upDynamicRateDecrease;
	// Minimum V(x) Value
	float upMinDynamicRate;
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
	cv::Mat resDistThreshold;
	// Per-pixel dynamic learning rate ('V(x)')
	cv::Mat resDynamicRate;
	// Per-pixel update rates('T(x)', which contains pixel - level 'sigma')
	cv::Mat resUpdateRate;

	// Minimum LCDP distance
	cv::Mat resMinLCDPDistance;
	// Minimum RGB distance
	cv::Mat resMinRGBDistance;
	// Total PERSISTENCE
	cv::Mat resTotalPersistence;
	// Current pixel distance
	cv::Mat resCurrPxDistance;

	// Current match result both RGB and LCDP
	cv::Mat resMatchResultBoth;
	// Current foreground mask
	cv::Mat resCurrFGMask;
	// Previous foreground mask
	cv::Mat resLastFGMask;
	// Previous raw foreground mask
	cv::Mat resLastRawFGMask;
	// t-1 foreground mask
	cv::Mat resT_1FGMask;
	// t-2 foreground mask
	cv::Mat resT_2FGMask;
	// Dark pixel result
	cv::Mat resDarkPixel;
	// Flooded holes foreground mask
	cv::Mat resFGMaskFloodedHoles;
	// Pre flooded holes foreground mask
	cv::Mat resFGMaskPreFlood;
	// Last image frame
	cv::Mat resLastImg;
	// Last Grayscale image frame
	cv::Mat resLastGrayImg;
	// Last Raw Blinking frame
	cv::Mat resLastRawBlink;
	// Current Raw Blinking frame
	cv::Mat resCurrRawBlink;
	// Blink frame
	cv::Mat resBlinkFrame;
	// Last foreground mask dilated
	cv::Mat resLastFGMaskDilated;
	// Last foreground mask dilated inverted
	cv::Mat resLastFGMaskDilatedInverted;

	// RGB Dark Pixel Parameter
	// Minimum Intensity Ratio
	float darkMinIntensityRatio;
	// Maximum Intensity Ratio
	float darkMaxIntensityRatio;
	// R-channel different ratio
	float darkRDiffRatioMin;
	float darkRDiffRatioMax;

	// G-channel different ratio
	float darkGDiffRatioMin;
	float darkGDiffRatioMax;
	
	/*=====METHODS=====*/
	/*=====DEFAULT methods=====*/
	// Refreshes all samples based on the last analyzed frame - checked
	void RefreshModel(float refreshFraction);

	/*=====DESCRIPTOR Methods=====*/
	// DescriptorStruct Generator-Generate pixels' descriptor (RGB+LCDP) - checked
	void DescriptorGenerator(cv::Mat inputFrame, PxInfo &pxInfoPtr, DescriptorStruct &wordPtr);
	// Generate LCD Descriptor - checked
	void LCDGenerator(cv::Mat inputFrame, PxInfo &pxInfoPtr, DescriptorStruct &wordPtr);
	// Calculate word persistence value
	void GetLocalWordPersistence(DescriptorStruct &wordPtr, size_t &currFrameIndex,
		size_t offsetValue, float &persistenceValue);

	/*=====LUT Methods=====*/
	// Generate neighborhood pixel offset value - checked
	void BackgroundSubtractorLCDP::GenerateNbOffset(PxInfo &pxInfoPtr);
	// Generate LCD differences Lookup table (0: 100% Same -> 1: 100% Different) - checked
	void GenerateLCDDiffLUT();

	/*=====MATCHING Methods=====*/
	// Descriptor matching (RETURN: LCDPResult-1:Not match, 0: Match)
	void BackgroundSubtractorLCDP::DescriptorMatching(DescriptorStruct &bgWord, DescriptorStruct &currWord
		,  size_t &descNeighNo,  double LCDPThreshold,  double upLCDPThreshold,  double RGBThreshold,
		float &LCDPDistance, float &RGBDistance, bool &matchResult, bool &matchResultBoth);
	
	// LCD Matching (RETURN-1:Not match, 0: Match)
	void LCDPMatching(DescriptorStruct &bgWord, DescriptorStruct &currWord,
		 size_t &descNeighNo,  double &LCDPThreshold, float &minDistance, bool &matchResult);
	// RGB Matching (RETURN-1:Not match, 0: Match)
	void RGBMatching(DescriptorStruct &bgWord, DescriptorStruct &currWord,
		double &RGBThreshold, float &minDistance, bool &matchResult);
	// RGB Dark Pixel (RETURN-1:Not Dark Pixel, 0: Dark Pixel) Checked May 14
	void BackgroundSubtractorLCDP::RGBDarkPixel(DescriptorStruct &bgWord, DescriptorStruct &currWord, bool &result);
	// Dark Pixel generator (RETURN-1: Not dark pixel, 0: dark pixel)
	void DarkPixelGenerator(cv::Mat &inputGrayImg, cv::Mat &inputRGBImg,
		 cv::Mat &lastGrayImg, cv::Mat &lastRGBImg, cv::Mat &darkPixel);

	/*=====POST-PROCESSING Methods=====*/
	// Compensation with Motion History - checked
	cv::Mat CompensationMotionHist(cv::Mat T_1FGMask, cv::Mat T_2FGMask, cv::Mat currFGMask, float postCompensationThreshold);
	// Contour filling the empty holes - checked
	cv::Mat ContourFill(cv::Mat inputImg);
	cv::Mat BorderLineReconst(cv::Mat inputMask);

	/*=====DEBUG=====*/
	cv::Point debPxLocation;
	size_t debPxPointer;
};
#endif