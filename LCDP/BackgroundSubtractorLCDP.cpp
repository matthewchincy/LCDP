#include "BackgroundSubtractorLCDP.h"
#include "RandUtils.h"
#include <iostream>
#include <bitset>
#include <fstream>
#include <vector>
#include <time.h>

// Parameters used to define 'unstable' regions, based on segm noise/bg dynamics and local distance threshold values
#define UNSTABLE_REG_RATIO_MIN  (0.10f)
// Parameter used to average the min distance
#define MIN_DISTANCE_ALPHA (0.01f)
// Local define used to specify the default frame size (320x240 = QVGA)
#define DEFAULT_FRAME_SIZE cv::Size(320,240)
// Pre-processing Gaussian size
#define PRE_DEFAULT_GAUSSIAN_SIZE cv::Size(9,9)

/*******CONSTRUCTOR*******/ // Checked
BackgroundSubtractorLCDP::BackgroundSubtractorLCDP(size_t inputWordsNo, bool inputPreSwitch,
	double inputDescColourDiffRatio, bool inputClsRGBDiffSwitch, double inputClsRGBThreshold, bool inputClsLCDPDiffSwitch,
	double inputClsLCDPThreshold, double inputClsUpLCDPThreshold, double inputClsLCDPMaxThreshold, int inputClsMatchThreshold,
	bool inputClsNbMatchSwitch, cv::Mat inputROI, cv::Size inputFrameSize, int inputFrameNo, bool inputUpRandomReplaceSwitch, bool inputUpRandomUpdateNbSwitch,
	bool inputUpFeedbackSwitch, float inputUpDynamicRateIncrease, float inputUpDynamicRateDecrease, float inputUpMinDynamicRate, float inputUpUpdateRateIncrease,
	float inputUpUpdateRateDecrease, float inputUpUpdateRateLowest, float inputUpUpdateRateHighest,
	float inputDarkMinIntensityRatio, float inputDarkMaxIntensityRatio, float inputDarkRDiffRatioMin, float inputDarkRDiffRatioMax,
	float inputDarkGDiffRatioMin, float inputDarkGDiffRatioMax,	bool inputPostSwitch) :
	/*=====LOOK-UP TABLE=====*/
	// Internal pixel info LUT for all possible pixel indexes
	pxInfoLUTPtr(nullptr),
	//// LCD differences LUT
	//LCDDiffLUTPtr(nullptr),

	/*=====MODEL Parameters=====*/
	// Store the background's words and it's iterator
	bgWordPtr(nullptr),
	bgWordPtrIter(nullptr),
	// Store the current frame's words and it's iterator
	currWordPtr(nullptr),
	currWordPtrIter(nullptr),
	// Total number of words to represent a pixel
	WORDS_NO(inputWordsNo),
	// Frame index
	frameIndex(1),

	/*=====PRE-PROCESS Parameters=====*/
	// Pre processing switch
	preSwitch(inputPreSwitch),
	// Size of Gaussian filter
	preGaussianSize(PRE_DEFAULT_GAUSSIAN_SIZE),

	/*=====DESCRIPTOR Parameters=====*/
	// Size of neighborhood 3(3x3)/5(5x5)
	descNbSize(5),
	// Total number of neighborhood pixel 8(3x3)/16(5x5)
	descNbNo(16),
	// LCD color differences ratio
	descColourDiffRatio(inputDescColourDiffRatio),
	// Total number of differences per descriptor
	descDiffNo(9),
	// Persistence's offset value
	descOffsetValue(1000),
	// Total length of descriptor pattern
	descPatternLength(std::pow(2, (9 * 2))),

	/*=====CLASSIFIER Parameters=====*/
	// RGB detection switch
	clsRGBDiffSwitch(inputClsRGBDiffSwitch),
	// RGB differences threshold
	clsRGBThreshold(inputClsRGBThreshold),
	// LCDP detection switch
	clsLCDPDiffSwitch(inputClsLCDPDiffSwitch),
	// LCDP differences threshold
	clsLCDPThreshold(inputClsLCDPThreshold),
	// UP LCDP differences threshold
	clsUpLCDPThreshold(inputClsUpLCDPThreshold),
	// Maximum number of LCDP differences threshold
	clsLCDPMaxThreshold(inputClsLCDPMaxThreshold),
	// neighborhood matching switch
	clsNbMatchSwitch(inputClsNbMatchSwitch),
	// Matching threshold
	clsMatchThreshold(inputClsMatchThreshold),

	/*=====FRAME Parameters=====*/
	// ROI frame
	frameRoi(inputROI),
	// Size of input frame
	frameSize(inputFrameSize),
	// Total number of input frame
	frameNo(inputFrameNo),
	// Size of input frame starting from 0
	frameSizeZero(cv::Size(inputFrameSize.width - 1, inputFrameSize.height - 1)),
	// Total number of pixel of region of interest
	frameRoiTotalPixel(cv::countNonZero(inputROI)),
	// Total number of pixel of input frame
	frameInitTotalPixel(inputFrameSize.area()),

	/*=====UPDATE Parameters=====*/
	// Random replace model switch
	upRandomReplaceSwitch(inputUpRandomReplaceSwitch),
	// Random update neighborhood model switch
	upRandomUpdateNbSwitch(inputUpRandomUpdateNbSwitch),
	// Feedback loop switch
	upFeedbackSwitch(inputUpFeedbackSwitch),
	// Feedback V(x) Increment
	upDynamicRateIncrease(inputUpDynamicRateIncrease),
	// Feedback V(x) Decrement
	upDynamicRateDecrease(inputUpDynamicRateDecrease),
	// Feedback V(x) Minimum
	upMinDynamicRate(inputUpMinDynamicRate),
	// Feedback T(x) Increment
	upUpdateRateIncrease(inputUpUpdateRateIncrease),
	// Feedback T(x) Decrement
	upUpdateRateDecrease(inputUpUpdateRateDecrease),
	// Feedback T(x) Lowest
	upUpdateRateLowerCap(inputUpUpdateRateLowest),
	// Feedback T(x) Highest
	upUpdateRateUpperCap(inputUpUpdateRateHighest),

	/*====RGB Dark Pixel Parameter=====*/
	// Minimum Intensity Ratio
	darkMinIntensityRatio(inputDarkMinIntensityRatio),
	// Maximum Intensity Ratio
	darkMaxIntensityRatio(inputDarkMaxIntensityRatio),
	// R-channel different ratio
	darkRDiffRatioMin(inputDarkRDiffRatioMin),
	darkRDiffRatioMax(inputDarkRDiffRatioMax),
	// G-channel different ratio
	darkGDiffRatioMin(inputDarkGDiffRatioMin),
	darkGDiffRatioMax(inputDarkGDiffRatioMax),

	/*====POST Parameters=====*/
	// Post processing switch
	postSwitch(inputPostSwitch),
	// The compensation motion history threshold
	postCompensationThreshold(0.7f)
{
	CV_Assert(WORDS_NO > 0);
}

/*******DESTRUCTOR*******/ // Checked
BackgroundSubtractorLCDP::~BackgroundSubtractorLCDP() {
	delete[] pxInfoLUTPtr;

	/*for (int i = 0; i < 3; ++i) {
		delete[] LCDDiffLUTPtr[i];
	}
	delete[] LCDDiffLUTPtr;*/
	delete[] bgWordPtr;
	delete[] currWordPtr;
}

/*******INITIALIZATION*******/ // Checked
void BackgroundSubtractorLCDP::Initialize(cv::Mat inputFrame, cv::Mat inputROI)
{
	/*=====LOOK-UP TABLE=====*/
	// Internal pixel info LUT for all possible pixel indexes
	pxInfoLUTPtr = new PxInfo[frameInitTotalPixel];
	memset(pxInfoLUTPtr, 0, sizeof(PxInfo)*frameInitTotalPixel);
	//// LCD differences LUT
	//LCDDiffLUTPtr = new float*[3];
	//for (int i = 0; i < 3; i++) {
	//	LCDDiffLUTPtr[i] = new float[6];
	//}
	//memset(LCDDiffLUTPtr, 0, sizeof(float)*3*9);
	// Generate LCD difference Lookup table
	GenerateLCDDiffLUT();

	/*=====MODEL Parameters=====*/
	// Store the background's word and it's iterator
	bgWordPtr = new DescriptorStruct[frameInitTotalPixel*WORDS_NO];
	memset(bgWordPtr, 0, sizeof(DescriptorStruct)*frameInitTotalPixel*WORDS_NO);
	bgWordPtrIter = bgWordPtr;
	// Store the current frame's word and it's iterator
	currWordPtr = new DescriptorStruct[frameInitTotalPixel];
	memset(currWordPtr, 0, sizeof(DescriptorStruct)*frameInitTotalPixel);
	currWordPtrIter = currWordPtr;

	/*=====CLASSIFIER Parameters=====*/
	// Minimum persistence threshold value 
	clsMinPersistenceThreshold = (1.0f / descOffsetValue);
	// Matched persistence value threshold
	clsPersistenceThreshold.create(frameSize, CV_32FC1);
	clsPersistenceThreshold = cv::Scalar(clsMinPersistenceThreshold);

	/*=====POST-PROCESS Parameters=====*/
	// Size of median filter
	postMedianFilterSize = 9;

	if ((frameRoiTotalPixel >= (frameInitTotalPixel / 2)) && (frameInitTotalPixel >= DEFAULT_FRAME_SIZE.area())) {
		/*=====POST-PROCESS Parameters=====*/
		double tempMedianFilterSize = std::min(double(14), floor((frameRoiTotalPixel / DEFAULT_FRAME_SIZE.area()) + 0.5) + postMedianFilterSize);
		if ((int(tempMedianFilterSize) % 2) == 0)
			tempMedianFilterSize = tempMedianFilterSize - 1;
		// Current kernel size for median blur post-processing filtering
		postMedianFilterSize = tempMedianFilterSize;
		// Specifies the PX update spread range
		upUse3x3Spread = !(frameInitTotalPixel > (DEFAULT_FRAME_SIZE.area() * 2));
		// Current learning rate caps
		upLearningRateLowerCap = upUpdateRateLowerCap;
		upLearningRateUpperCap = upUpdateRateUpperCap;
	}
	else {
		/*=====POST-PROCESS Parameters=====*/
		// Current kernel size for median blur post-processing filtering
		postMedianFilterSize = postMedianFilterSize;
		// Specifies the PX update spread range
		upUse3x3Spread = true;
		// Current learning rate caps
		upLearningRateLowerCap = upUpdateRateLowerCap * 2;
		upLearningRateUpperCap = upUpdateRateUpperCap * 2;
	}

	/*=====RESULTS=====*/
	// Per-pixel distance thresholds ('R(x)', but used as a relative value to determine both 
	// intensity and descriptor variation thresholds)
	resDistThreshold.create(frameSize, CV_32FC1);
	resDistThreshold = cv::Scalar(1.0f);
	// Per-pixel dynamic learning rate ('V(x)')
	resDynamicRate.create(frameSize, CV_32FC1);
	resDynamicRate = cv::Scalar(10.0f);
	// Per-pixel update rates('T(x)')
	resUpdateRate.create(frameSize, CV_32FC1);
	resUpdateRate = cv::Scalar(upLearningRateLowerCap);
	// Current pixel distance
	resCurrPxDistance.create(frameSize, CV_32FC1);
	resCurrPxDistance = cv::Scalar(0.0f);
	// Minimum LCDP distance
	resMinLCDPDistance.create(frameSize, CV_32FC1);
	resMinLCDPDistance = cv::Scalar(1.0f);
	// Minimum RGB distance
	resMinRGBDistance.create(frameSize, CV_32FC1);
	resMinRGBDistance = cv::Scalar(1.0f);
	// Total PERSISTENCE
	resTotalPersistence.create(frameSize, CV_32FC1);
	resTotalPersistence = cv::Scalar(1.0f);
	// Current foreground mask
	resCurrFGMask.create(frameSize, CV_8UC1);
	resCurrFGMask = cv::Scalar_<uchar>::all(0);
	// Current match result both RGB and LCDP
	resMatchResultBoth.create(frameSize, CV_8UC1);
	resMatchResultBoth = cv::Scalar_<uchar>::all(0);

	// Dark Pixel
	resDarkPixel.create(frameSize, CV_8UC1);
	resDarkPixel = cv::Scalar_<uchar>::all(0);
	// Previous foreground mask
	resLastFGMask.create(frameSize, CV_8UC1);
	resLastFGMask = cv::Scalar_<uchar>::all(0);
	// Previous raw foreground mask
	resLastRawFGMask.create(frameSize, CV_8UC1);
	resLastRawFGMask = cv::Scalar_<uchar>::all(0);
	// Last Raw Blinking frame
	resLastRawBlink.create(frameSize, CV_8UC1);
	resLastRawBlink = cv::Scalar_<uchar>::all(0);
	// Current Raw Blinking frame
	resCurrRawBlink.create(frameSize, CV_8UC1);
	resCurrRawBlink = cv::Scalar_<uchar>::all(0);
	// Blink frame
	resBlinkFrame.create(frameSize, CV_8UC1);
	resBlinkFrame = cv::Scalar_<uchar>::all(0);
	// t-1 foreground mask
	resT_1FGMask.create(frameSize, CV_8UC1);
	resT_1FGMask = cv::Scalar_<uchar>::all(0);
	// t-2 foreground mask
	resT_2FGMask.create(frameSize, CV_8UC1);
	resT_2FGMask = cv::Scalar_<uchar>::all(0);
	// Flooded holes foreground mask
	//resFGMaskFloodedHoles.create(frameSize, CV_8UC1);
	resFGMaskFloodedHoles.create(frameSize, CV_8UC1);
	resFGMaskFloodedHoles = cv::Scalar_<uchar>::all(0);
	// Pre flooded holes foreground mask
	resFGMaskPreFlood.create(frameSize, CV_8UC1);
	resFGMaskPreFlood = cv::Scalar_<uchar>::all(0);
	// Last foreground mask dilated
	resLastFGMaskDilated.create(frameSize, CV_8UC1);
	resLastFGMaskDilated = cv::Scalar_<uchar>::all(0);
	// Last foreground mask dilated inverted
	resLastFGMaskDilatedInverted.create(frameSize, CV_8UC1);
	resLastFGMaskDilatedInverted = cv::Scalar_<uchar>::all(0);

	// Last frame image
	inputFrame.copyTo(resLastImg);
	cv::cvtColor(inputFrame, resLastGrayImg, CV_RGB2GRAY);
	// Pixel pointer index
	size_t pxPointer = 0;
	// PRE PROCESSING
	cv::GaussianBlur(inputFrame, inputFrame, preGaussianSize, 0, 0);

	for (size_t rowIndex = 0; rowIndex < frameSize.height; rowIndex++) {
		for (size_t colIndex = 0; colIndex < frameSize.width; colIndex++) {
			// Coordinate Y value
			pxInfoLUTPtr[pxPointer].coor_y = (int)rowIndex;
			// Coordinate X value
			pxInfoLUTPtr[pxPointer].coor_x = (int)colIndex;
			// Data index for current pixel's pointer
			pxInfoLUTPtr[pxPointer].dataIndex = pxPointer;
			// Data index for current pixel's BGR pointer
			pxInfoLUTPtr[pxPointer].bgrDataIndex = pxPointer * 3;
			// Model index for current pixel
			pxInfoLUTPtr[pxPointer].modelIndex = pxPointer*WORDS_NO;
			/*=====LUT Methods=====*/
			// Generate neighborhood pixel offset value
			GenerateNbOffset(pxInfoLUTPtr[pxPointer]);
			// Descriptor Generator-Generate pixels' descriptor (RGB+LCDP)
			DescriptorGenerator(inputFrame, pxInfoLUTPtr[pxPointer], currWordPtr[pxPointer]);
			pxPointer++;
		}
	}

	// Refresh model
	RefreshModel(1.0f);
}

/*=====METHODS=====*/
/*=====DEFAULT methods=====*/
// Refreshes all samples based on the last analyzed frame - checked
void BackgroundSubtractorLCDP::RefreshModel(float refreshFraction)
{
	srand(time(NULL));
	const size_t noSampleBeRefresh = refreshFraction < 1.0f ? (size_t)(refreshFraction*WORDS_NO) : WORDS_NO;
	const size_t refreshStartPos = refreshFraction < 1.0f ? rand() % WORDS_NO : 0;
	DescriptorStruct * bgWord = nullptr;
	DescriptorStruct * currWord = nullptr;
	for (size_t pxPointer = 0; pxPointer < frameInitTotalPixel; pxPointer++) {
		if (frameRoi.data[pxPointer]) {
			// Start index of the model of the current pixel
			size_t modelIndex = pxInfoLUTPtr[pxPointer].modelIndex;
			for (size_t currModelIndex = refreshStartPos; currModelIndex < refreshStartPos + noSampleBeRefresh; ++currModelIndex) {

				cv::Point sampleCoor;
				getRandSamplePosition_7x7(sampleCoor, cv::Point(pxInfoLUTPtr[pxPointer].coor_x, pxInfoLUTPtr[pxPointer].coor_y), 0, frameSize);
				size_t samplePxIndex = (frameSize.width*sampleCoor.y) + sampleCoor.x;
				currWord = (currWordPtr + samplePxIndex);

				bgWord = (bgWordPtr + modelIndex + currModelIndex);
				for (size_t channel = 0; channel < 3; channel++) {
					(*bgWord).rgb[channel] = (*currWord).rgb[channel];
				}
				for (size_t channel = 0; channel < 2; channel++) {
					(*bgWord).LCDPColour[channel] = std::bitset<96>((*currWord).LCDPColour[channel]);
					(*bgWord).LCDPTexture[channel] = std::bitset<48>((*currWord).LCDPTexture[channel]);
				}
				(*bgWord).frameCount = 1;
				(*bgWord).p = frameIndex;
				(*bgWord).q = frameIndex;
			}
		}
	}
}

// Program processing
void BackgroundSubtractorLCDP::Process(cv::Mat inputImg, cv::Mat &outputImg)
{
	srand(time(NULL));
	cv::Mat inputGrayImg;
	cv::cvtColor(inputImg, inputGrayImg, CV_RGB2GRAY);
	// Update average image
	resLastImg = (inputImg + (resLastImg*(frameIndex - 1))) / frameIndex;
	// PRE PROCESSING
	if (preSwitch) {
		cv::GaussianBlur(inputImg, inputImg, preGaussianSize, 0, 0);
	}
	bool bootstrapping = frameIndex <= 500;
	// DETECTION PROCESS
	// Generate a map to indicate dark pixel (255: Not dark pixel, 0: Dark pixel)
	DarkPixelGenerator(inputGrayImg, inputImg, resLastGrayImg, resLastImg, resDarkPixel);
	// BG Word pointer
	DescriptorStruct * bgWord = nullptr;
	// NB Word pointer
	DescriptorStruct * nbBgWord = nullptr;
	// Current bg word's persistence	
	float currWordPersistence;
	for (size_t pxPointer = 0; pxPointer < frameInitTotalPixel; ++pxPointer) {
		if (frameRoi.data[pxPointer]) {
			// Descriptor Generator-Generate pixels' descriptor (RGB+LCDP)
			DescriptorGenerator(inputImg, pxInfoLUTPtr[pxPointer], currWordPtr[pxPointer]);
			// Current distance threshold ('R(x)')r
			float * currDistThreshold = (float*)(resDistThreshold.data + (pxPointer * 4));
			// Current dynamic rate ('V(x)')
			float * currDynamicRate = (float*)(resDynamicRate.data + (pxPointer * 4));
			// Current pixel's update rate ('T(x)')
			float * currUpdateRate = (float*)(resUpdateRate.data + (pxPointer * 4));
			const size_t updateRate = ceil(*currUpdateRate);
			// Model index for current pixel
			const size_t currModelIndex = pxInfoLUTPtr[pxPointer].modelIndex;

			// Current dark pixel result
			uchar * currDarkPixel = (resDarkPixel.data + pxPointer);
			// LCDP differences threshold
			const double currLCDPThreshold = std::min(clsLCDPMaxThreshold, std::max(clsLCDPThreshold, (std::pow(2, double((*currDistThreshold))) / 512)));
			// Up LCDP differences threshold
			const double currUpLCDPThreshold = clsUpLCDPThreshold;
			// RGB differences threshold
			const double currRGBThreshold = std::max(clsRGBThreshold, floor(clsRGBThreshold*(*currDistThreshold)));
			// Persistence threshold
			float * currPersistenceThreshold = (float*)(clsPersistenceThreshold.data + (pxPointer * 4));
			// Current pixel's foreground mask
			uchar * currFGMask = (resCurrFGMask.data + pxPointer);
			// Current pixel's foreground mask
			uchar * matchResultBoth = (resMatchResultBoth.data + pxPointer);
			// Total number of neighborhood pixel 8(3x3)/16(5x5)
			size_t currDescNeighNo = descNbNo;
			// Current pixel's descriptor
			DescriptorStruct currWord = currWordPtr[pxPointer];

			// Current pixel's min LCDP distance
			float * minLCDPDistance = (float*)(resMinLCDPDistance.data + (pxPointer * 4));
			// Current pixel's min RGB distance
			float * minRGBDistance = (float*)(resMinRGBDistance.data + (pxPointer * 4));
			// Current pixel's total persistence
			float * totalPersistence = (float*)(resTotalPersistence.data + (pxPointer * 4));
			// Current pixel distance
			float * currPxDistance = (float*)(resCurrPxDistance.data + (pxPointer * 4));

			// Last word's persistence
			float currLastWordPersistence = FLT_MAX;
			// Current pixel's background word index
			int currLocalWordIdx = 0;

			// Number of potential matched model
			int clsPotentialMatch = 0;
			while (currLocalWordIdx < WORDS_NO && (clsPotentialMatch < clsMatchThreshold)) {
				// Current bg word
				bgWord = (bgWordPtr + currModelIndex + currLocalWordIdx);
				GetLocalWordPersistence(*bgWord, frameIndex, descOffsetValue, currWordPersistence);
				float tempLCDPDistance = 1.0f;
				float tempRGBDistance = 1.0f;
				bool matchResult = false;
				bool matchBoth = false;
				// False:Match true:Not match
				DescriptorMatching(*bgWord, currWord, currDescNeighNo, currLCDPThreshold, currUpLCDPThreshold, currRGBThreshold,
					tempLCDPDistance, tempRGBDistance, matchResult, matchBoth);
				// Both BG
				if (!matchResult) {
					if (matchBoth) {
						(*matchResultBoth) = 255;
					}
					(*bgWord).frameCount += 1;
					(*bgWord).q = frameIndex;
					clsPotentialMatch++;
					// Update MIN LCDP distance
					(*minLCDPDistance) = std::min(tempLCDPDistance, (*minLCDPDistance));
					// Update MIN RGB distance
					(*minRGBDistance) = std::min(tempRGBDistance, (*minRGBDistance));
					// Update MIN PERSISTENCE distance
					(*totalPersistence) = (*totalPersistence) + currWordPersistence;
					/*(*totalPersistence) = std::min((*currPersistenceThreshold), (*totalPersistence) + currWordPersistence);*/
					// BG
					if (std::rand() % (updateRate) == 0) {
						if (tempLCDPDistance < (currLCDPThreshold / 2)) {
							for (size_t channel = 0; channel < 3; channel++) {
								(*bgWord).rgb[channel] = currWord.rgb[channel];
							}
							for (size_t channel = 0; channel < 2; channel++) {
								(*bgWord).LCDPColour[channel] = std::bitset<96>(currWord.LCDPColour[channel]);
								(*bgWord).LCDPTexture[channel] = std::bitset<48>(currWord.LCDPTexture[channel]);
							}
							/*nbBgWord = (bgWordPtr + currModelIndex + WORDS_NO - 1);
							for (size_t channel = 0; channel < 3; channel++) {
							(*nbBgWord).rgb[channel] = currWord.rgb[channel];
							}
							for (size_t channel = 0; channel < descNbNo; channel++) {
							(*nbBgWord).LCDP[channel] = currWord.LCDP[channel];
							}
							(*nbBgWord).frameCount = 1;
							(*nbBgWord).p = frameIndex;
							(*nbBgWord).q = frameIndex;*/
						}
					}
				}
				// Sort background model based on persistence
				if (currWordPersistence > currLastWordPersistence) {
					std::swap(bgWordPtr[currModelIndex + currLocalWordIdx], bgWordPtr[currModelIndex + currLocalWordIdx - 1]);
				}
				else {
					currLastWordPersistence = currWordPersistence;
				}
				++currLocalWordIdx;
			}

			// Sorting remaining models
			while (currLocalWordIdx < WORDS_NO) {
				bgWord = (bgWordPtr + currModelIndex + currLocalWordIdx);
				GetLocalWordPersistence(*bgWord, frameIndex, descOffsetValue, currWordPersistence);
				if (currWordPersistence > currLastWordPersistence) {
					std::swap(bgWordPtr[currModelIndex + currLocalWordIdx], bgWordPtr[currModelIndex + currLocalWordIdx - 1]);
				}
				else {
					currLastWordPersistence = currWordPersistence;
				}
				++currLocalWordIdx;
			}
			// Successful classified as BG Pixels
			if (clsPotentialMatch >= clsMatchThreshold) {
				(*currFGMask) = 0;
				(*currDarkPixel) = 0;
				// Replace Model from NB to the last model of bg word

				cv::Point sampleCoor;
				if (!upUse3x3Spread) {
					getRandSamplePosition_5x5(sampleCoor, cv::Point(pxInfoLUTPtr[pxPointer].coor_x, pxInfoLUTPtr[pxPointer].coor_y), 0, frameSize);
				}
				else {
					getRandSamplePosition_3x3(sampleCoor, cv::Point(pxInfoLUTPtr[pxPointer].coor_x, pxInfoLUTPtr[pxPointer].coor_y), 0, frameSize);
				}
				int randNum = rand() % WORDS_NO;
				const size_t samplePxIndex = frameSize.width*sampleCoor.y + sampleCoor.x;
				// Start index of the model of the current pixel
				const size_t startNBModelIndex = pxInfoLUTPtr[samplePxIndex].modelIndex;
				// Current pixel's update rate ('T(x)')
				const size_t nbUpdateRate = ceil(*((float*)(resUpdateRate.data + (samplePxIndex * 4))));
				if (std::rand() % (nbUpdateRate * 2) == 0) {
					nbBgWord = (bgWordPtr + startNBModelIndex + randNum);
					for (size_t channel = 0; channel < 3; channel++) {
						(*nbBgWord).rgb[channel] = currWord.rgb[channel];
					}
					for (size_t channel = 0; channel < 2; channel++) {
						(*nbBgWord).LCDPColour[channel] = std::bitset<96>(currWord.LCDPColour[channel]);
						(*nbBgWord).LCDPTexture[channel] = std::bitset<48>(currWord.LCDPTexture[channel]);
					}
					(*nbBgWord).frameCount = 1;
					(*nbBgWord).p = frameIndex;
					(*nbBgWord).q = frameIndex;
				}
				//(*currDynamicRate) = std::max(upMinDynamicRate, (*currDynamicRate) - upDynamicRateDecrease);
			}
			// Classified as FG Pixels
			else {
				(*currFGMask) = 255;
				size_t nbMatchNo = 0;
				if (clsNbMatchSwitch) {
					// Compare with neighbor's model
					// neighbor matching size (Max: 5x5)				
					nbMatchNo = std::max(16.0f, std::floor((((*currDistThreshold) / 9) * 48)));

					for (size_t nbIndex = 0; nbIndex < nbMatchNo; nbIndex++) {
						// neighbor pixel pointer
						size_t nbPxPointer = pxInfoLUTPtr[pxPointer].nbIndex[nbIndex].dataIndex;
						// neighbor pixel's model index
						const size_t nbModelIndex = pxInfoLUTPtr[pxPointer].nbIndex[nbIndex].modelIndex;
						// Current neighbor pixel's matching threshold
						int clsNBMatchThreshold = clsMatchThreshold;
						// Number of potential matched model
						int clsNBPotentialMatch = 0;
						// Current neighbor distance threshold
						float * nbDistThreshold = (float*)(resDistThreshold.data + (nbPxPointer * 4));
						// neighbor LCD descriptor threshold
						const double nbLCDPThreshold = std::min(clsLCDPMaxThreshold, std::max(clsLCDPThreshold, (std::pow(2, double((*nbDistThreshold))) / 512)));
						// neighbor Up LCD descriptor threshold
						const double nbUpLCDPThreshold = clsUpLCDPThreshold;
						// neighbor RGB descriptor threshold
						const double nbRGBThreshold = std::max(clsRGBThreshold, floor(clsRGBThreshold*(*nbDistThreshold)));

						// Current Match Distance
						float currMatchDistance = 1.0f;
						int currMatchModel = 0;

						float nbLastWordPersistence = FLT_MAX;
						size_t nbLocalWordIdx = 0;
						while ((nbLocalWordIdx < WORDS_NO) && (clsNBPotentialMatch < clsNBMatchThreshold)) {

							bgWord = (bgWordPtr + nbModelIndex + nbLocalWordIdx);
							GetLocalWordPersistence(*bgWord, frameIndex, descOffsetValue, currWordPersistence);
							float tempLCDPDistance = 1.0f;
							float tempRGBDistance = 1.0f;

							bool matchResult = false;
							bool matchBoth = false;

							// False:Match true:Not match
							DescriptorMatching(*bgWord, currWord, currDescNeighNo, nbLCDPThreshold, nbUpLCDPThreshold, nbRGBThreshold,
								tempLCDPDistance, tempRGBDistance, matchResult, matchBoth);

							if (!matchResult) {
								if (std::rand() % (updateRate) == 0) {
									if (tempLCDPDistance < (nbLCDPThreshold / 2)) {
										for (size_t channel = 0; channel < 3; channel++) {
											(*bgWord).rgb[channel] = currWord.rgb[channel];
										}
										for (size_t channel = 0; channel < 2; channel++) {
											(*bgWord).LCDPColour[channel] = std::bitset<96>(currWord.LCDPColour[channel]);
											(*bgWord).LCDPTexture[channel] = std::bitset<48>(currWord.LCDPTexture[channel]);
										}
									}
								}
								if (matchBoth) {
									(*matchResultBoth) = 255;
								}
								(*bgWord).frameCount += 1;
								(*bgWord).q = frameIndex;
								clsNBPotentialMatch++;
								if (currMatchDistance > ((tempLCDPDistance + tempRGBDistance) / 2.0f)) {
									currMatchModel = nbLocalWordIdx;
									currMatchDistance = ((tempLCDPDistance + tempRGBDistance) / 2.0f);
								}
							}

							// Update position of model in background model
							if (currWordPersistence > nbLastWordPersistence) {
								std::swap(bgWordPtr[nbModelIndex + nbLocalWordIdx], bgWordPtr[nbModelIndex + nbLocalWordIdx - 1]);
							}
							else
								nbLastWordPersistence = currWordPersistence;
							++nbLocalWordIdx;
						}
						// Sorting remaining models
						while (nbLocalWordIdx < WORDS_NO) {
							bgWord = (bgWordPtr + nbModelIndex + nbLocalWordIdx);
							GetLocalWordPersistence(*bgWord, frameIndex, descOffsetValue, currWordPersistence);
							if (currWordPersistence > nbLastWordPersistence) {
								std::swap(bgWordPtr[nbModelIndex + nbLocalWordIdx], bgWordPtr[nbModelIndex + nbLocalWordIdx - 1]);
							}
							else {
								nbLastWordPersistence = currWordPersistence;
							}
							++nbLocalWordIdx;
						}
						if (clsNBPotentialMatch >= clsNBMatchThreshold) {
							(*currFGMask) = 0;
							(*currDarkPixel) = 0;
							break;
						}
					}
				}
			}
			(*totalPersistence) = (*totalPersistence) > (*currPersistenceThreshold) ? (*currPersistenceThreshold) : (*totalPersistence);
			
			//// Update minimum distance
			if (*currFGMask) {
				//FG
				float currNormalizedMinDist = 0.0f;
				if (clsLCDPDiffSwitch) {
					currNormalizedMinDist = std::max((((*currPersistenceThreshold) - (*totalPersistence)) / (*currPersistenceThreshold)), std::max(*minRGBDistance, *minLCDPDistance));
				}
				else {
					currNormalizedMinDist = std::max((((*currPersistenceThreshold) - (*totalPersistence)) / (*currPersistenceThreshold)), *minRGBDistance);
				}
				
				(*currPxDistance) = ((1 - MIN_DISTANCE_ALPHA)*(*currPxDistance)) + (MIN_DISTANCE_ALPHA*currNormalizedMinDist);
			}
			else {
				//BG
				float currNormalizedMinDist = 0.0f;
				if (clsLCDPDiffSwitch) {
					currNormalizedMinDist = std::max(*minRGBDistance, *minLCDPDistance);
				}
				else {
					currNormalizedMinDist = *minRGBDistance;
				}
				(*currPxDistance) = ((1 - MIN_DISTANCE_ALPHA)*(*currPxDistance)) + (MIN_DISTANCE_ALPHA*currNormalizedMinDist);
			}

			if (upFeedbackSwitch) {
				// Last foreground mask
				uchar * lastFGMask = (resLastFGMask.data + pxPointer);

				bool check1 = ((*currPxDistance)< UNSTABLE_REG_RATIO_MIN) && (*currFGMask);
				//bool check1 = (((*currPxDistance)< UNSTABLE_REG_RATIO_MIN) && (*currFGMask)) || !(*currFGMask);

				//bool check2 = check1 && ((*currUpdateRate) < upLearningRateUpperCap);
				if (check1) {
					float valueIncrease = upUpdateRateIncrease / ((*currPxDistance)*(*currDynamicRate));
					/*if (isinf(valueIncrease)) {
					valueIncrease = 150;
					}*/
					(*currUpdateRate) = std::min(upLearningRateUpperCap, (*currUpdateRate) + valueIncrease);
				}
				//check2 = !check1 && ((*currUpdateRate) >= upLearningRateLowerCap);
				if (!check1) {
					float valueDecrease = ((upUpdateRateDecrease*(*currDynamicRate)) / (*currPxDistance));
					/*if (isinf(valueDecrease)) {
					valueDecrease = 10;
					}*/

					(*currUpdateRate) = std::max(upLearningRateLowerCap, (*currUpdateRate) - valueDecrease);
				}

				if (((*currPxDistance)>UNSTABLE_REG_RATIO_MIN) && resBlinkFrame.data[pxPointer])
					(*currDynamicRate) += bootstrapping ? upDynamicRateIncrease * 2 : upDynamicRateIncrease;
				else
					(*currDynamicRate) = std::max((*currDynamicRate) - upDynamicRateDecrease*((bootstrapping) ? 2 : resLastFGMask.data[pxPointer] ? 0.5f : 1), upDynamicRateDecrease);

				check1 = (*currDistThreshold) < (std::pow((1.0f + ((*currPxDistance) * 2)), 2));
				if (check1) {
					(*currDistThreshold) += (*currDynamicRate)*0.01f;
				}
				else {
					(*currDistThreshold) = std::max(1.0f, (*currDistThreshold) - (0.01f / (*currDynamicRate)));
				}
				// Top BG word
				bgWord = (bgWordPtr + currModelIndex);
				GetLocalWordPersistence(*bgWord, frameIndex, descOffsetValue, currWordPersistence);
				(*currPersistenceThreshold) = currWordPersistence / ((*currDistThreshold) * 2);
			}
		}
	}
	// POST PROCESSING
	if (postSwitch) {
		cv::bitwise_xor(resCurrFGMask, resLastRawFGMask, resCurrRawBlink);
		cv::bitwise_or(resCurrRawBlink, resLastRawBlink, resBlinkFrame);
		resCurrRawBlink.copyTo(resLastRawBlink);
		cv::Mat element = cv::getStructuringElement(0, cv::Size(5, 5));
		cv::Mat tempCurrFGMask;
		resCurrFGMask.copyTo(tempCurrFGMask);

		postCompensationResult = CompensationMotionHist(resT_1FGMask, resT_2FGMask, resCurrFGMask, postCompensationThreshold);
		cv::Mat grad_x, grad_y, grad;
		cv::Mat abs_grad_x, abs_grad_y;
		int ddepth = CV_16S;
		int scale = 1;
		int delta = 0;
		/// Gradient X
		cv::Sobel(inputGrayImg, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT);
		/// Gradient Y
		cv::Sobel(inputGrayImg, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT);
		convertScaleAbs(grad_x, abs_grad_x);
		convertScaleAbs(grad_y, abs_grad_y);
		addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

		cv::Mat gradientResult, gradientResult2;

		cv::inRange(grad, cv::Scalar(75), cv::Scalar(110), gradientResult);
		//cv::bitwise_not(resDarkPixel, gradientResult2);
		//cv::bitwise_and(gradientResult, gradientResult2, gradientResult);

		cv::dilate(gradientResult, gradientResult, cv::Mat(), cv::Point(-1, -1), 3);
		cv::bitwise_not(gradientResult, gradientResult);
		cv::bitwise_and(grad, gradientResult, gradientResult);
		cv::dilate(gradientResult, gradientResult, cv::Mat(), cv::Point(-1, -1), 1);
		cv::inRange(grad, cv::Scalar(75), cv::Scalar(150), gradientResult2);
		cv::dilate(gradientResult2, gradientResult2, cv::Mat(), cv::Point(-1, -1), 4);
		//inputGrayImg.copyTo(gradientResult);
		//cv::threshold(gradientResult, gradientResult, 80, 255, CV_THRESH_BINARY);
		//cv::morphologyEx(gradientResult, gradientResult, cv::MORPH_GRADIENT, element);
		//cv::Mat invDarkPixel;
		//resDarkPixel.copyTo(invDarkPixel);
		//cv::bitwise_not(invDarkPixel, invDarkPixel);
		//cv::erode(invDarkPixel, invDarkPixel, cv::Mat(), cv::Point(-1, -1), 3);
		//cv::dilate(invDarkPixel, invDarkPixel, cv::Mat(), cv::Point(-1, -1), 3);

		//cv::bitwise_and(invDarkPixel, gradientResult, gradientResult);
		//cv::bitwise_and(invDarkPixel, gradientResult2, gradientResult2);
		//cv::dilate(gradientResult, gradientResult, cv::Mat(), cv::Point(-1, -1), 5);
		//cv::dilate(gradientResult2, gradientResult2, cv::Mat(), cv::Point(-1, -1), 5);
		//cv::bitwise_not(gradientResult, gradientResult);
		cv::bitwise_not(gradientResult2, gradientResult2);

		// ADD NEW
		cv::erode(resCurrFGMask, resCurrFGMask, cv::Mat(), cv::Point(-1, -1), 1);
		cv::dilate(resCurrFGMask, resCurrFGMask, cv::Mat(), cv::Point(-1, -1), 1);
		//cv::bitwise_and(resCurrFGMask, gradientResult, tempCurrFGMask);
		cv::bitwise_and(resCurrFGMask, gradientResult2, tempCurrFGMask);
		cv::erode(resDarkPixel, resDarkPixel, cv::Mat(), cv::Point(-1, -1), 3);
		cv::dilate(resDarkPixel, resDarkPixel, cv::Mat(), cv::Point(-1, -1), 2);
		cv::medianBlur(resDarkPixel, resDarkPixel, 3);
		//cv::bitwise_or(tempCurrFGMask, resDarkPixel, gradientResult);
		cv::bitwise_or(tempCurrFGMask, resDarkPixel, gradientResult2);
		//cv::morphologyEx(gradientResult, gradientResult, cv::MORPH_CLOSE, element);
		cv::morphologyEx(gradientResult2, gradientResult2, cv::MORPH_CLOSE, element);
		//cv::dilate(gradientResult, gradientResult, cv::Mat(), cv::Point(-1, -1), 3);
		cv::Mat reconstructLine = BorderLineReconst(gradientResult2);
		cv::bitwise_or(tempCurrFGMask, reconstructLine, resFGMaskPreFlood);
		//cv::bitwise_or(resDarkPixel, resFGMaskPreFlood, resFGMaskPreFlood);
		cv::dilate(resFGMaskPreFlood, resFGMaskPreFlood, cv::Mat(), cv::Point(-1, -1), 3);
		cv::morphologyEx(resFGMaskPreFlood, resFGMaskPreFlood, cv::MORPH_CLOSE, element);
		resFGMaskFloodedHoles = ContourFill(resFGMaskPreFlood);
		cv::bitwise_not(resMatchResultBoth, resMatchResultBoth);
		cv::bitwise_and(resFGMaskFloodedHoles, resMatchResultBoth, resFGMaskFloodedHoles);
		//cv::dilate(reconstructLine, reconstructLine, cv::Mat(), cv::Point(-1, -1), 3);
		//cv::bitwise_not(reconstructLine, reconstructLine);
		//cv::bitwise_and(reconstructLine, resFGMaskFloodedHoles, resFGMaskFloodedHoles);
		cv::bitwise_or(resCurrFGMask, resFGMaskFloodedHoles, resCurrFGMask);
		cv::bitwise_or(resCurrFGMask, postCompensationResult, resLastFGMask);


		//cv::bitwise_and(gradientResult, resLastFGMask, resLastFGMask);
		//cv::morphologyEx(resLastFGMask, resLastFGMask, cv::MORPH_CLOSE, element);
		cv::medianBlur(resLastFGMask, resLastFGMask, postMedianFilterSize);
		//cv::dilate(resLastFGMask, resLastFGMask, cv::Mat(), cv::Point(-1, -1), 2);

		cv::dilate(resLastFGMask, resLastFGMaskDilated, cv::Mat(), cv::Point(-1, -1), 3);
		cv::bitwise_and(resBlinkFrame, resLastFGMaskDilatedInverted, resBlinkFrame);
		cv::bitwise_not(resLastFGMaskDilated, resLastFGMaskDilatedInverted);
		cv::bitwise_and(resBlinkFrame, resLastFGMaskDilatedInverted, resBlinkFrame);
		cv::medianBlur(resLastFGMask, resLastFGMask, postMedianFilterSize);
		cv::morphologyEx(resLastFGMask, resLastFGMask, cv::MORPH_OPEN, element);
		cv::morphologyEx(resLastFGMask, resLastFGMask, cv::MORPH_CLOSE, element);
		resLastFGMask = ContourFill(resLastFGMask);
		resLastFGMask.copyTo(resCurrFGMask);
		resT_1FGMask.copyTo(resT_2FGMask);
		resLastFGMask.copyTo(resT_1FGMask);
	}
	resCurrFGMask.copyTo(outputImg);
	// Frame Index
	frameIndex++;
	// Reset minimum matching distance
	resMinLCDPDistance = cv::Scalar(1.0f);
	resMinRGBDistance = cv::Scalar(1.0f);
	resTotalPersistence = cv::Scalar(0.0f);
	resCurrFGMask = cv::Scalar_<uchar>::all(0);
	resMatchResultBoth = cv::Scalar_<uchar>::all(0);
	resLastGrayImg = (inputGrayImg + (resLastGrayImg*(frameIndex - 1))) / frameIndex;
}

/*=====DESCRIPTOR Methods=====*/
// Descriptor Generator-Generate pixels' descriptor (RGB+LCDP)
void BackgroundSubtractorLCDP::DescriptorGenerator(cv::Mat inputFrame,  PxInfo &pxInfoPtr,
	DescriptorStruct &wordPtr)
{
	// Debug usage
	float gray = 0.0f;
	wordPtr.frameCount = 1;
	for (int channel = 0; channel < 3; channel++) {
		wordPtr.rgb[channel] = inputFrame.data[pxInfoPtr.bgrDataIndex + channel];
		gray += inputFrame.data[pxInfoPtr.bgrDataIndex + channel];
	}
	gray = gray / 3.0f;
	wordPtr.p = frameIndex;
	wordPtr.q = frameIndex;
	wordPtr.gray = gray;
	wordPtr.lastF = frameIndex;
	wordPtr.nowF = frameIndex;
	LCDGenerator(inputFrame, pxInfoPtr, wordPtr);
}
// Generate LCD Descriptor
void BackgroundSubtractorLCDP::LCDGenerator(cv::Mat inputFrame, PxInfo &pxInfoPtr, DescriptorStruct &wordPtr)
{
	// Current pixel RGB intensity
	int B_CURR = inputFrame.data[pxInfoPtr.bgrDataIndex];
	int G_CURR = inputFrame.data[pxInfoPtr.bgrDataIndex + 1];
	int R_CURR = inputFrame.data[pxInfoPtr.bgrDataIndex + 2];

	// Define neighbor differences variables
	int R_NB, G_NB, B_NB;
	double ratioBNB_GCURR_MIN, ratioBNB_RCURR_MIN, ratioGNB_BCURR_MIN, ratioGNB_RCURR_MIN, ratioRNB_BCURR_MIN, ratioRNB_GCURR_MIN,
		ratioBNB_GCURR_MAX, ratioBNB_RCURR_MAX, ratioGNB_BCURR_MAX, ratioGNB_RCURR_MAX, ratioRNB_BCURR_MAX, ratioRNB_GCURR_MAX,
		ratioBNB_BCURR_MIN, ratioGNB_GCURR_MIN, ratioRNB_RCURR_MIN, ratioBNB_BCURR_MAX, ratioGNB_GCURR_MAX, ratioRNB_RCURR_MAX;

	
	// New ratio calculation method
	double ratioBNB_GCURR = std::max(3.0, abs(double(descColourDiffRatio*(B_CURR - G_CURR))));
	double ratioBNB_RCURR = std::max(3.0, abs(double(descColourDiffRatio*(B_CURR - R_CURR))));
	double ratioGNB_BCURR = std::max(3.0, abs(double(descColourDiffRatio*(G_CURR - B_CURR))));
	double ratioGNB_RCURR = std::max(3.0, abs(double(descColourDiffRatio*(G_CURR - R_CURR))));
	double ratioRNB_BCURR = std::max(3.0, abs(double(descColourDiffRatio*(R_CURR - B_CURR))));
	double ratioRNB_GCURR = std::max(3.0, abs(double(descColourDiffRatio*(R_CURR - G_CURR))));

	ratioBNB_GCURR_MIN = std::max(-255.0, std::min((B_CURR - G_CURR) - ratioBNB_GCURR, (B_CURR - G_CURR) + ratioBNB_GCURR));
	ratioBNB_RCURR_MIN = std::max(-255.0, std::min((B_CURR - R_CURR) - ratioBNB_RCURR, (B_CURR - R_CURR) + ratioBNB_RCURR));
	ratioGNB_BCURR_MIN = std::max(-255.0, std::min((G_CURR - B_CURR) - ratioGNB_BCURR, (G_CURR - B_CURR) + ratioGNB_BCURR));
	ratioGNB_RCURR_MIN = std::max(-255.0, std::min((G_CURR - R_CURR) - ratioGNB_RCURR, (G_CURR - R_CURR) + ratioGNB_RCURR));
	ratioRNB_BCURR_MIN = std::max(-255.0, std::min((R_CURR - B_CURR) - ratioRNB_BCURR, (R_CURR - B_CURR) + ratioRNB_BCURR));
	ratioRNB_GCURR_MIN = std::max(-255.0, std::min((R_CURR - G_CURR) - ratioRNB_GCURR, (R_CURR - G_CURR) + ratioRNB_GCURR));

	ratioBNB_GCURR_MAX = std::min(255.0, std::max((B_CURR - G_CURR) - ratioBNB_GCURR, (B_CURR - G_CURR) + ratioBNB_GCURR));
	ratioBNB_RCURR_MAX = std::min(255.0, std::max((B_CURR - R_CURR) - ratioBNB_RCURR, (B_CURR - R_CURR) + ratioBNB_RCURR));
	ratioGNB_BCURR_MAX = std::min(255.0, std::max((G_CURR - B_CURR) - ratioGNB_BCURR, (G_CURR - B_CURR) + ratioGNB_BCURR));
	ratioGNB_RCURR_MAX = std::min(255.0, std::max((G_CURR - R_CURR) - ratioGNB_RCURR, (G_CURR - R_CURR) + ratioGNB_RCURR));
	ratioRNB_BCURR_MAX = std::min(255.0, std::max((R_CURR - B_CURR) - ratioRNB_BCURR, (R_CURR - B_CURR) + ratioRNB_BCURR));
	ratioRNB_GCURR_MAX = std::min(255.0, std::max((R_CURR - G_CURR) - ratioRNB_GCURR, (R_CURR - G_CURR) + ratioRNB_GCURR));

	double ratioBNB_BCURR = std::max(3.0, double(descColourDiffRatio*B_CURR));
	double ratioGNB_GCURR = std::max(3.0, double(descColourDiffRatio*G_CURR));
	double ratioRNB_RCURR = std::max(3.0, double(descColourDiffRatio*R_CURR));

	ratioBNB_BCURR_MIN = std::max(-255.0, -ratioBNB_BCURR);
	ratioGNB_GCURR_MIN = std::max(-255.0, -ratioGNB_GCURR);
	ratioRNB_RCURR_MIN = std::max(-255.0, -ratioRNB_RCURR);

	ratioBNB_BCURR_MAX = std::min(255.0, ratioBNB_BCURR);
	ratioGNB_GCURR_MAX = std::min(255.0, ratioGNB_GCURR);
	ratioRNB_RCURR_MAX = std::min(255.0, ratioRNB_RCURR);

	double tempBNB_GCURR, tempBNB_RCURR, tempGNB_BCURR, tempGNB_RCURR, tempRNB_BCURR, tempRNB_GCURR, tempBNB_BCURR,
		tempGNB_GCURR, tempRNB_RCURR;
	wordPtr.LCDPTexture[0].reset();
	wordPtr.LCDPTexture[1].reset();
	wordPtr.LCDPColour[0].reset();
	wordPtr.LCDPColour[1].reset();
	for (int nbPixelIndex = 0; nbPixelIndex < descNbNo; nbPixelIndex++) {
		// Obtain neighborhood pixel's value
		B_NB = inputFrame.data[pxInfoPtr.nbIndex[nbPixelIndex].bgrDataIndex];
		G_NB = inputFrame.data[pxInfoPtr.nbIndex[nbPixelIndex].bgrDataIndex + 1];
		R_NB = inputFrame.data[pxInfoPtr.nbIndex[nbPixelIndex].bgrDataIndex + 2];
		int tempResult = 0;

		// R_NB - R_CURR
		tempRNB_RCURR = R_NB - R_CURR;
		//tempResult += ((tempRNB_RCURR > ratioRNB_RCURR_MAX) ? 65536 : ((tempRNB_RCURR < (ratioRNB_RCURR_MIN)) ? 196608 : 0));
		if (tempRNB_RCURR > ratioRNB_RCURR_MAX) {
			wordPtr.LCDPTexture[0].set(2 + (3 * nbPixelIndex));
		}
		else if (tempRNB_RCURR < ratioRNB_RCURR_MIN) {
			wordPtr.LCDPTexture[0].set(2 + (3 * nbPixelIndex));
			wordPtr.LCDPTexture[1].set(2 + (3 * nbPixelIndex));
		}

		// G_NB - G_CURR
		tempGNB_GCURR = G_NB - G_CURR;
		//tempResult += ((tempGNB_GCURR > ratioGNB_GCURR_MAX) ? 16384 : ((tempGNB_GCURR < (ratioGNB_GCURR_MIN)) ? 49152 : 0));
		if (tempGNB_GCURR > ratioGNB_GCURR_MAX) {
			wordPtr.LCDPTexture[0].set(1 + (3 * nbPixelIndex));
		}
		else if (tempGNB_GCURR < ratioGNB_GCURR_MIN) {
			wordPtr.LCDPTexture[0].set(1 + (3 * nbPixelIndex));
			wordPtr.LCDPTexture[1].set(1 + (3 * nbPixelIndex));
		}
		
		// B_NB - B_CURR
		tempBNB_BCURR = B_NB - B_CURR;
		//tempResult += ((tempBNB_BCURR > ratioBNB_BCURR_MAX) ? 4096 : ((tempBNB_BCURR < (ratioBNB_BCURR_MIN)) ? 12288 : 0));
		if (tempBNB_BCURR > ratioBNB_BCURR_MAX) {
			wordPtr.LCDPTexture[0].set(3 * nbPixelIndex);
		}
		else if (tempBNB_BCURR < ratioBNB_BCURR_MIN) {
			wordPtr.LCDPTexture[0].set(3 * nbPixelIndex);
			wordPtr.LCDPTexture[1].set(3 * nbPixelIndex);
		}

		// R_NB - G_CURR
		tempRNB_GCURR = R_NB - G_CURR;
		//tempResult += ((tempRNB_GCURR > ratioRNB_GCURR_MAX) ? 1024 : ((tempRNB_GCURR < (ratioRNB_GCURR_MIN)) ? 3072 : 0));
		if (tempRNB_GCURR > ratioRNB_GCURR_MAX) {
			wordPtr.LCDPColour[0].set(5 + (6 * nbPixelIndex));
		}
		else if (tempRNB_GCURR < ratioRNB_GCURR_MIN) {
			wordPtr.LCDPColour[0].set(5 + (6 * nbPixelIndex));
			wordPtr.LCDPColour[1].set(5 + (6 * nbPixelIndex));
		}
		
		// R_NB - B_CURR
		tempRNB_BCURR = R_NB - B_CURR;
		//tempResult += ((tempRNB_BCURR > ratioRNB_BCURR_MAX) ? 256 : ((tempRNB_BCURR < (ratioRNB_BCURR_MIN)) ? 768 : 0));
		if (tempRNB_BCURR > ratioRNB_BCURR_MAX) {
			wordPtr.LCDPColour[0].set(4 + (6 * nbPixelIndex));
		}
		else if (tempRNB_BCURR < ratioRNB_BCURR_MIN) {
			wordPtr.LCDPColour[0].set(4 + (6 * nbPixelIndex));
			wordPtr.LCDPColour[1].set(4 + (6 * nbPixelIndex));
		}

		// G_NB - R_CURR
		tempGNB_RCURR = G_NB - R_CURR;
		//tempResult += ((tempGNB_RCURR > ratioGNB_RCURR_MAX) ? 64 : ((tempGNB_RCURR < (ratioGNB_RCURR_MIN)) ? 192 : 0));
		if (tempGNB_RCURR > ratioGNB_RCURR_MAX) {
			wordPtr.LCDPColour[0].set(3 + (6 * nbPixelIndex));
		}
		else if (tempGNB_RCURR < ratioGNB_RCURR_MIN) {
			wordPtr.LCDPColour[0].set(3 + (6 * nbPixelIndex));
			wordPtr.LCDPColour[1].set(3 + (6 * nbPixelIndex));
		}
		// G_NB - B_CURR
		tempGNB_BCURR = G_NB - B_CURR;
		//tempResult += ((tempGNB_BCURR > ratioGNB_BCURR_MAX) ? 16 : ((tempGNB_BCURR < (ratioGNB_BCURR_MIN)) ? 48 : 0));
		if (tempGNB_BCURR > ratioGNB_BCURR_MAX) {
			wordPtr.LCDPColour[0].set(2 + (6 * nbPixelIndex));
		}
		else if (tempGNB_BCURR < ratioGNB_BCURR_MIN) {
			wordPtr.LCDPColour[0].set(2 + (6 * nbPixelIndex));
			wordPtr.LCDPColour[1].set(2 + (6 * nbPixelIndex));
		}

		// B_NB - R_CURR
		tempBNB_RCURR = B_NB - R_CURR;
		//tempResult += ((tempBNB_RCURR > ratioBNB_RCURR_MAX) ? 4 : ((tempBNB_RCURR < (ratioBNB_RCURR_MIN)) ? 12 : 0));
		if (tempBNB_RCURR > ratioBNB_RCURR_MAX) {
			wordPtr.LCDPColour[0].set(1 + (6 * nbPixelIndex));
		}
		else if (tempBNB_RCURR < ratioBNB_RCURR_MIN) {
			wordPtr.LCDPColour[0].set(1 + (6 * nbPixelIndex));
			wordPtr.LCDPColour[1].set(1 + (6 * nbPixelIndex));
		}
		// B_NB - G_CURR
		tempBNB_GCURR = B_NB - G_CURR;
		//tempResult += ((tempBNB_GCURR > ratioBNB_GCURR_MAX) ? 1 : ((tempBNB_GCURR < (ratioBNB_GCURR_MIN)) ? 3 : 0));
		if (tempBNB_GCURR > ratioBNB_GCURR_MAX) {
			wordPtr.LCDPColour[0].set(6 * nbPixelIndex);
		}
		else if (tempBNB_GCURR < ratioBNB_GCURR_MIN) {
			wordPtr.LCDPColour[0].set(6 * nbPixelIndex);
			wordPtr.LCDPColour[1].set(6 * nbPixelIndex);
		}		
	}
}
// Calculate word persistence value
void BackgroundSubtractorLCDP::GetLocalWordPersistence(DescriptorStruct &wordPtr, size_t &currFrameIndex,
	size_t offsetValue, float &persistenceValue) {
	persistenceValue = (float)(wordPtr.frameCount) / ((wordPtr.q - wordPtr.p) + ((currFrameIndex - wordPtr.q) * 2) + offsetValue);
}

/*=====LUT Methods=====*/
// Generate neighborhood pixel offset value
void BackgroundSubtractorLCDP::GenerateNbOffset(PxInfo &pxInfoPtr)
{
	int currX = pxInfoPtr.coor_x;
	int currY = pxInfoPtr.coor_y;
	for (int nbIndex = 0; nbIndex < (sizeof(nbOffset) / sizeof(cv::Point)); nbIndex++) {

		// Coordinate X value for neighborhood pixel
		pxInfoPtr.nbIndex[nbIndex].coor_x = std::min(frameSizeZero.width, std::max(0, (currX + nbOffset[nbIndex].x)));
		// Coordinate X value for neighborhood pixel
		pxInfoPtr.nbIndex[nbIndex].coor_y = std::min(frameSizeZero.height, std::max(0, (currY + nbOffset[nbIndex].y)));
		// Data index for neighborhood pixel's pointer
		pxInfoPtr.nbIndex[nbIndex].dataIndex = ((pxInfoPtr.nbIndex[nbIndex].coor_y*(frameSize.width)) + (pxInfoPtr.nbIndex[nbIndex].coor_x));
		// Data index for neighborhood pixel's BGR pointer
		pxInfoPtr.nbIndex[nbIndex].bgrDataIndex = 3 * pxInfoPtr.nbIndex[nbIndex].dataIndex;
		// Model index for neighborhood pixel
		pxInfoPtr.nbIndex[nbIndex].modelIndex = WORDS_NO * pxInfoPtr.nbIndex[nbIndex].dataIndex;
	}
}
// Generate LCD differences Lookup table (0: 100% Same -> 1: 100% Different)
void BackgroundSubtractorLCDP::GenerateLCDDiffLUT() {
	float countColour = 0;
	float countTexture = 0;
	int tempDiffIndex;
	//for (int diffIndex = 0; diffIndex < descPatternLength; diffIndex++) {
	//		countColour = 0;
	//		countTexture = 0;
	//		tempDiffIndex = diffIndex;
	//		// Checking bit by bit 
	//		// color variation
	//		for (size_t bitIndex = 0; bitIndex < 6; bitIndex++) {
	//			// The second bit is 1 - 11/01
	//			if ((tempDiffIndex & 1) == 1) {
	//				countColour += 1;
	//				tempDiffIndex >>= 2;
	//			}
	//			// The second bit is 0 - 00/10
	//			else {
	//				tempDiffIndex >>= 1;
	//				// The first bit is 1 - 10
	//				if ((tempDiffIndex & 1) == 1)
	//					countColour += 1;
	//				tempDiffIndex >>= 1;
	//			}
	//		}
	//		// Texture variation
	//		for (size_t bitIndex = 0; bitIndex < 3; bitIndex++) {
	//			// The second bit is 1 - 11/01
	//			if ((tempDiffIndex & 1) == 1) {
	//				countTexture += 1;
	//				tempDiffIndex >>= 2;
	//			}
	//			// The second bit is 0 - 00/10
	//			else {
	//				tempDiffIndex >>= 1;
	//				// The first bit is 1 - 10
	//				if ((tempDiffIndex & 1) == 1)
	//					countTexture += 1;
	//				tempDiffIndex >>= 1;
	//			}
	//		}
	//		//if (countTexture > 0) {
	//		//	LCDDiffLUTPtr[diffIndex] = 1;
	//		//}
	//		//else if (countColour > 0) {
	//		//	LCDDiffLUTPtr[diffIndex] = 1;
	//		//}
	//		//else {
	//		//	LCDDiffLUTPtr[diffIndex] = 0;
	//		//}
	//		LCDDiffLUTPtr[diffIndex]= ((countTexture / 3) + (countColour / 6)) / 2; 1 3   2 3  3 3  0 5

	//}
	for (int diffColourIndex = 0; diffColourIndex < 97; diffColourIndex++) {
		for (int diffTextureIndex = 0; diffTextureIndex < 49; diffTextureIndex++) {
			LCDDiffLUTPtr[diffColourIndex][diffTextureIndex] = ((float(diffColourIndex)/ 96.0f) + (float(diffTextureIndex) / 48.0f)) / 2.0f;
		}
	}

}

/*=====MATCHING Methods=====*/ // Edited on 14 May 2017
							   // Descriptor matching (RETURN: matchResult-1:Not match, 0: Match)
void BackgroundSubtractorLCDP::DescriptorMatching(DescriptorStruct &bgWord, DescriptorStruct &currWord
	,  size_t &descNeighNo,  double LCDPThreshold,  double upLCDPThreshold,  double RGBThreshold,
	float &LCDPDistance, float &RGBDistance, bool &matchResult, bool &matchResultBoth)
{

	// Match LCD descriptor
	if (clsLCDPDiffSwitch) {
		// LCD Matching (RETURN: LCDPResult-1:Not match, 0: Match)
		LCDPMatching(bgWord, currWord, descNeighNo, LCDPThreshold, LCDPDistance, matchResult);
	}
	// Match RGB descriptor
	if (clsRGBDiffSwitch) {
		bool RGBResult = false;
		// RGB Matching (RETURN: RGBResult-1:Not match, 0: Match)
		RGBMatching(bgWord, currWord, RGBThreshold, RGBDistance, RGBResult);

		if (clsLCDPDiffSwitch) {
			// LCDP BG and RGB FG
			if (!matchResult == RGBResult) {
				bool darkResult = false;
				// Check dark pixel (RETURN-1:Not Dark Pixel, 0: Dark Pixel)
				RGBDarkPixel(bgWord, currWord, darkResult);
				// If dark pixel, then it classify as shadow pixel
				matchResult = (darkResult) ? true : false;
			}
			// LCDP FG and RGB BG
			else if (matchResult == !RGBResult) {
				// If previous results is not match, matching again with more larger threshold to indicate the pixel exactly belong to FG
				matchResult = (LCDPDistance > upLCDPThreshold) ? true : false;
			}
			else if (!matchResult == !RGBResult) {
				matchResultBoth = true;
			}
		}
		else {
			matchResult = RGBResult;
		}
	}
}
// LCD Matching (RETURN-1:Not match, 0: Match)
void BackgroundSubtractorLCDP::LCDPMatching(DescriptorStruct &bgWord, DescriptorStruct &currWord,
	 size_t &descNeighNo, double &LCDPThreshold, float &minDistance, bool &matchResult) {
	float tempDistance = 0.0f;
	//for (size_t neighbourIndex = 0; neighbourIndex < descNeighNo; neighbourIndex++) {
	//	// Calculate the total number of bit that are different
	//	//tempDistance += LCDDiffLUTPtr[bgWord.LCDP[neighbourIndex]][currWord.LCDP[neighbourIndex]];

	//	tempDistance += LCDDiffLUTPtr[std::abs(bgWord.LCDP[neighbourIndex] - currWord.LCDP[neighbourIndex])];
	//}

	//XOR FIRST ROUND
	std::bitset<96> resultColour1 = bgWord.LCDPColour[0]^currWord.LCDPColour[0];
	std::bitset<96> resultColour2 = (~resultColour1)&(bgWord.LCDPColour[1]^currWord.LCDPColour[1]);
	std::bitset<48> resultTexture1 = bgWord.LCDPTexture[0]^currWord.LCDPTexture[0];
	std::bitset<48> resultTexture2 = (~resultTexture1)&(bgWord.LCDPTexture[1]^currWord.LCDPTexture[1]);
	minDistance = LCDDiffLUTPtr[(resultColour1.count() + resultColour2.count())][(resultTexture1.count() + resultTexture2.count())];

	//minDistance = tempDistance / descNeighNo;
	matchResult = (minDistance > LCDPThreshold) ? true : false;
}
// RGB Matching (RETURN-1:Not match, 0: Match) Checked May 14
void BackgroundSubtractorLCDP::RGBMatching(DescriptorStruct &bgWord, DescriptorStruct &currWord,
	double &RGBThreshold, float &minDistance, bool &matchResult)
{
	matchResult = false;
	// Maximum of color differences
	uint distance = 255;
	uint tempDistance = 0;
	for (int channel = 0; channel < 3; channel++) {
		tempDistance = std::abs(bgWord.rgb[channel] - currWord.rgb[channel]);
		// Update minimum distance
		distance = tempDistance < distance ? tempDistance : distance;
		if (tempDistance > RGBThreshold) {
			// return as foreground pixel
			matchResult = true;
			break;
		}
	}
	minDistance = float(distance) / 255.0f;
}
// RGB Dark Pixel (RETURN-1:Not Dark Pixel, 0: Dark Pixel) Checked May 14
void BackgroundSubtractorLCDP::RGBDarkPixel(DescriptorStruct &bgWord, DescriptorStruct &currWord, bool &result)
{
	double IntensityRatio, totalCurrIntensityValue = 0.0, totalBgIntensityValue = 0.0, currIntensityValue, bgIntensityValue, currRValue, bgRValue,
		currGValue, bgGValue, RDiff, GDiff;
	result = true;
	for (int channel = 0; channel < 3; channel++) {
		totalCurrIntensityValue += currWord.rgb[channel];
		totalBgIntensityValue += bgWord.rgb[channel];
	}
	currIntensityValue = totalCurrIntensityValue / 3.0;
	bgIntensityValue = totalBgIntensityValue / 3.0;
	IntensityRatio = (currIntensityValue / bgIntensityValue);
	//if ((IntensityRatio <0.8) && (IntensityRatio > 0.25)) {
	if ((IntensityRatio < darkMaxIntensityRatio) && (IntensityRatio > darkMinIntensityRatio)) {
		bgRValue = double(bgWord.rgb[2] / totalBgIntensityValue);
		currRValue = double(currWord.rgb[2] / totalCurrIntensityValue);
		bgGValue = double(bgWord.rgb[1] / totalBgIntensityValue);
		currGValue = double(currWord.rgb[1] / totalCurrIntensityValue);
		RDiff = std::abs(bgRValue - currRValue);
		GDiff = std::abs(bgGValue - currGValue);
		if (((RDiff <= darkRDiffRatioMax) && (GDiff <= darkGDiffRatioMax)) && ((RDiff >= darkRDiffRatioMin) && (GDiff >= darkGDiffRatioMin))) {
			//if((RDiff < 0.15) && (GDiff < 0.1)) {
			result = false;
		}
		//else if ((IntensityRatio < 0.675507f) && (IntensityRatio > 0.430716f)) {
		//	if (((RDiff <= 0.023803f) && (GDiff <= 0.013378f)) && ((RDiff >= 0.000554f) && (GDiff >= -0.002061f))) {
		//		//if((RDiff < 0.15) && (GDiff < 0.1)) {
		//		//if((RDiff < 0.15) && (GDiff < 0.1)) {
		//		result = false;
		//	}
		//}
	}
}
// Dark Pixel generator (RETURN-255: Not dark pixel, 0: dark pixel)
void BackgroundSubtractorLCDP::DarkPixelGenerator(cv::Mat &inputGrayImg, cv::Mat &inputRGBImg,
	cv::Mat &lastGrayImg, cv::Mat &lastRGBImg, cv::Mat &darkPixel) {
	darkPixel = cv::Scalar_<uchar>::all(255);
	// Store the pixel's intensity values
	double IntensityRatio, totalCurrIntensityValue, totalLastIntensityValue, currIntensityValue, lastIntensityValue, currRValue, lastRValue, currGValue, lastGValue, RDiff, GDiff;

	for (size_t pxPointer = 0; pxPointer < frameInitTotalPixel; ++pxPointer) {
		totalCurrIntensityValue = double(inputRGBImg.data[(pxPointer * 3)] + inputRGBImg.data[(pxPointer * 3) + 1] + inputRGBImg.data[(pxPointer * 3) + 2]);
		totalLastIntensityValue = double(lastRGBImg.data[(pxPointer * 3)] + lastRGBImg.data[(pxPointer * 3) + 1] + lastRGBImg.data[(pxPointer * 3) + 2]);
		currIntensityValue = totalCurrIntensityValue / 3.0;
		lastIntensityValue = totalLastIntensityValue / 3.0;

		IntensityRatio = (currIntensityValue / lastIntensityValue);

		if ((IntensityRatio < darkMaxIntensityRatio) && (IntensityRatio > darkMinIntensityRatio)) {
			lastRValue = double(lastRGBImg.data[(pxPointer * 3) + 2]) / totalLastIntensityValue;
			currRValue = double(inputRGBImg.data[(pxPointer * 3) + 2]) / totalCurrIntensityValue;
			lastGValue = double(lastRGBImg.data[(pxPointer * 3) + 1]) / totalLastIntensityValue;
			currGValue = double(inputRGBImg.data[(pxPointer * 3) + 1]) / totalCurrIntensityValue;
			RDiff = std::abs(lastRValue - currRValue);
			GDiff = std::abs(lastGValue - currGValue);
			if ((RDiff <= darkRDiffRatioMax) && (GDiff <= darkGDiffRatioMax)) {
				if ((RDiff >= darkRDiffRatioMin) && (GDiff >= darkGDiffRatioMin)) {
					darkPixel.data[pxPointer] = 0;
				}
			}
		}
	}
}

/*=====POST-PROCESSING Methods=====*/
// Compensation with Motion History - checked
cv::Mat BackgroundSubtractorLCDP::CompensationMotionHist(cv::Mat T_1FGMask, cv::Mat T_2FGMask, cv::Mat currFGMask,
	float postCompensationThreshold) {
	cv::Mat compensationResult;
	compensationResult.create(frameSize, CV_8UC1);
	compensationResult = cv::Scalar_<uchar>::all(0);
	for (size_t pxPointer = 0; pxPointer < frameInitTotalPixel; pxPointer++) {
		if (!currFGMask.data[pxPointer]) {
			double totalFGMask = 0.0;
			for (size_t nbIndex = 0; nbIndex < 9; nbIndex++) {
				totalFGMask += T_1FGMask.data[pxInfoLUTPtr[pxPointer].nbIndex[nbIndex].dataIndex];
				totalFGMask += T_2FGMask.data[pxInfoLUTPtr[pxPointer].nbIndex[nbIndex].dataIndex];
				totalFGMask += currFGMask.data[pxInfoLUTPtr[pxPointer].nbIndex[nbIndex].dataIndex];
			}
			totalFGMask /= 255.0;
			compensationResult.data[pxPointer] = ((totalFGMask / 26.0) > postCompensationThreshold) ? 255 : 0;
		}
	}

	return compensationResult;
}
// Contour filling the empty holes - checked
cv::Mat BackgroundSubtractorLCDP::ContourFill(cv::Mat inputImg) {
	cv::Mat input;
	cv::threshold(inputImg, input, 125, 255, cv::THRESH_BINARY);
	// Loop through the border pixels and if they're black, floodFill from there
	cv::Mat mask;
	input.copyTo(mask);
	for (int i = 0; i < mask.cols; i++) {
		if (mask.at<char>(0, i) == 0) {
			cv::floodFill(mask, cv::Point(i, 0), 255, 0, 10, 10);
		}
		if (mask.at<char>(mask.rows - 1, i) == 0) {
			cv::floodFill(mask, cv::Point(i, mask.rows - 1), 255, 0, 10, 10);
		}
	}
	for (int i = 0; i < mask.rows; i++) {
		if (mask.at<char>(i, 0) == 0) {
			cv::floodFill(mask, cv::Point(0, i), 255, 0, 10, 10);
		}
		if (mask.at<char>(i, mask.cols - 1) == 0) {
			cv::floodFill(mask, cv::Point(mask.cols - 1, i), 255, 0, 10, 10);
		}
	}
	// Compare mask with original.
	cv::Mat output;
	inputImg.copyTo(output);
	for (size_t pxPointer = 0; pxPointer < frameInitTotalPixel; pxPointer++) {
		if (mask.data[pxPointer] == 0) {
			output.data[pxPointer] = 255;
		}
	}
	return output;
}
// Border line reconstruct
cv::Mat BackgroundSubtractorLCDP::BorderLineReconst(cv::Mat inputMask)
{
	cv::Mat reconstructResult;
	reconstructResult.create(frameSize, CV_8UC1);
	reconstructResult = cv::Scalar_<uchar>(0);
	const size_t maxHeight = frameSize.height - 1;
	const size_t maxWidth = frameSize.width - 1;
	const size_t maxLineHeight = maxHeight*0.7;
	const size_t maxLineWidth = maxWidth*0.7;
	const size_t startIndexList_Y[4] = { 0,0,0,maxHeight };
	const size_t endIndexList_Y[4] = { 0,maxHeight,maxHeight,maxHeight };
	const size_t startIndexList_X[4] = { 0,0,maxWidth,0 };
	const size_t endIndexList_X[4] = { maxWidth,0,maxWidth,maxWidth };
	for (int line = 0; line < 4; line++) {
		uchar previousIndex = 0;
		size_t previousIndex_Y_start = startIndexList_Y[line];
		size_t previousIndex_Y_end = startIndexList_Y[line];
		size_t previousIndex_X_start = startIndexList_X[line];
		size_t previousIndex_X_end = startIndexList_X[line];
		bool previous = false;
		bool completeLine = false;
		for (int rowIndex = startIndexList_Y[line]; rowIndex <= endIndexList_Y[line]; rowIndex++) {
			for (int colIndex = startIndexList_X[line]; colIndex <= endIndexList_X[line]; colIndex++) {
				size_t pxPointer = (rowIndex*frameSize.width) + colIndex;
				uchar currFGMask = *(inputMask.data + pxPointer);

				if ((currFGMask != previousIndex) && (currFGMask == 255)) {
					if (!previous) {
						previous = true;
						previousIndex = 255;
						if (((previousIndex_Y_start < previousIndex_Y_end) || (previousIndex_X_start < previousIndex_X_end)) && completeLine) {
							for (int recRowIndex = previousIndex_Y_start; recRowIndex <= previousIndex_Y_end; recRowIndex++) {
								for (int recColIndex = previousIndex_X_start; recColIndex <= previousIndex_X_end; recColIndex++) {
									size_t upPxPointer = (recRowIndex*frameSize.width) + recColIndex;
									*(reconstructResult.data + upPxPointer) = 255;
								}
							}
							previousIndex_Y_end = rowIndex;
							previousIndex_X_end = colIndex;
							completeLine = false;
						}
					}
					previousIndex_Y_start = rowIndex;
					previousIndex_X_start = colIndex;
				}
				else if ((currFGMask != previousIndex) && currFGMask == 0) {
					if (previous) {
						previous = false;
						previousIndex = 0;
						completeLine = true;
					}
					previousIndex_Y_end = rowIndex;
					previousIndex_X_end = colIndex;
				}
				else if (currFGMask == 255) {
					if (previous) {
						previousIndex_Y_start = rowIndex;
						previousIndex_X_start = colIndex;
					}
				}
				else if (currFGMask == 0) {
					if (!previous) {
						size_t ybalance = previousIndex_Y_end - previousIndex_Y_start;
						size_t xbalance = previousIndex_X_end - previousIndex_X_start;
						if (ybalance > (maxLineHeight)) {
							previousIndex_Y_start = rowIndex;
							previousIndex_X_start = colIndex;
						}
						else if (xbalance > (maxLineWidth)) {
							previousIndex_Y_start = rowIndex;
							previousIndex_X_start = colIndex;
						}
						previousIndex_Y_end = rowIndex;
						previousIndex_X_end = colIndex;
					}
				}
			}
		}
	}
	return reconstructResult;
}

/*=====OTHERS Methods=====*/
// Save parameters
void BackgroundSubtractorLCDP::SaveParameter(std::string versionFolderName, std::string saveFolderName) {
	std::ofstream myfile;
	myfile.open(saveFolderName + "/parameter.txt", std::ios::app);
	myfile << "\n----VIDEO PARAMETER----";
	myfile << "\nTOTAL NUMBER OF FRAMES:";
	myfile << frameNo;
	myfile << "\nVIDEO WIDTH:";
	myfile << frameSize.width;
	myfile << "\nVIDEO HEIGHT:";
	myfile << frameSize.height;
	myfile << "\n\n----METHOD THRESHOLD----\n<< << <-DESCRIPTOR DEFAULT PARAMETER-> >> >>";
	myfile << "\nTotal number of LCD differences per pixel:";
	myfile << descDiffNo;
	myfile << "\nTotal number of LCD descriptor's neighbour:";
	myfile << descNbNo;
	myfile << "\nLCD color differences ratio:";
	myfile << descColourDiffRatio;
	myfile << "\nPersistence's offset value:";
	myfile << descOffsetValue;
	myfile << "\n\n<<<<<-CLASSIFIER DEFAULT PARAMETER->>>>>";
	myfile << "\nRGB detection switch:";
	myfile << clsRGBDiffSwitch;
	myfile << "\nDefault RGB differences threshold:";
	myfile << clsRGBThreshold;
	myfile << "\nLCD detection switch:";
	myfile << clsLCDPDiffSwitch;
	myfile << "\nDefault LCD differences threshold:";
	myfile << clsLCDPThreshold;
	myfile << "\nDefault Up LCD differences threshold:";
	myfile << clsUpLCDPThreshold;
	myfile << "\nMaximum of LCD differences threshold:";
	myfile << clsLCDPMaxThreshold;
	myfile << "\nInitial matched persistence value threshold:";
	float * persistenceThreshold = (float*)(clsPersistenceThreshold.data);
	myfile << *(persistenceThreshold);
	myfile << "\nNeighbourhood matching switch:";
	myfile << clsNbMatchSwitch;
	myfile << "\nClassify matching threshold:";
	myfile << clsMatchThreshold;
	myfile << "\nPost-processing switch:";
	myfile << postSwitch;

	myfile << "\n\n<<<<<-RGB DARK PIXEL PARAMETER->>>>>";
	myfile << "\nMinimum Intensity Ratio:";
	myfile << darkMinIntensityRatio;
	myfile << "\Maximum Intensity Ratio:";
	myfile << darkMaxIntensityRatio;	
	myfile << "\nMinimum R - channel different ratio:";
	myfile << darkRDiffRatioMin;
	myfile << "\nMaximum R - channel different ratio:";
	myfile << darkRDiffRatioMax;
	myfile << "\nnMinimum G - channel different ratio:";
	myfile << darkGDiffRatioMin;
	myfile << "\nMaximum G - channel different ratio:";
	myfile << darkGDiffRatioMax;

	myfile << "\n\n<<<<<-UPDATE DEFAULT PARAMETER->>>>>";
	myfile << "\nFeedback loop switch:";
	myfile << upFeedbackSwitch;
	myfile << "\nDynamic rate increase value:";
	myfile << upDynamicRateIncrease;
	myfile << "\nDynamic rate decrease value:";
	myfile << upDynamicRateDecrease;
	myfile << "\nDynamic rate minimum value:";
	myfile << upMinDynamicRate;
	myfile << "\nLocal update rate change factor (Desc):";
	myfile << upUpdateRateDecrease;
	myfile << "\nLocal update rate change factor (Inc.):";
	myfile << upUpdateRateIncrease;
	myfile << "\nLocal update rate (Lower):";
	myfile << upLearningRateLowerCap;
	myfile << "\nLocal update rate (Upper):";
	myfile << upLearningRateUpperCap;
	myfile << "\nMaximum number of model:";
	myfile << WORDS_NO;
	
	myfile.close();
}