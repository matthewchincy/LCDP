#include "BackgroundSubtractorLCDP.h"
#include "RandUtils.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <time.h>

// Parameters used to define 'unstable' regions, based on segm noise/bg dynamics and local dist threshold values
#define UNSTABLE_REG_RATIO_MIN  (0.10f)
// Parameter used to average the min distance
#define MIN_DISTANCE_ALPHA (0.01f)
// Local define used to specify the default frame size (320x240 = QVGA) // Checked
#define DEFAULT_FRAME_SIZE cv::Size(320,240)
// Pre-processing gaussian size
#define PRE_DEFAULT_GAUSSIAN_SIZE cv::Size(9,9)

/*******CONSTRUCTOR*******/ // Checked
BackgroundSubtractorLCDP::BackgroundSubtractorLCDP(size_t inputWordsNo, bool inputPreSwitch,
	double inputDescColourDiffRatio, bool inputClsRGBDiffSwitch, double inputClsRGBThreshold, double inputClsUpRGBThreshold, bool inputClsLCDPDiffSwitch,
	double inputClsLCDPThreshold, double inputClsUpLCDPThreshold, double inputClsLCDPMaxThreshold, int inputClsMatchThreshold,
	bool inputClsNbMatchSwitch, cv::Mat inputROI, cv::Size inputFrameSize, bool inputUpRandomReplaceSwitch, bool inputUpRandomUpdateNbSwitch,
	bool inputUpFeedbackSwitch, float inputUpDynamicRateIncrease, float inputUpDynamicRateDecrease, float inputUpUpdateRateIncrease,
	float inputUpUpdateRateDecrease, float inputUpUpdateRateLowest, float inputUpUpdateRateHighest, bool inputPostSwitch) :
	/*=====LOOK-UP TABLE=====*/
	// Internal pixel info LUT for all possible pixel indexes
	pxInfoLUTPtr(nullptr),
	// LCD differences LUT
	LCDDiffLUTPtr(nullptr),

	/*=====MODEL Parameters=====*/
	// Store the background's words and it's iterator
	bgWordPtr(nullptr),
	bgWordPtrIter(nullptr),
	// Store the currect frame's words and it's iterator
	currWordPtr(nullptr),
	currWordPtrIter(nullptr),
	// Total number of words to represent a pixel
	WORDS_NO(inputWordsNo),
	// Frame index
	frameIndex(1),

	/*=====PRE-PROCESS Parameters=====*/
	// Pre processing switch
	preSwitch(inputPreSwitch),
	// Size of gaussian filter
	preGaussianSize(PRE_DEFAULT_GAUSSIAN_SIZE),

	/*=====DESCRIPTOR Parameters=====*/
	// Size of neighbourhood 3(3x3)/5(5x5)
	descNbSize(5),
	// Total number of neighbourhood pixel 8(3x3)/16(5x5)
	descNbNo(16),
	// LCD colour differences ratio
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
	// UP RGB differences threshold
	clsUpRGBThreshold(inputClsUpRGBThreshold),
	// LCDP detection switch
	clsLCDPDiffSwitch(inputClsLCDPDiffSwitch),
	// LCDP differences threshold
	clsLCDPThreshold(inputClsLCDPThreshold),
	// UP LCDP differences threshold
	clsUpLCDPThreshold(inputClsUpLCDPThreshold),
	// Maximum number of LCDP differences threshold
	clsLCDPMaxThreshold(inputClsLCDPMaxThreshold),
	// Neighbourhood matching switch
	clsNbMatchSwitch(inputClsNbMatchSwitch),
	// Matching threshold
	clsMatchThreshold(inputClsMatchThreshold),

	/*=====FRAME Parameters=====*/
	// ROI frame
	frameRoi(inputROI),
	// Size of input frame
	frameSize(inputFrameSize),
	// Size of input frame starting from 0
	frameSizeZero(cv::Size(inputFrameSize.width - 1, inputFrameSize.height - 1)),
	// Total number of pixel of region of interest
	frameRoiTotalPixel(cv::countNonZero(inputROI)),
	// Total number of pixel of input frame
	frameInitTotalPixel(inputFrameSize.area()),

	/*=====UPDATE Parameters=====*/
	// Random replace model switch
	upRandomReplaceSwitch(inputUpRandomReplaceSwitch),
	// Random update neighbourhood model switch
	upRandomUpdateNbSwitch(inputUpRandomUpdateNbSwitch),
	// Feedback loop switch
	upFeedbackSwitch(inputUpFeedbackSwitch),
	// Feedback V(x) Increment
	upDynamicRateIncrease(inputUpDynamicRateIncrease),
	// Feedback V(x) Decrement
	upDynamicRateDecrease(inputUpDynamicRateDecrease),
	// Feedback T(x) Increment
	upUpdateRateIncrease(inputUpUpdateRateIncrease),
	// Feedback T(x) Decrement
	upUpdateRateDecrease(inputUpUpdateRateDecrease),
	// Feedback T(x) Lowest
	upUpdateRateLowerCap(inputUpUpdateRateLowest),
	// Feedback T(x) Highest
	upUpdateRateUpperCap(inputUpUpdateRateHighest),

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
	delete[] LCDDiffLUTPtr;
	delete[] bgWordPtr;
	delete[] currWordPtr;
}

/*******INITIALIZATION*******/ // Checked
void BackgroundSubtractorLCDP::Initialize(const cv::Mat inputFrame, cv::Mat inputROI)
{
	/*=====LOOK-UP TABLE=====*/
	// Internal pixel info LUT for all possible pixel indexes
	pxInfoLUTPtr = new PxInfo[frameInitTotalPixel];
	memset(pxInfoLUTPtr, 0, sizeof(PxInfo)*frameInitTotalPixel);
	// LCD differences LUT
	LCDDiffLUTPtr = new float[descPatternLength];
	memset(LCDDiffLUTPtr, 0, sizeof(float)*descPatternLength);
	// Generate LCD difference Lookup table
	GenerateLCDDiffLUT();

	/*=====MODEL Parameters=====*/
	// Store the background's word and it's iterator
	bgWordPtr = new DescriptorStruct[frameInitTotalPixel*WORDS_NO];
	memset(bgWordPtr, 0, sizeof(DescriptorStruct)*frameInitTotalPixel*WORDS_NO);
	bgWordPtrIter = bgWordPtr;
	// Store the currect frame's word and it's iterator
	currWordPtr = new DescriptorStruct[frameInitTotalPixel];
	memset(currWordPtr, 0, sizeof(DescriptorStruct)*frameInitTotalPixel);
	currWordPtrIter = currWordPtr;
	
	/*=====CLASSIFIER Parameters=====*/
	// Minimum persistence threhsold value 
	clsMinPersistenceThreshold = (1.0f / descOffsetValue) * 2;
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
		// Current kernel size for median blur post-proc filtering
		postMedianFilterSize = tempMedianFilterSize;
		// Specifies the px update spread range
		upUse3x3Spread = !(frameInitTotalPixel > (DEFAULT_FRAME_SIZE.area() * 2));
		// Current learning rate caps
		upLearningRateLowerCap = upUpdateRateLowerCap;
		upLearningRateUpperCap = upUpdateRateUpperCap;
	}
	else {
		/*=====POST-PROCESS Parameters=====*/
		// Current kernel size for median blur post-proc filtering
		postMedianFilterSize = postMedianFilterSize;
		// Specifies the px update spread range
		upUse3x3Spread = true;
		// Current learning rate caps
		upLearningRateLowerCap = upUpdateRateLowerCap * 2;
		upLearningRateUpperCap = upUpdateRateUpperCap * 2;
	}
	
	/*=====RESULTS=====*/
	// Per-pixel distance thresholds ('R(x)', but used as a relative value to determine both 
	// intensity and descriptor variation thresholds)
	resLCDPDistThreshold.create(frameSize, CV_32FC1);
	resLCDPDistThreshold = cv::Scalar(1.0f);
	resRGBDistThreshold.create(frameSize, CV_32FC1);
	resRGBDistThreshold = cv::Scalar(1.0f);
	// Per-pixel dynamic learning rate ('V(x)')
	resLCDPDynamicRate.create(frameSize, CV_32FC1);
	resLCDPDynamicRate = cv::Scalar(1.0f);
	resRGBDynamicRate.create(frameSize, CV_32FC1);
	resRGBDynamicRate = cv::Scalar(1.0f);
	// Per-pixel update rates('T(x)')
	resLCDPUpdateRate.create(frameSize, CV_32FC1);
	resLCDPUpdateRate = cv::Scalar(upLearningRateLowerCap);
	resRGBUpdateRate.create(frameSize, CV_32FC1);
	resRGBUpdateRate = cv::Scalar(upLearningRateLowerCap);
	// Minimum LCDP distance
	resMinLCDPDistance.create(frameSize, CV_32FC1);
	resMinLCDPDistance = cv::Scalar(1.0f);
	// Minimum RGB distance
	resMinRGBDistance.create(frameSize, CV_32FC1);
	resMinRGBDistance = cv::Scalar(1.0f);
	// Current LCDP pixel distance
	resCurrLCDPPxDistance.create(frameSize, CV_32FC1);
	resCurrLCDPPxDistance = cv::Scalar(0.0f);
	// Current RGB pixel distance
	resCurrRGBPxDistance.create(frameSize, CV_32FC1);
	resCurrRGBPxDistance = cv::Scalar(0.0f);
	// Current foreground mask
	resCurrFGMask.create(frameSize, CV_8UC1);
	resCurrFGMask = cv::Scalar_<uchar>::all(0);
	// Current LCDP foreground mask
	resCurrLCDPFGMask.create(frameSize, CV_8UC1);
	resCurrLCDPFGMask = cv::Scalar_<uchar>::all(0);
	// Current Up LCDP foreground mask
	resCurrUpLCDPFGMask.create(frameSize, CV_8UC1);
	resCurrUpLCDPFGMask = cv::Scalar_<uchar>::all(0);
	// Current RGB foreground mask
	resCurrRGBFGMask.create(frameSize, CV_8UC1);
	resCurrRGBFGMask = cv::Scalar_<uchar>::all(0);
	// Dark Pixel
	resDarkPixel.create(frameSize, CV_8UC1);
	resDarkPixel = cv::Scalar_<uchar>::all(0);
	// Previous foreground mask
	resLastFGMask.create(frameSize, CV_8UC1);
	resLastFGMask = cv::Scalar_<uchar>::all(0);
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
			// Generate neighbourhood pixel offset value
			GenerateNbOffset(pxInfoLUTPtr[pxPointer]);
			// Descripto Generator-Generate pixels' descriptor (RGB+LCDP)
			DescriptorGenerator(inputFrame, pxInfoLUTPtr[pxPointer], currWordPtr[pxPointer]);
			pxPointer++;
		}
	}

	// Refresh model
	RefreshModel(1.0f);
}

// Program processing
void BackgroundSubtractorLCDP::Process(const cv::Mat inputImg, cv::Mat &outputImg)
{
	cv::Mat inputGrayImg;
	cv::cvtColor(inputImg, inputGrayImg, CV_RGB2GRAY);
	// PRE PROCESSING
	if (preSwitch) {
		cv::GaussianBlur(inputImg, inputImg, preGaussianSize, 0, 0);
	}

	// DETECTION PROCESS
	// Generate a map to indicate dark pixel (255: Not dark pixel, 0: Dark pixel)
	DarkPixelGenerator(inputGrayImg, inputImg, resLastGrayImg, resLastImg, resDarkPixel);
	// BG Word pointer
	DescriptorStruct * bgWord = nullptr;
	// Current bg word's persistence	
	float currWordPersistence;
	for (size_t pxPointer = 0; pxPointer < frameInitTotalPixel; ++pxPointer) {
		if (frameRoi.data[pxPointer]) {
			// Descriptor Generator-Generate pixels' descriptor (RGB+LCDP)
			DescriptorGenerator(inputImg, pxInfoLUTPtr[pxPointer], currWordPtr[pxPointer]);
			// Current distance threshold ('R(x)')
			float * currLCDPDistThreshold = (float*)(resLCDPDistThreshold.data + (pxPointer * 4));
			float * currRGBDistThreshold = (float*)(resRGBDistThreshold.data + (pxPointer * 4));
			// Current dynamic learning rate ('V(x)')
			float * currLCDPDynamicRate = (float*)(resLCDPDynamicRate.data + (pxPointer * 4));
			float * currRGBDynamicRate = (float*)(resRGBDynamicRate.data + (pxPointer * 4));
			// Current pixel's update rate ('T(x)')
			float * currLCDPUpdateRate = (float*)(resLCDPUpdateRate.data + (pxPointer * 4));
			float * currRGBUpdateRate = (float*)(resRGBUpdateRate.data + (pxPointer * 4));
			// Model index for current pixel
			const size_t currModelIndex = pxInfoLUTPtr[pxPointer].modelIndex;

			// LCDP differences threshold
			const double currLCDPThreshold = std::min(clsLCDPMaxThreshold, std::max(clsLCDPThreshold, (std::pow(2, double((*currLCDPDistThreshold))) / 512)));
			// Up LCDP differences threshold
			const double currUpLCDPThreshold = clsUpLCDPThreshold;
			// RGB differences threshold
			const double currRGBThreshold = std::max(clsRGBThreshold, floor(clsRGBThreshold*(*currRGBDistThreshold)));
			// Up RGB differences threshold
			const double currUpRGBThreshold = clsUpRGBThreshold;
			// Current pixel's foreground mask
			uchar * currFGMask = (resCurrFGMask.data + pxPointer);
			// Current pixel's LCDP foreground mask
			uchar * currLCDPFGMask = (resCurrLCDPFGMask.data + pxPointer);
			// Current pixel's Up LCDP foreground mask
			uchar * currUpLCDPFGMask = (resCurrUpLCDPFGMask.data + pxPointer);
			// Current pixel's RGB foreground mask
			uchar * currRGBFGMask = (resCurrRGBFGMask.data + pxPointer);
			// Current pixel's Up RGB foreground mask
			uchar * currUpRGBFGMask = (resCurrUpRGBFGMask.data + pxPointer);
			// Current dark pixel result
			uchar * currDarkPixel = (resDarkPixel.data + pxPointer);
			// Total number of neighbourhood pixel 8(3x3)/16(5x5)
			size_t currDescNeighNo = descNbNo;
			// Current pixel's descriptor
			DescriptorStruct currWord = currWordPtr[pxPointer];

			// Current pixel's min LCDP distance
			float * minLCDPDistance = (float*)(resMinLCDPDistance.data + (pxPointer * 4));
			// Current pixel's min RGB distance
			float * minRGBDistance = (float*)(resMinRGBDistance.data + (pxPointer * 4));
			// Current LCDP pixel distance
			float * currLCDPPxDistance = (float*)(resCurrLCDPPxDistance.data + (pxPointer * 4));
			// Current RGB pixel distance
			float * currRGBPxDistance = (float*)(resCurrRGBPxDistance.data + (pxPointer * 4));

			// Last word's persistence
			float currLastWordPersistence = FLT_MAX;
			// Current pixel's background word index
			int currLocalWordIdx = 0;

			// Number of potential matched model
			int clsLCDPPotentialMatch = 0;
			int clsUpLCDPPotentialNotMatch = 0;
			int clsRGBPotentialMatch = 0;
			int clsUpRGBPotentialNotMatch = 0;
			while (currLocalWordIdx < WORDS_NO && ((clsLCDPPotentialMatch < clsMatchThreshold) || (clsRGBPotentialMatch < clsMatchThreshold))) {
				// Current bg word
				bgWord = (bgWordPtr + currModelIndex + currLocalWordIdx);							
				GetLocalWordPersistence(*bgWord, frameIndex, descOffsetValue, currWordPersistence);
				float tempLCDPDistance = 0.0f;
				float tempRGBDistance = 0.0f;
				bool LCDPResult = false;
				bool upLCDPResult = false;
				bool RGBResult = false;
				bool upRGBResult = false;
				// False:Match true:Not match
				DescriptorMatching(*bgWord, currWord, currDescNeighNo, currLCDPThreshold, currUpLCDPThreshold, currRGBThreshold, currUpRGBThreshold,
					tempLCDPDistance, tempRGBDistance, LCDPResult, upLCDPResult, RGBResult, upRGBResult);

				if (((*bgWord).frameCount > 0) && (!LCDPResult)) {
					(*bgWord).frameCount += 1;
					(*bgWord).q = frameIndex;

					clsLCDPPotentialMatch++;
				}
				else if (upLCDPResult) {
					clsUpLCDPPotentialNotMatch++;
				}

				if (((*bgWord).frameCount > 0) && (!RGBResult)) {
					if (LCDPResult) {
						(*bgWord).frameCount += 1;
						(*bgWord).q = frameIndex;
					}
					clsRGBPotentialMatch++;
				}
				else if (upRGBResult) {
					clsUpRGBPotentialNotMatch++;
				}
				// Update MIN LCDP distance
				(*minLCDPDistance) = tempLCDPDistance < (*minLCDPDistance) ? tempLCDPDistance : (*minLCDPDistance);
				// Update MIN RGB distance
				(*minRGBDistance) = tempRGBDistance < (*minRGBDistance) ? tempRGBDistance : (*minRGBDistance);
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
			if ((clsLCDPPotentialMatch >= clsMatchThreshold) && (clsRGBPotentialMatch >= clsMatchThreshold)) {
				(*currFGMask) = 0;
				(*currLCDPFGMask) = 0;
				(*currRGBFGMask) = 0;
				(*currLCDPDynamicRate) = std::max(1.0f, (*currLCDPDynamicRate) - upDynamicRateDecrease);
				(*currRGBDynamicRate) = std::max(1.0f, (*currRGBDynamicRate) - upDynamicRateDecrease);
				// Check dark pixel
				if (*currDarkPixel) {
					(*currDarkPixel) = 0;
				}
			}
			// Classified as FG Pixels
			else {
				if (clsLCDPPotentialMatch >= clsMatchThreshold) {
					(*currLCDPFGMask) = 0;
					(*currLCDPDynamicRate) = std::max(1.0f, (*currLCDPDynamicRate) - upDynamicRateDecrease);
				}
				else {
					//if (clsUpLCDPPotentialNotMatch > (0.9*WORDS_NO)) {
					//	(*currUpLCDPFGMask) = 255;
					//}
					(*currLCDPFGMask) = 255;
				}
				if (clsRGBPotentialMatch >= clsMatchThreshold) {
					(*currRGBFGMask) = 0;
					(*currRGBDynamicRate) = std::max(1.0f, (*currRGBDynamicRate) - upDynamicRateDecrease);
				}
				else {
					//if (clsUpRGBPotentialNotMatch > (0.9*WORDS_NO)) {
					//	(*currUpRGBFGMask) = 255;
					//}
					(*currRGBFGMask) = 255;
				}
				size_t nbLCDPMatchNo = 0;
				size_t nbRGBMatchNo = 0;
				if (clsNbMatchSwitch) {
					// Compare with neighbour's model
					// Neighbour matching size (Max: 7x7)					
					if (clsLCDPPotentialMatch < clsMatchThreshold) {
						nbLCDPMatchNo = std::max(16.0f, std::floor((((*currLCDPDistThreshold) / 9) * 48)));
					}
					if (clsRGBPotentialMatch < clsMatchThreshold) {
						nbRGBMatchNo = std::max(16.0f, std::floor((((*currRGBDistThreshold) / 9) * 48)));
					}
					for (size_t nbIndex = 0; nbIndex < std::max(nbLCDPMatchNo, nbRGBMatchNo); nbIndex++) {
						// Neighbour pixel pointer
						size_t nbPxPointer = pxInfoLUTPtr[pxPointer].nbIndex[nbIndex].dataIndex;
						// Current neighbour pixel's matching threshold
						int clsNBMatchThreshold = clsMatchThreshold;
						// Number of potential matched model
						int clsNBPotentialLCDPMatch = 0;
						//int clsNBPotentialUpLCDPNotMatch = 0;
						// Number of potential matched model in RGB
						int clsNBPotentialRGBMatch = 0;
						//int clsNBPotentialUpRGBNotMatch = 0;
						// Neighbour pixel's model index
						const size_t nbModelIndex = pxInfoLUTPtr[pxPointer].nbIndex[nbIndex].modelIndex;
						// Current neighbour distance threshold
						float * nbLCDPDistThreshold = (float*)(resLCDPDistThreshold.data + (nbPxPointer * 4));
						float * nbRGBDistThreshold = (float*)(resRGBDistThreshold.data + (nbPxPointer * 4));
						// Neighbour LCD descriptor threshold
						const double nbLCDPThreshold = std::min(clsLCDPMaxThreshold, std::max(clsLCDPThreshold, (std::pow(2, double((*nbLCDPDistThreshold))) / 512)));
						// Neighbour Up LCD descriptor threshold
						const double nbUpLCDPThreshold = clsUpLCDPThreshold;
						// Neighbour RGB descriptor threshold
						const double nbRGBThreshold = std::max(clsRGBThreshold, floor(clsRGBThreshold*(*nbRGBDistThreshold)));
						// Neighbour Up RGB descriptor threshold
						const double nbUpRGBThreshold = clsUpRGBThreshold;

						float nbLastWordPersistence = FLT_MAX;
						size_t nbLocalWordIdx = 0;
						while ((nbLocalWordIdx < WORDS_NO) && ((clsNBPotentialLCDPMatch < clsNBMatchThreshold) || (clsNBPotentialRGBMatch < clsNBMatchThreshold))) {

							bgWord = (bgWordPtr + nbModelIndex + nbLocalWordIdx);
							GetLocalWordPersistence(*bgWord, frameIndex, descOffsetValue, currWordPersistence);
							float tempLCDPDistance = 0.0f;
							float tempRGBDistance = 0.0f;

							bool LCDPResult = false;
							bool upLCDPResult = false;
							bool RGBResult = false;
							bool upRGBResult = false;
							// False:Match true:Not match
							DescriptorMatching(*bgWord, currWord, currDescNeighNo, nbLCDPThreshold, nbUpLCDPThreshold, nbRGBThreshold, nbUpRGBThreshold,
								tempLCDPDistance, tempRGBDistance, LCDPResult, upLCDPResult, RGBResult, upRGBResult);

							if (((*bgWord).frameCount > 0) && (!LCDPResult)) {
								(*bgWord).frameCount += 1;
								(*bgWord).q = frameIndex;
								clsNBPotentialLCDPMatch++;

							}
							else if (upLCDPResult) {
								clsUpLCDPPotentialNotMatch++;
							}

							if (((*bgWord).frameCount > 0) && (!RGBResult)) {
								if (LCDPResult) {
									(*bgWord).frameCount += 1;
									(*bgWord).q = frameIndex;
								}
								clsNBPotentialRGBMatch++;
							}
							else if (upRGBResult) {
								clsUpRGBPotentialNotMatch++;
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
						// BG Pixels 
						if (nbIndex < nbLCDPMatchNo) {
							if (clsNBPotentialLCDPMatch >= clsNBMatchThreshold) {
								(*currLCDPFGMask) = 0;
								(*currLCDPDynamicRate) += upDynamicRateIncrease;
							}
							//else {
							//	if (clsNBPotentialUpLCDPNotMatch >(0.9*WORDS_NO)) {
							//		(*currUpLCDPFGMask) = 255;
							//	}
							//}
						}

						if (nbIndex < nbRGBMatchNo) {
							if (clsNBPotentialRGBMatch >= clsNBMatchThreshold) {
								(*currRGBFGMask) = 0;
								(*currRGBDynamicRate) += upDynamicRateIncrease;
							}
							//else {
							//	if (clsNBPotentialUpRGBNotMatch >(0.9*WORDS_NO)) {
							//		(*currUpRGBFGMask) = 255;
							//	}
							//}
						}
						if (!(*currLCDPFGMask) && !(*currRGBFGMask)) {
							break;
						}
					}
					
				}
				if ((clsUpLCDPPotentialNotMatch / (nbLCDPMatchNo + 1)) > (0.7*WORDS_NO)) {
					(*currUpLCDPFGMask) = 255;
				}
				if ((clsUpRGBPotentialNotMatch / (nbRGBMatchNo + 1)) > (0.7*WORDS_NO)) {
					(*currUpRGBFGMask) = 255;
				}

				// LCDP:BG, RGB:FG
				if (!(*currLCDPFGMask) && (*currRGBFGMask)) {
					// Check dark pixel
					if (!(*currDarkPixel)) {
						(*currFGMask) = 0;
					}
					else {
						(*currFGMask) = 255;
					}
				}
				// LCDP:FG, RGB:BG
				else if ((*currLCDPFGMask) && (!*currRGBFGMask)) {
					// Use different LCDP threshold
					if (currUpLCDPFGMask) {
						(*currFGMask) = 255;
						std::cout << "LCDP:FG,RGB:BG-FG";
					}
					else {
						//(*currDarkPixel) = 0;
						(*currFGMask) = 0;
					}
				}
				else if ((!*currLCDPFGMask) && (!*currRGBFGMask)) {
					(*currFGMask) = 0;
					//(*currDarkPixel) = 0;
				}
				else if ((*currLCDPFGMask) && (*currRGBFGMask)) {
					(*currFGMask) = 255;
				}
			}
			// UPDATE PROCESS
			// Random replace current frame's descriptor with the model
			if (upRandomReplaceSwitch) {
				if ((*currFGMask)) {
					// FG
					float randNumber = ((double)std::rand() / (RAND_MAX));
					bool checkLCDP = ((1 / (*currLCDPUpdateRate)) >= randNumber);
					bool checkRGB = ((1 / (*currRGBUpdateRate)) >= randNumber);
					if (checkLCDP || checkRGB) {
						float randNumber2 = ((double)std::rand() / (RAND_MAX / 2));
						bool checkLCDP2 = ((1 / (*currLCDPUpdateRate)) >= randNumber2);
						bool checkRGB2 = ((1 / (*currRGBUpdateRate)) >= randNumber2);
						if (checkLCDP2 || checkRGB2) {
							DescriptorStruct* bgWord = (bgWordPtr + currModelIndex + WORDS_NO - 1);
							if (checkRGB&&checkRGB2) {
								for (size_t channel = 0; channel < 3; channel++) {
									(*bgWord).rgb[channel] = currWord.rgb[channel];
								}
							}
							if (checkLCDP&&checkLCDP2) {
								for (size_t channel = 0; channel < descNbNo; channel++) {
									(*bgWord).LCDP[channel] = currWord.LCDP[channel];
								}
							}
						}
					}

				}
				else {
					// BG
					float randNumber = ((double)std::rand() / (RAND_MAX));
					bool checkLCDP = ((1 / (*currLCDPUpdateRate)) >= randNumber);
					bool checkRGB = ((1 / (*currRGBUpdateRate)) >= randNumber);
					if (checkLCDP || checkRGB) {
						int randNum = rand() % WORDS_NO;
						DescriptorStruct* bgWord = (bgWordPtr + currModelIndex + randNum);
						if (checkRGB) {
							for (size_t channel = 0; channel < 3; channel++) {
								(*bgWord).rgb[channel] = currWord.rgb[channel];
							}
						}
						if (checkLCDP) {
							for (size_t channel = 0; channel < descNbNo; channel++) {
								(*bgWord).LCDP[channel] = currWord.LCDP[channel];
							}
						}
					}
				}
			}
			// UPDATE PROCESS
			// Randomly update a selected neighbour descriptor with current descriptor
			if (upRandomUpdateNbSwitch) {
				float randNumber = ((double)std::rand() / (RAND_MAX));
				const bool checkLBSP = ((1 / *(currLCDPUpdateRate)) >= randNumber);
				const bool checkRGB = ((1 / *(currRGBUpdateRate)) >= randNumber);
				if (checkLBSP || checkRGB) {
					cv::Point sampleCoor;
					if (!upUse3x3Spread) {
						getRandSamplePosition_5x5(sampleCoor, cv::Point(pxInfoLUTPtr[pxPointer].coor_x, pxInfoLUTPtr[pxPointer].coor_y), 0, frameSize);
					}
					else {
						getRandSamplePosition_3x3(sampleCoor, cv::Point(pxInfoLUTPtr[pxPointer].coor_x, pxInfoLUTPtr[pxPointer].coor_y), 0, frameSize);
					}

					const size_t samplePxIndex = frameSize.width*sampleCoor.y + sampleCoor.x;
					// Start index of the model of the current pixel
					const size_t startNBModelIndex = pxInfoLUTPtr[samplePxIndex].modelIndex;

					// Current neighbour pixel's matching threshold
					int clsNBMatchThreshold = clsMatchThreshold;

					// Number of potential matched LCDP model
					int clsNBPotentialLCDPMatch= 0;
					// Number of potential matched LCDP model with Up threshold
					int clsNBPotentialMatchUpLCDP = 0;
					// Number of potential matched RGB model
					int clsNBPotentialRGBMatch = 0;
					// Number of potential matched RGB model with Up threshold
					int clsNBPotentialMatchUpRGB = 0;

					currLastWordPersistence = FLT_MAX;
					currLocalWordIdx = 0;
					while (currLocalWordIdx < WORDS_NO && ((clsNBPotentialLCDPMatch < clsNBMatchThreshold) || (clsNBPotentialRGBMatch < clsNBMatchThreshold))) {
						bgWord = (bgWordPtr + startNBModelIndex + currLocalWordIdx);
						GetLocalWordPersistence(*bgWord, frameIndex, descOffsetValue, currWordPersistence);
						float tempLCDPDistance = 0.0f;
						float tempRGBDistance = 0.0f;

						bool LCDPResult = false;
						bool upLCDPResult = false;
						bool RGBResult = false;
						bool upRGBResult = false;
						// False:Match true:Not match
						DescriptorMatching(*bgWord, currWord, currDescNeighNo, currLCDPThreshold, currUpLCDPThreshold, currRGBThreshold, currUpRGBThreshold,
							tempLCDPDistance, tempRGBDistance, LCDPResult, upLCDPResult, RGBResult, upRGBResult);

						if (((*bgWord).frameCount > 0) && (!LCDPResult)) {
							(*bgWord).frameCount += 1;
							(*bgWord).q = frameIndex;
							clsNBPotentialLCDPMatch++;
						}
						if (((*bgWord).frameCount > 0) && (!RGBResult)) {
							if (LCDPResult) {
								(*bgWord).frameCount += 1;
								(*bgWord).q = frameIndex;
							}
							clsNBPotentialRGBMatch++;
						}
						// Update position of model in background model
						if (currWordPersistence > currLastWordPersistence) {
							std::swap(bgWordPtr[startNBModelIndex + currLocalWordIdx], bgWordPtr[startNBModelIndex + currLocalWordIdx - 1]);
						}
						else
							currLastWordPersistence = currWordPersistence;
						++currLocalWordIdx;
					}
					// Sorting remaining models
					while (currLocalWordIdx < WORDS_NO) {
						bgWord = (bgWordPtr + startNBModelIndex + currLocalWordIdx);
						GetLocalWordPersistence(*bgWord, frameIndex, descOffsetValue, currWordPersistence);
						if (currWordPersistence > currLastWordPersistence) {
							std::swap(bgWordPtr[startNBModelIndex + currLocalWordIdx], bgWordPtr[startNBModelIndex + currLocalWordIdx - 1]);
						}
						else {
							currLastWordPersistence = currWordPersistence;
						}
						++currLocalWordIdx;
					}
				}
			}

			if (upFeedbackSwitch) {
				// Last foreground mask
				uchar * lastFGMask = (uchar*)(resLastFGMask.data + pxPointer);
				// RGB Update
				if (clsRGBDiffSwitch) {
					(*currRGBPxDistance) = ((1 - MIN_DISTANCE_ALPHA)*(*currRGBPxDistance)) + (MIN_DISTANCE_ALPHA*(*minRGBDistance));
					bool check1 = (*lastFGMask) | (((*currRGBPxDistance) < UNSTABLE_REG_RATIO_MIN) && (*currFGMask));
					bool check2 = check1 && ((*currRGBUpdateRate) < upLearningRateUpperCap);
					if (check2) {
						(*currRGBUpdateRate) = std::min(upUpdateRateUpperCap, (*currRGBUpdateRate) +
							(upUpdateRateIncrease / ((*currRGBPxDistance)*(*currRGBDynamicRate))));
					}
					check2 = !check1 && ((*currRGBUpdateRate) >= upLearningRateLowerCap);
					if (check2) {
						(*currRGBUpdateRate) = std::max(upUpdateRateLowerCap, (*currRGBUpdateRate) -
							((upUpdateRateDecrease*(*currRGBDynamicRate)) / (*currRGBPxDistance)));
					}
					check1 = (*currRGBDistThreshold) < (std::pow((1 + ((*currRGBPxDistance) * 2)), 2));
					// FG
					if (check1) {
						(*currRGBDistThreshold) += (*currRGBDynamicRate);
					}
					else {
						(*currRGBDistThreshold) = std::max(1.0f, (*currRGBDistThreshold) - (1.0f / (*currRGBDynamicRate)));
					}
				}


				// LCDP Update
				if (clsLCDPDiffSwitch) {
					(*currLCDPPxDistance) = ((1 - MIN_DISTANCE_ALPHA)*(*currLCDPPxDistance)) + (MIN_DISTANCE_ALPHA*(*minLCDPDistance));
					bool check1 = (*lastFGMask) | (((*currLCDPPxDistance) < UNSTABLE_REG_RATIO_MIN) && (*currFGMask));
					bool check2 = check1 && ((*currLCDPUpdateRate) < upLearningRateUpperCap);
					if (check2) {
						(*currLCDPUpdateRate) = std::min(upUpdateRateUpperCap, (*currLCDPUpdateRate) +
							(upUpdateRateIncrease / ((*currLCDPPxDistance)*(*currLCDPDynamicRate))));
					}
					check2 = !check1 && ((*currLCDPUpdateRate) >= upLearningRateLowerCap);
					if (check2) {
						(*currLCDPUpdateRate) = std::max(upUpdateRateLowerCap, (*currLCDPUpdateRate) -
							((upUpdateRateDecrease*(*currLCDPDynamicRate)) / (*currLCDPPxDistance)));
					}
					check1 = (*currLCDPDistThreshold) < (std::pow((1 + ((*currLCDPPxDistance) * 2)), 2));
					// FG
					if (check1) {
						(*currLCDPDistThreshold) += (*currLCDPDynamicRate);
					}
					else {
						(*currLCDPDistThreshold) = std::max(1.0f, (*currLCDPDistThreshold) - (1.0f / (*currLCDPDynamicRate)));
					}
				}
			}
		}
	}

	// POST PROCESSING
	if (postSwitch) {
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
		cv::inRange(grad, cv::Scalar(75), cv::Scalar(110), gradientResult2);
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
		//cv::bitwise_or(tempCurrFGMask, reconstructLine, resFGMaskPreFlood);
		//cv::bitwise_or(resDarkPixel, resFGMaskPreFlood, resFGMaskPreFlood);
		cv::dilate(resFGMaskPreFlood, resFGMaskPreFlood, cv::Mat(), cv::Point(-1, -1), 3);
		cv::morphologyEx(resFGMaskPreFlood, resFGMaskPreFlood, cv::MORPH_CLOSE, element);
		resFGMaskFloodedHoles = ContourFill(resFGMaskPreFlood);
		//cv::dilate(reconstructLine, reconstructLine, cv::Mat(), cv::Point(-1, -1), 3);
		//cv::bitwise_not(reconstructLine, reconstructLine);
		//cv::bitwise_and(reconstructLine, resFGMaskFloodedHoles, resFGMaskFloodedHoles);
		cv::bitwise_or(resCurrFGMask, resFGMaskFloodedHoles, resCurrFGMask);
		cv::bitwise_or(resCurrFGMask, postCompensationResult, resLastFGMask);
		////cv::dilate(resLastFGMask, resLastFGMask, cv::Mat(), cv::Point(-1, -1), 2);
		cv::morphologyEx(resLastFGMask, resLastFGMask, cv::MORPH_CLOSE, element);
		cv::medianBlur(resLastFGMask, resLastFGMask, postMedianFilterSize);
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
	// Update average image
	resLastImg = (inputImg + (resLastImg*(frameIndex - 1))) / frameIndex;
	resLastGrayImg = (inputGrayImg + (resLastGrayImg*(frameIndex - 1))) / frameIndex;
}

/*=====METHODS=====*/
/*=====DEFAULT methods=====*/
// Refreshes all samples based on the last analyzed frame - checked
void BackgroundSubtractorLCDP::RefreshModel(float refreshFraction)
{
	srand(time(NULL));
	const size_t noSampleBeRefresh = refreshFraction < 1.0f ? (size_t)(refreshFraction*WORDS_NO) : WORDS_NO;
	const size_t refreshStartPos = refreshFraction < 1.0f ? rand() % WORDS_NO : 0;
	for (size_t pxPointer = 0; pxPointer < frameInitTotalPixel; pxPointer++) {
		if (frameRoi.data[pxPointer]) {
			// Start index of the model of the current pixel
			const size_t modelIndex = pxInfoLUTPtr[pxPointer].modelIndex;
			for (size_t currModelIndex = refreshStartPos; currModelIndex < refreshStartPos + noSampleBeRefresh; ++currModelIndex) {

				cv::Point sampleCoor;
				getRandSamplePosition_7x7(sampleCoor, cv::Point(pxInfoLUTPtr[pxPointer].coor_x, pxInfoLUTPtr[pxPointer].coor_y), 0, frameSize);
				const size_t samplePxIndex = (frameSize.width*sampleCoor.y) + sampleCoor.x;
				DescriptorStruct* currWord = (currWordPtr + samplePxIndex);
				// WHY CAN POINT LIKE THIS
				DescriptorStruct * bgWord = (bgWordPtr + modelIndex + currModelIndex);
				for (size_t channel = 0; channel < 3; channel++) {
					(*bgWord).rgb[channel] = (*currWord).rgb[channel];
				}
				for (size_t channel = 0; channel < descNbNo; channel++) {
					(*bgWord).LCDP[channel] = (*currWord).LCDP[channel];
				}
				(*bgWord).frameCount = 1;
				(*bgWord).p = frameIndex;
				(*bgWord).q = frameIndex;
			}
		}
	}
}

/*=====DESCRIPTOR Methods=====*/
// Descriptor Generator-Generate pixels' descriptor (RGB+LCDP)
void BackgroundSubtractorLCDP::DescriptorGenerator(const cv::Mat inputFrame, const PxInfo &pxInfoPtr,
	DescriptorStruct &wordPtr)
{
	wordPtr.frameCount = 1;
	for (int channel = 0; channel < 3; channel++) {
		wordPtr.rgb[channel] = inputFrame.data[pxInfoPtr.bgrDataIndex + channel];
	}
	wordPtr.p = frameIndex;
	wordPtr.q = frameIndex;
	LCDGenerator(inputFrame, pxInfoPtr, wordPtr);
}
// Generate LCD Descriptor
void BackgroundSubtractorLCDP::LCDGenerator(const cv::Mat inputFrame, const PxInfo &pxInfoPtr, DescriptorStruct &wordPtr)
{
	// Current pixel RGB intensity
	int B_CURR = inputFrame.data[pxInfoPtr.bgrDataIndex];
	int G_CURR = inputFrame.data[pxInfoPtr.bgrDataIndex + 1];
	int R_CURR = inputFrame.data[pxInfoPtr.bgrDataIndex + 2];

	// Define Neighbour differences variables
	int R_NB, G_NB, B_NB;
	double ratioBNB_GCURR_MIN, ratioBNB_RCURR_MIN, ratioGNB_BCURR_MIN, ratioGNB_RCURR_MIN, ratioRNB_BCURR_MIN, ratioRNB_GCURR_MIN,
		ratioBNB_GCURR_MAX, ratioBNB_RCURR_MAX, ratioGNB_BCURR_MAX, ratioGNB_RCURR_MAX, ratioRNB_BCURR_MAX, ratioRNB_GCURR_MAX,
		ratioBNB_BCURR_MIN, ratioGNB_GCURR_MIN, ratioRNB_RCURR_MIN, ratioBNB_BCURR_MAX, ratioGNB_GCURR_MAX, ratioRNB_RCURR_MAX;

	//if (!descRatioCalculationMethod) {
	//	// OLD ratio calculation method
	//	// Calculate thresholding ratio
	//	double ratioBNB_GCURR = std::max(3.0, double(*(colorDiffRatio)*G_CURR));
	//	double ratioBNB_RCURR = std::max(3.0, double(*(colorDiffRatio)*R_CURR));
	//	double ratioGNB_BCURR = std::max(3.0, double(*(colorDiffRatio)*B_CURR));
	//	double ratioGNB_RCURR = std::max(3.0, double(*(colorDiffRatio)*R_CURR));
	//	double ratioRNB_BCURR = std::max(3.0, double(*(colorDiffRatio)*B_CURR));
	//	double ratioRNB_GCURR = std::max(3.0, double(*(colorDiffRatio)*G_CURR));

	//	ratioBNB_GCURR_MIN = std::max(-255.0, -ratioBNB_GCURR);
	//	ratioBNB_RCURR_MIN = std::max(-255.0, -ratioBNB_RCURR);
	//	ratioGNB_BCURR_MIN = std::max(-255.0, -ratioGNB_BCURR);
	//	ratioGNB_RCURR_MIN = std::max(-255.0, -ratioGNB_RCURR);
	//	ratioRNB_BCURR_MIN = std::max(-255.0, -ratioRNB_BCURR);
	//	ratioRNB_GCURR_MIN = std::max(-255.0, -ratioRNB_GCURR);

	//	ratioBNB_GCURR_MAX = std::min(255.0, ratioBNB_GCURR);
	//	ratioBNB_RCURR_MAX = std::min(255.0, ratioBNB_RCURR);
	//	ratioGNB_BCURR_MAX = std::min(255.0, ratioGNB_BCURR);
	//	ratioGNB_RCURR_MAX = std::min(255.0, ratioGNB_RCURR);
	//	ratioRNB_BCURR_MAX = std::min(255.0, ratioRNB_BCURR);
	//	ratioRNB_GCURR_MAX = std::min(255.0, ratioRNB_GCURR);

	//	double ratioBNB_BCURR = std::max(3.0, double((*(colorDiffRatio) / 10)*B_CURR));
	//	double ratioGNB_GCURR = std::max(3.0, double((*(colorDiffRatio) / 10)*G_CURR));
	//	double ratioRNB_RCURR = std::max(3.0, double((*(colorDiffRatio) / 10)*R_CURR));

	//	ratioBNB_BCURR_MIN = std::max(-255.0, -ratioBNB_BCURR);
	//	ratioGNB_GCURR_MIN = std::max(-255.0, -ratioGNB_GCURR);
	//	ratioRNB_RCURR_MIN = std::max(-255.0, -ratioRNB_RCURR);

	//	ratioBNB_BCURR_MAX = std::min(255.0, ratioBNB_BCURR);
	//	ratioGNB_GCURR_MAX = std::min(255.0, ratioGNB_GCURR);
	//	ratioRNB_RCURR_MAX = std::min(255.0, ratioRNB_RCURR);
	//}
	//else if (descRatioCalculationMethod) {
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
	//}

	double tempBNB_GCURR, tempBNB_RCURR, tempGNB_BCURR, tempGNB_RCURR, tempRNB_BCURR, tempRNB_GCURR, tempBNB_BCURR,
		tempGNB_GCURR, tempRNB_RCURR;

	for(int nbPixelIndex = 0; nbPixelIndex < descNbNo; nbPixelIndex++) {
		// Obtain neighbourhood pixel's value
		B_NB = inputFrame.data[pxInfoPtr.nbIndex[nbPixelIndex].bgrDataIndex];
		G_NB = inputFrame.data[pxInfoPtr.nbIndex[nbPixelIndex].bgrDataIndex + 1];
		R_NB = inputFrame.data[pxInfoPtr.nbIndex[nbPixelIndex].bgrDataIndex + 2];
		int tempResult = 0;

		// R_NB - R_CURR
		tempRNB_RCURR = R_NB - R_CURR;
		tempResult += ((tempRNB_RCURR > ratioRNB_RCURR_MAX) ? 65536 : ((tempRNB_RCURR < (ratioRNB_RCURR_MIN)) ? 196608 : 0));
		// G_NB - G_CURR
		tempGNB_GCURR = G_NB - G_CURR;
		tempResult += ((tempGNB_GCURR > ratioGNB_GCURR_MAX) ? 16384 : ((tempGNB_GCURR < (ratioGNB_GCURR_MIN)) ? 49152 : 0));
		// B_NB - B_CURR
		tempBNB_BCURR = B_NB - B_CURR;
		tempResult += ((tempBNB_BCURR > ratioBNB_BCURR_MAX) ? 4096 : ((tempBNB_BCURR < (ratioBNB_BCURR_MIN)) ? 12288 : 0));

		// R_NB - G_CURR
		tempRNB_GCURR = R_NB - G_CURR;
		tempResult += ((tempRNB_GCURR > ratioRNB_GCURR_MAX) ? 1024 : ((tempRNB_GCURR < (ratioRNB_GCURR_MIN)) ? 3072 : 0));
		// R_NB - B_CURR
		tempRNB_BCURR = R_NB - B_CURR;
		tempResult += ((tempRNB_BCURR > ratioRNB_BCURR_MAX) ? 256 : ((tempRNB_BCURR < (ratioRNB_BCURR_MIN)) ? 768 : 0));

		// G_NB - R_CURR
		tempGNB_RCURR = G_NB - R_CURR;
		tempResult += ((tempGNB_RCURR > ratioGNB_RCURR_MAX) ? 64 : ((tempGNB_RCURR < (ratioGNB_RCURR_MIN)) ? 192 : 0));
		// G_NB - B_CURR
		tempGNB_BCURR = G_NB - B_CURR;
		tempResult += ((tempGNB_BCURR > ratioGNB_BCURR_MAX) ? 16 : ((tempGNB_BCURR < (ratioGNB_BCURR_MIN)) ? 48 : 0));

		// B_NB - R_CURR
		tempBNB_RCURR = B_NB - R_CURR;
		tempResult += ((tempBNB_RCURR > ratioBNB_RCURR_MAX) ? 4 : ((tempBNB_RCURR < (ratioBNB_RCURR_MIN)) ? 12 : 0));
		// B_NB - G_CURR
		tempBNB_GCURR = B_NB - G_CURR;
		tempResult += ((tempBNB_GCURR > ratioBNB_GCURR_MAX) ? 1 : ((tempBNB_GCURR < (ratioBNB_GCURR_MIN)) ? 3 : 0));

		wordPtr.LCDP[nbPixelIndex] = tempResult;
	}

}
// Calculate word persistence value
void BackgroundSubtractorLCDP::GetLocalWordPersistence(DescriptorStruct &wordPtr, const size_t &currFrameIndex,
	const size_t offsetValue, float &persistenceValue) {
	persistenceValue = (float)(wordPtr.frameCount) / ((wordPtr.q - wordPtr.p) + ((currFrameIndex - wordPtr.q) * 2) + offsetValue);
}

/*=====LUT Methods=====*/
// Generate neighbourhood pixel offset value
void BackgroundSubtractorLCDP::GenerateNbOffset(PxInfo &pxInfoPtr)
{
	const int currX = pxInfoPtr.coor_x;
	const int currY = pxInfoPtr.coor_y;
	for (int nbIndex = 0; nbIndex < (sizeof(nbOffset) / sizeof(cv::Point)); nbIndex++) {

		// Coordinate X value for neighbourhood pixel
		pxInfoPtr.nbIndex[nbIndex].coor_x = std::min(frameSizeZero.width, std::max(0, (currX + nbOffset[nbIndex].x)));
		// Coordinate X value for neighbourhood pixel
		pxInfoPtr.nbIndex[nbIndex].coor_y = std::min(frameSizeZero.height, std::max(0, (currY + nbOffset[nbIndex].y)));
		// Data index for neighbourhood pixel's pointer
		pxInfoPtr.nbIndex[nbIndex].dataIndex = ((pxInfoPtr.nbIndex[nbIndex].coor_y*(frameSize.width)) + (pxInfoPtr.nbIndex[nbIndex].coor_x));
		// Data index for neighbourhood pixel's BGR pointer
		pxInfoPtr.nbIndex[nbIndex].bgrDataIndex = 3 * pxInfoPtr.nbIndex[nbIndex].dataIndex;
		// Model index for neighbourhood pixel
		pxInfoPtr.nbIndex[nbIndex].modelIndex = WORDS_NO * pxInfoPtr.nbIndex[nbIndex].dataIndex;
	}
}
// Generate LCD differences Lookup table (0: 100% Same -> 1: 100% Different)
void BackgroundSubtractorLCDP::GenerateLCDDiffLUT() {
	float countColour = 0;
	float countTexture = 0;
	int tempDiffIndex;
	for (int diffIndex = 0; diffIndex < descPatternLength; diffIndex++) {		
		countColour = 0;
		countTexture = 0;
		tempDiffIndex = diffIndex;
		// Checking bit by bit 
		// Colour variation
		for (size_t bitIndex = 0; bitIndex < 6; bitIndex++) {
			// The second bit is 1 - 11/01
			if ((tempDiffIndex & 1) == 1) {
				countColour += 1;
				tempDiffIndex >>= 2;
			}
			// The second bit is 0 - 00/10
			else {
				tempDiffIndex >>= 1;
				// The first bit is 1 - 10
				if ((tempDiffIndex & 1) == 1)
					countColour += 1;
				tempDiffIndex >>= 1;
			}
		}
		// Texture variation
		for (size_t bitIndex = 0; bitIndex < 3; bitIndex++) {
			// The second bit is 1 - 11/01
			if ((tempDiffIndex & 1) == 1) {
				countTexture += 1;
				tempDiffIndex >>= 2;
			}
			// The second bit is 0 - 00/10
			else {
				tempDiffIndex >>= 1;
				// The first bit is 1 - 10
				if ((tempDiffIndex & 1) == 1)
					countTexture += 1;
				tempDiffIndex >>= 1;
			}
		}
		//if (countTexture > 0) {
		//	LCDDiffLUTPtr[diffIndex] = 1;
		//}
		//else if (countColour > 0) {
		//	LCDDiffLUTPtr[diffIndex] = 1;
		//}
		//else {
		//	LCDDiffLUTPtr[diffIndex] = 0;
		//}
		LCDDiffLUTPtr[diffIndex] = ((countTexture / 3) + (countColour / 6)) / 2;
	}
}

/*=====MATCHING Methods=====*/
// Descriptor matching (RETURN: LCDPResult-1:Not match, 0: Match)
void BackgroundSubtractorLCDP::DescriptorMatching(DescriptorStruct &bgWord, DescriptorStruct &currWord
	, const size_t &descNeighNo, const double LCDPThreshold, const double upLCDPThreshold, const double RGBThreshold, const double upRGBThreshold,
	float &LCDPDistance, float &RGBDistance, bool &LCDPResult, bool &upLCDPResult, bool &RGBResult, bool &upRGBResult)
{
	// Match LCD descriptor
	if (clsLCDPDiffSwitch) {
		// LCD Matching (RETURN: LCDPResult-1:Not match, 0: Match)
		LCDPMatching(bgWord, currWord, descNeighNo, LCDPThreshold, LCDPDistance, LCDPResult);

		// If previous results is not match, matching again with more larger threshold to indicate the pixel exactly belong to FG
		if (LCDPResult) {
			upLCDPResult = (LCDPDistance > upLCDPThreshold) ? true : false;
		}
	}
	// Match RGB descriptor
	if (clsRGBDiffSwitch) {
		// RGB Matching (RETURN: RGBResult-1:Not match, 0: Match)
		RGBMatching(bgWord, currWord, RGBThreshold, RGBDistance, RGBResult);
		// If previous results is not match, matching again with more larger threshold to indicate the pixel exactly belong to FG
		if (RGBResult) {
			upRGBResult = (RGBDistance > upRGBThreshold) ? true : false;
		}
	}
}
// LCD Matching (RETURN-1:Not match, 0: Match)
void BackgroundSubtractorLCDP::LCDPMatching(DescriptorStruct &bgWord, DescriptorStruct &currWord,
	const size_t &descNeighNo, const double &LCDPThreshold, float &minDistance, bool &result) {
	float tempDistance = 0.0f;
	for (size_t neighbourIndex = 0; neighbourIndex < descNeighNo; neighbourIndex++) {
		// Calculate the total number of bit that are different
		tempDistance += LCDDiffLUTPtr[std::abs(bgWord.LCDP[neighbourIndex] - currWord.LCDP[neighbourIndex])];
	}
	minDistance = tempDistance / descNeighNo;
	result = (minDistance > LCDPThreshold) ? true : false;
}
// RGB Matching (RETURN-1:Not match, 0: Match)
void BackgroundSubtractorLCDP::RGBMatching(DescriptorStruct &bgWord, DescriptorStruct &currWord,
	const double &RGBThreshold, float &minDistance, bool &result)
{
	// Maximum of colour differences
	uint distance = 255;
	uint tempDistance = 0;
	for (int channel = 0; channel < 3; channel++) {
		tempDistance = std::abs(bgWord.rgb[channel] - currWord.rgb[channel]);
		// Update minimum distance
		distance = tempDistance < distance ? tempDistance : distance;
		if (tempDistance > RGBThreshold) {
			result = true;
			break;
		}
	}
	minDistance = distance / 255;
}
// Dark Pixel generator (RETURN-255: Not dark pixel, 0: dark pixel)
void BackgroundSubtractorLCDP::DarkPixelGenerator(const cv::Mat &inputGrayImg, const cv::Mat &inputRGBImg,
	const cv::Mat &lastGrayImg, const cv::Mat &lastRGBImg, cv::Mat &darkPixel) {
	darkPixel = cv::Scalar_<uchar>::all(255);
	//cv::Mat debugMat;
	//cv::Mat debugMatG;
	//cv::Mat debugMatGy;
	//debugMat.create(frameSize, CV_32FC1);
	//debugMat = cv::Scalar(0.0f);
	//debugMatG.create(frameSize, CV_32FC1);
	//debugMatG = cv::Scalar(0.0f);
	//debugMatGy.create(frameSize, CV_32FC1);
	//debugMatGy = cv::Scalar(0.0f);
	// Store the pixel's intensity values
	static double IntensityRatio,totalCurrIntensityValue,totalLastIntensityValue, currIntensityValue, lastIntensityValue, currRValue, lastRValue, currGValue, lastGValue, RDiff, GDiff;

	for (size_t pxPointer = 0; pxPointer < frameInitTotalPixel; ++pxPointer) {
		totalCurrIntensityValue = double(inputRGBImg.data[(pxPointer * 3)] + inputRGBImg.data[(pxPointer * 3) + 1] + inputRGBImg.data[(pxPointer * 3) + 2]);
		totalLastIntensityValue = double(lastRGBImg.data[(pxPointer * 3)] + lastRGBImg.data[(pxPointer * 3) + 1] + lastRGBImg.data[(pxPointer * 3) + 2]);
		currIntensityValue = totalCurrIntensityValue / 3.0;
		lastIntensityValue = totalLastIntensityValue / 3.0;

		IntensityRatio = (currIntensityValue / lastIntensityValue);

		if ((IntensityRatio <0.8) && (IntensityRatio > 0.25)) {
			lastRValue = double(lastRGBImg.data[(pxPointer * 3) + 2]) / totalLastIntensityValue;
			currRValue = double(inputRGBImg.data[(pxPointer * 3) + 2]) / totalCurrIntensityValue;
			lastGValue = double(lastRGBImg.data[(pxPointer * 3) + 1]) / totalLastIntensityValue;
			currGValue = double(inputRGBImg.data[(pxPointer * 3) + 1]) / totalCurrIntensityValue;
			RDiff = std::abs(lastRValue - currRValue);
			GDiff = std::abs(lastGValue - currGValue);
			if ((RDiff < 0.15) && (GDiff < 0.1)) {
				darkPixel.data[pxPointer] = 0;
			}
		}
	}
}

/*=====POST-PROCESSING Methods=====*/
// Compensation with Motion Hist - checked
cv::Mat BackgroundSubtractorLCDP::CompensationMotionHist(const cv::Mat T_1FGMask, const cv::Mat T_2FGMask, const cv::Mat currFGMask,
	const float postCompensationThreshold) {
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
cv::Mat BackgroundSubtractorLCDP::ContourFill(const cv::Mat inputImg) {
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
cv::Mat BackgroundSubtractorLCDP::BorderLineReconst(const cv::Mat inputMask)
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
				const uchar currFGMask = *(inputMask.data + pxPointer);

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
	myfile << "\n----VIDEO PARAMETER----\n";
	myfile << "VIDEO WIDTH:";
	myfile << frameSize.width;
	myfile << "\nVIDEO HEIGHT:";
	myfile << frameSize.height;
	myfile << "\n\n----METHOD THRESHOLD----\n<< << <-DESCRIPTOR DEFAULT PARAMETER-> >> >>";
	myfile << "\nTotal number of LCD differences per pixel:";
	myfile << descDiffNo;
	myfile << "\nTotal number of LCD descriptor's neighbour:";
	myfile << descNbNo;
	myfile << "\nLCD colour differences ratio:";
	myfile << descColourDiffRatio;
	myfile << "\nPersistence's offset value:";
	myfile << descOffsetValue;
	myfile << "\n\n<<<<<-CLASSIFIER DEFAULT PARAMETER->>>>>";
	myfile << "\nRGB detection switch:";
	myfile << clsRGBDiffSwitch;
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

	myfile << "\n\n<<<<<-UPDATE DEFAULT PARAMETER->>>>>";
	myfile << "\nRandom replace model switch:";
	myfile << upRandomReplaceSwitch;
	myfile << "\nRandom update neighbourhood model switch:";
	myfile << upRandomUpdateNbSwitch;
	myfile << "\nFeedback loop switch:";
	myfile << upFeedbackSwitch;
	myfile << "\nInitial commonValue, R:";
	myfile << "\nSegmentation noise accumulator increase value:";
	myfile << upDynamicRateIncrease;
	myfile << "\nSegmentation noise accumulator decrease value:";
	myfile << upDynamicRateDecrease;
	myfile << "\nLocal update rate change factor (Desc):";
	myfile << upUpdateRateDecrease;
	myfile << "\nLocal update rate change factor (Incr):";
	myfile << upUpdateRateIncrease;
	myfile << "\nLocal update rate (Lower):";
	myfile << upLearningRateLowerCap;
	myfile << "\nLocal update rate (Upper):";
	myfile << upLearningRateUpperCap;
	myfile << "\nMaximum number of model:";
	myfile << WORDS_NO;

	myfile.close();

	myfile.open(versionFolderName + "/parameter.csv", std::ios::app);
	myfile << "," << frameSize.width << "," << frameSize.height << "," << descDiffNo;
	myfile << "," << descNbNo;
	myfile << "," << descColourDiffRatio << "," << descOffsetValue << "," << clsRGBDiffSwitch;
	myfile << "," << clsLCDPDiffSwitch << "," << clsLCDPThreshold << "," << clsUpLCDPThreshold;
	myfile << "," << clsLCDPMaxThreshold;
	myfile << "," << *(persistenceThreshold) << "," << clsMatchThreshold << ",";
	myfile << clsNbMatchSwitch << "," << upRandomReplaceSwitch;
	myfile << "," << upRandomUpdateNbSwitch;
	myfile << "," << upFeedbackSwitch;
	myfile << "," << upDynamicRateIncrease << "," << upDynamicRateDecrease << "," << upUpdateRateIncrease;
	myfile << "," << upUpdateRateDecrease << "," << upLearningRateLowerCap << "," << upLearningRateUpperCap;
	myfile << "," << WORDS_NO;
	myfile.close();
}