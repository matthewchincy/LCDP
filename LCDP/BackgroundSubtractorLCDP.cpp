#include "BackgroundSubtractorLCDP.h"
#include "RandUtils.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <time.h>

// Parameters used to define 'unstable' regions, based on segm noise/bg dynamics and local dist threshold values
#define UNSTABLE_REG_RATIO_MIN  (0.10f)
#define UNSTABLE_REG_RDIST_MIN  (3.00f)
//// Parameters used to scale dynamic learning rate adjustments  ('T(x)')
//#define FEEDBACK_T_DECR  (0.2500f)
//#define FEEDBACK_T_INCR  (0.5000f)
//#define FEEDBACK_T_LOWER (2.0000f)
//#define FEEDBACK_T_UPPER (256.00f)
//// Parameters used to adjust the variation step size of 'v(x)'
//#define FEEDBACK_V_INCR  (0.50f)
//#define FEEDBACK_V_DECR  (0.40f)
// Parameter used to scale dynamic distance threshold adjustments ('R(x)')
#define FEEDBACK_R_VAR (0.1f)
// Parameter used to average the min distance
#define MIN_DISTANCE_ALPHA (0.01f)

// Local define used to specify the default frame size (320x240 = QVGA) // Checked
#define DEFAULT_FRAME_SIZE cv::Size(320,240)
// Pre-processing gaussian size // Checked
#define PRE_DEFAULT_GAUSSIAN_SIZE cv::Size(9,9)

/*******CONSTRUCTOR*******/ // Checked
BackgroundSubtractorLCDP::BackgroundSubtractorLCDP(std::string folderName, size_t inputWordsNo, bool inputPreSwitch,
	double inputDescColourDiffRatioInit, bool inputDescRatioCalculationMethod, bool inputRGBDiffSwitch,
	double inputRGBThreshold, bool inputRGBBrightPxSwitch, bool inputLCDPDiffSwitch, double inputLCDPThreshold,
	double inputLCDPMaxThreshold, bool inputMatchingMethod, int inputMatchThreshold, bool inputAndOrSwitch, bool inputNbMatchSwitch,
	cv::Mat inputROI, cv::Size inputFrameSize,
	bool inputRandomReplaceSwitch, bool inputRandomUpdateNbSwitch, bool inputFeedbackSwitch,
	float inputDynamicRateIncrease, float inputDynamicRateDecrease, float inputUpdateRateIncrease, float inputUpdateRateDecrease,
	float inputUpdateRateLowest, float inputUpdateRateHighest, bool inputPostSwitch) :
	folderName(folderName),
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
	// Size of gaussian filter
	preGaussianSize(PRE_DEFAULT_GAUSSIAN_SIZE),
	// Pre processing switch
	preSwitch(inputPreSwitch),

	/*=====DESCRIPTOR Parameters=====*/
	// LCD colour differences ratio initial
	descColourDiffRatioInit(inputDescColourDiffRatioInit),
	// Total number of differences per descriptor
	descDiffNo(9),
	// Persistence's offset value;
	descOffsetValue(1000),
	// Ratio calculation method
	descRatioCalculationMethod(inputDescRatioCalculationMethod),

	/*=====CLASSIFIER Parameters=====*/
	// RGB detection switch
	clsRGBDiffSwitch(inputRGBDiffSwitch),
	// RGB differences threshold
	clsRGBThreshold(inputRGBThreshold),
	// RGB bright pixel switch
	clsRGBBrightPxSwitch(inputRGBBrightPxSwitch),
	// LCDP detection switch
	clsLCDPDiffSwitch(inputLCDPDiffSwitch),
	// LCDP differences threshold
	clsLCDPThreshold(inputLCDPThreshold),
	// Maximum number of LCDP differences threshold
	clsLCDPMaxThreshold(inputLCDPMaxThreshold),
	// LCDP detection AND (true) OR (false) switch
	clsAndOrSwitch(inputAndOrSwitch),
	// Neighbourhood matching switch
	clsNbMatchSwitch(inputNbMatchSwitch),
	// Classify method
	clsMatchingMethod(inputMatchingMethod),
	// Matching threshold
	clsMatchThreshold(inputMatchThreshold),

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
	upRandomReplaceSwitch(inputRandomReplaceSwitch),
	// Random update neighbourhood model switch
	upRandomUpdateNbSwitch(inputRandomUpdateNbSwitch),
	// Feedback loop switch
	upFeedbackSwitch(inputFeedbackSwitch),
	// Feedback V(x) Increment
	upDynamicRateIncrease(inputDynamicRateIncrease),
	// Feedback V(x) Decrement
	upDynamicRateDecrease(inputDynamicRateDecrease),
	// Feedback T(x) Increment
	upUpdateRateIncrease(inputUpdateRateIncrease),
	// Feedback T(x) Decrement
	upUpdateRateDecrease(inputUpdateRateDecrease),
	// Feedback T(x) Lowest
	upUpdateRateLowest(inputUpdateRateLowest),
	// Feedback T(x) Highest
	upUpdateRateHighest(inputUpdateRateHighest),

	/*====POST Parameters=====*/
	// Post processing switch
	postSwitch(inputPostSwitch)
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
	// Total length of descriptor pattern
	const int totalDescPatternLength = std::pow(2, (descDiffNo * 2));
	/*=====LOOK-UP TABLE=====*/
	// Internal pixel info LUT for all possible pixel indexes
	pxInfoLUTPtr = new PxInfo[frameInitTotalPixel];
	memset(pxInfoLUTPtr, 0, sizeof(PxInfo)*frameInitTotalPixel);
	// LCD differences LUT
	LCDDiffLUTPtr = new float[totalDescPatternLength];
	memset(LCDDiffLUTPtr, 0, sizeof(float)*totalDescPatternLength);
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

	/*=====DESCRIPTOR Parameters=====*/
	// Size of neighbourhood 3(3x3)/5(5x5)
	descNbSize.create(frameSize, CV_8UC1);
	descNbSize = cv::Scalar_<uchar>(5);
	// Total number of neighbourhood pixel 8(3x3)/16(5x5)
	descNbNo.create(frameSize, CV_8UC1);
	descNbNo = cv::Scalar_<uchar>(16);
	// LCD colour differences ratio
	descColorDiffRatio.create(frameSize, CV_32FC1);
	descColorDiffRatio = cv::Scalar(descColourDiffRatioInit);

	/*=====CLASSIFIER Parameters=====*/
	// Total number of neighbour 0(0)/8(3x3)/16(5x5)
	clsNbNo.create(frameSize, CV_8UC1);
	clsNbNo = cv::Scalar_<uchar>(0);
	// Minimum persistence threhsold value 
	clsMinPersistenceThreshold = (1.0f / descOffsetValue) * 2;
	// Matched persistence value threshold
	clsPersistenceThreshold.create(frameSize, CV_32FC1);
	clsPersistenceThreshold = cv::Scalar(clsMinPersistenceThreshold);

	/*=====POST-PROCESS Parameters=====*/
	// Size of median filter
	postMedianFilterSize = 9;
	// The compensation motion history threshold
	postCompensationThreshold.create(frameSize, CV_32FC1);
	postCompensationThreshold = cv::Scalar(0.7f);

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
		upLearningRateLowerCap = upUpdateRateLowest;
		upLearningRateUpperCap = upUpdateRateHighest;
	}
	else {
		/*=====POST-PROCESS Parameters=====*/
		// Current kernel size for median blur post-proc filtering
		postMedianFilterSize = postMedianFilterSize;
		// Specifies the px update spread range
		upUse3x3Spread = true;
		// Current learning rate caps
		upLearningRateLowerCap = upUpdateRateLowest * 2;
		upLearningRateUpperCap = upUpdateRateHighest * 2;
	}

	/*=====UPDATE Parameters=====*/
	// Total number of neighbour undergo neighbourhood updates 8(3X3)/16(5X5)
	upNbNo.create(frameSize, CV_8UC1);
	upNbNo = cv::Scalar_<uchar>(16);

	/*=====RESULTS=====*/
	// Per-pixel distance thresholds ('R(x)', but used as a relative value to determine both 
	// intensity and descriptor variation thresholds)
	resDistThreshold.create(frameSize, CV_32FC1);
	resDistThreshold = cv::Scalar(1.0f);
	resDynamicRate.create(frameSize, CV_32FC1);
	resDynamicRate = cv::Scalar(1.0f);
	// Per-pixel update rates('T(x)', which contains pixel - level 'sigmas')
	resUpdateRate.create(frameSize, CV_32FC1);
	resUpdateRate = cv::Scalar(upLearningRateLowerCap);
	// Minimum LCDP distance
	resMinLCDPDistance.create(frameSize, CV_32FC1);
	resMinLCDPDistance = cv::Scalar(1.0f);
	// Minimum RGB distance
	resMinRGBDistance.create(frameSize, CV_32FC1);
	resMinRGBDistance = cv::Scalar(1.0f);
	// Current pixel distance
	resCurrPxDistance.create(frameSize, CV_32FC1);
	resCurrPxDistance = cv::Scalar(0.0f);
	// Current foreground mask
	resCurrFGMask.create(frameSize, CV_8UC1);
	resCurrFGMask = cv::Scalar_<uchar>::all(0);
	// Previous foreground mask
	resLastFGMask.create(frameSize, CV_8UC1);
	resLastFGMask = cv::Scalar_<uchar>::all(0);
	// The foreground mask generated by the method at [t-1] (without post-proc, used for blinking px detection)
	resLastRawFGMask.create(frameSize, CV_8UC1);
	resLastRawFGMask = cv::Scalar_<uchar>::all(0);
	// t-1 foreground mask
	resT_1FGMask.create(frameSize, CV_8UC1);
	resT_1FGMask = cv::Scalar_<uchar>::all(255);
	// t-2 foreground mask
	resT_2FGMask.create(frameSize, CV_8UC1);
	resT_2FGMask = cv::Scalar_<uchar>::all(255);
	// Flooded holes foreground mask
	//resFGMaskFloodedHoles.create(frameSize, CV_8UC1);
	resFGMaskFloodedHoles.create(frameSize, CV_8UC1);
	resFGMaskFloodedHoles = cv::Scalar_<uchar>::all(0);
	// Pre flooded holes foreground mask
	resFGMaskPreFlood.create(frameSize, CV_8UC1);
	resFGMaskPreFlood = cv::Scalar_<uchar>::all(0);
	// Dark Pixel
	resDarkPixel.create(frameSize, CV_8UC1);
	resDarkPixel = cv::Scalar_<uchar>::all(0);
	// Last frame image
	inputFrame.copyTo(resLastImg);

	// Pixel pointer index
	size_t pxPointer = 0;
	// Model pointer index
	size_t modelPointer = 0;
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
			pxInfoLUTPtr[pxPointer].modelIndex = modelPointer*WORDS_NO;
			++modelPointer;
			/*=====LUT Methods=====*/
			// Generate neighbourhood pixel offset value
			GenerateNbOffset(&pxInfoLUTPtr[pxPointer]);
			// Descripto Generator-Generate pixels' descriptor (RGB+LCDP)
			DescriptorGenerator(inputFrame, &pxInfoLUTPtr[pxPointer], &currWordPtr[pxPointer]);
			pxPointer++;
		}
	}

	// Refresh model
	RefreshModel(1.0f, 0);
}

// Program processing
void BackgroundSubtractorLCDP::Process(const cv::Mat inputImg, cv::Mat &outputImg)
{
	cv::Mat inputImgGray;
	cv::cvtColor(inputImg, inputImgGray, CV_RGB2GRAY);
	// PRE PROCESSING
	if (preSwitch) {
		cv::GaussianBlur(inputImg, inputImg, preGaussianSize, 0, 0);
	}

	// DETECTION PROCESS
	for (size_t pxPointer = 0; pxPointer < frameInitTotalPixel; ++pxPointer) {
		if (frameRoi.data[pxPointer]) {
			// Descriptor Generator-Generate pixels' descriptor (RGB+LCDP)
			DescriptorGenerator(inputImg, &pxInfoLUTPtr[pxPointer], &currWordPtr[pxPointer]);
			// Current distance threshold
			float * currDistThreshold = (float*)(resDistThreshold.data + (pxPointer * 4));
			// Current dynamic learning rate ('V(x)')
			float * currDynamicRate = (float*)(resDynamicRate.data + (pxPointer * 4));
			// Current pixel's update rate ('T(x)')
			float * currUpdateRate = (float*)(resUpdateRate.data + (pxPointer * 4));
			// Model index for current pixel
			const size_t currModelIndex = pxInfoLUTPtr[pxPointer].modelIndex;

			// LCD descriptor threshold
			const double currLCDPThreshold = std::min(clsLCDPMaxThreshold, std::max(clsLCDPThreshold, (std::pow(2, double((*currDistThreshold))) / 512)));
			// RGB descriptor threshold
			const double currRGBThreshold = std::max(clsRGBThreshold, floor(clsRGBThreshold*(*currDistThreshold)));
			// Current pixel's foreground mask
			uchar * currFGMask = (resCurrFGMask.data + pxPointer);
			// Current dark pixel mask
			uchar * currDKMask = (resDarkPixel.data + pxPointer);
			// Current pixel's descriptor
			DescriptorStruct* currWord = &currWordPtr[pxPointer];
			// Potential persistence value accumulator
			float currPotentialPersistenceSum = 0.0f;

			// Current pixel's min LCDP distance
			float * minLCDPDistance = (float*)(resMinLCDPDistance.data + (pxPointer * 4));
			// Current pixel's min RGB distance
			float * minRGBDistance = (float*)(resMinRGBDistance.data + (pxPointer * 4));
			// Current pixel distance
			float * currPxDistance = (float*)(resCurrPxDistance.data + (pxPointer * 4));

			// Last word's persistence
			float currLastWordPersistence = FLT_MAX;
			// Current pixel's background word index
			int currLocalWordIdx = 0;
			bool darkPixel = false;
			bool rgbMatchPixel = false;
			if (!clsMatchingMethod) {
				// Old Classify method (PAWCS)		
				// Current pixel's persistence threshold
				float * currPersistenceThreshold = (float*)(clsPersistenceThreshold.data + (pxPointer * 4));

				while (currLocalWordIdx < WORDS_NO && currPotentialPersistenceSum < (*currPersistenceThreshold)) {
					// Current bg word
					DescriptorStruct* bgWord = (bgWordPtr + currModelIndex + currLocalWordIdx);
					// Current bg word's persistence
					const float currWordPersistence = GetLocalWordPersistence(bgWord, frameIndex, descOffsetValue);
					float tempMatchLCDPDistance = 0.0f;
					float tempMatchRGBDistance = 0.0f;
					bool tempDarkPixel = false;
					bool tempRgbMatchPixel = false;
					if (((*bgWord).frameCount > 0)
						&& (!DescriptorMatching(bgWord, currWord, pxPointer, currLCDPThreshold, currRGBThreshold,
							tempMatchLCDPDistance, tempMatchRGBDistance, tempRgbMatchPixel, tempDarkPixel))) {
						const float randNumber = ((double)std::rand() / (RAND_MAX));
						if ((randNumber <= (1 / (*currUpdateRate))) && (tempMatchLCDPDistance < (currLCDPThreshold / 2))) {
							for (size_t channel = 0; channel < 3; channel++) {
								(*bgWord).rgb[channel] = (*currWord).rgb[channel];
							}
							for (size_t channel = 0; channel < 16; channel++) {
								(*bgWord).LCDP[channel] = (*currWord).LCDP[channel];
							}
						}
						(*bgWord).frameCount += 1;
						(*bgWord).q = frameIndex;
						currPotentialPersistenceSum += currWordPersistence;
					}
					else {
						if (tempRgbMatchPixel) {
							rgbMatchPixel = true;
						}
						else if (tempDarkPixel) {
							darkPixel = true;
						}
					}
					// Update MIN LCDP distance
					(*minLCDPDistance) = tempMatchLCDPDistance < (*minLCDPDistance) ? tempMatchLCDPDistance : (*minLCDPDistance);
					// Update MIN RGB distance
					(*minRGBDistance) = tempMatchRGBDistance < (*minRGBDistance) ? tempMatchRGBDistance : (*minRGBDistance);
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
					DescriptorStruct* bgWord = (bgWordPtr + currModelIndex + currLocalWordIdx);
					const float currWordPersistence = GetLocalWordPersistence(bgWord, frameIndex, descOffsetValue);
					if (currWordPersistence > currLastWordPersistence) {
						std::swap(bgWordPtr[currModelIndex + currLocalWordIdx], bgWordPtr[currModelIndex + currLocalWordIdx - 1]);
					}
					else {
						currLastWordPersistence = currWordPersistence;
					}
					++currLocalWordIdx;
				}
				// Successful classified as BG Pixels
				if (currPotentialPersistenceSum >= (*currPersistenceThreshold)) {
					(*currFGMask) = 0;
					(*currDynamicRate) = std::max(1.0f, (*currDynamicRate) - upDynamicRateDecrease);
				}
				// Classified as FG Pixels
				else {
					(*currFGMask) = 255;
					if (clsNbMatchSwitch) {
						// Compare with neighbour's model
						// Neighbour matching size (Max: 7x7)
						size_t nbMatchNo = std::floor((((*currDistThreshold) / 9) * 48));

						for (size_t nbIndex = 0; nbIndex < nbMatchNo; nbIndex++) {
							// Neighbour pixel pointer
							size_t nbPxPointer = pxInfoLUTPtr[pxPointer].nbIndex[nbIndex].dataIndex;
							// Current neighbour pixel's persistence threshold
							float * nbPersistenceThreshold = (float*)(clsPersistenceThreshold.data + (nbPxPointer * 4));
							// Neighbour pixel's model index
							const size_t nbModelIndex = pxInfoLUTPtr[pxPointer].nbIndex[nbIndex].modelIndex;
							// Current neighbour distance threshold
							float * nbDistThreshold = (float*)(resDistThreshold.data + (nbPxPointer * 4));
							// Neighbour LCD descriptor threshold
							const double nbLCDPThreshold = std::min(clsLCDPMaxThreshold, std::max(clsLCDPThreshold, (std::pow(2, double((*nbDistThreshold))) / 512)));
							// Neighbour RGB descriptor threshold
							const double nbRGBThreshold = std::max(clsRGBThreshold, floor(clsRGBThreshold*(*nbDistThreshold)));

							float nbPotentialPersistenceSum = 0.0f;
							float nbLastWordPersistence = FLT_MAX;
							size_t nbLocalWordIdx = 0;
							bool darkPixel = false;
							bool rgbMatchPixel = false;
							while (nbLocalWordIdx < WORDS_NO && nbPotentialPersistenceSum < (*nbPersistenceThreshold)) {
								DescriptorStruct* nbWord = (bgWordPtr + nbModelIndex + nbLocalWordIdx);
								const float currWordPersistence = GetLocalWordPersistence(nbWord, frameIndex, descOffsetValue);
								float tempMatchLCDPDistance = 0.0f;
								float tempMatchRGBDistance = 0.0f;
								bool tempDarkPixel = false;
								bool tempRgbMatchPixel = false;
								if (((*nbWord).frameCount > 0)
									&& (!DescriptorMatching(nbWord, currWord, nbPxPointer, nbLCDPThreshold, nbRGBThreshold,
										tempMatchLCDPDistance, tempMatchRGBDistance, tempRgbMatchPixel, tempDarkPixel))) {
									(*nbWord).frameCount += 1;
									(*nbWord).q = frameIndex;
									nbPotentialPersistenceSum += currWordPersistence;
								}
								if (tempRgbMatchPixel) {
									rgbMatchPixel = true;
								}
								else if (tempDarkPixel) {
									darkPixel = true;
								}

								// Update position of model in background model
								if (currWordPersistence > nbLastWordPersistence) {
									std::swap(bgWordPtr[nbModelIndex + nbLocalWordIdx], bgWordPtr[nbModelIndex + nbLocalWordIdx - 1]);
								}
								else
									nbLastWordPersistence = currWordPersistence;
								++nbLocalWordIdx;
							}
							// BG Pixels
							if (nbPotentialPersistenceSum >= (*nbPersistenceThreshold)) {
								(*currFGMask) = 0;
								(*currDynamicRate) += upDynamicRateIncrease;
								break;
							}
						}
					}
					// BG Pixels
					/*if (!(*currFGMask)) {
						(*currDKMask) = (rgbMatchPixel | darkPixel) ? 255 : 0;
					}*/
				}
				// UPDATE PROCESS
				// Replace current frame's descriptor with the model that having lowest persistence value among others
				// - only applicable for no match pixel (Potential persistence less than init weight)
				if ((*currFGMask)) {
					if (currPotentialPersistenceSum < clsMinPersistenceThreshold) {
						DescriptorStruct* bgWord = (bgWordPtr + currModelIndex + WORDS_NO - 1);
						(*bgWord).frameCount = (*currWord).frameCount;
						(*bgWord).p = (*currWord).p;
						(*bgWord).q = (*currWord).q;
						for (size_t channel = 0; channel < 3; channel++) {
							(*bgWord).rgb[channel] = (*currWord).rgb[channel];
						}
						for (size_t channel = 0; channel < 16; channel++) {
							(*bgWord).LCDP[channel] = (*currWord).LCDP[channel];
						}
					}
				}
				// RESORTING BACKGROUND MODEL
				// Last word's persistence weight
				currLastWordPersistence = FLT_MAX;
				currLocalWordIdx = WORDS_NO - 1;
				while (currLocalWordIdx >= 0) {
					DescriptorStruct* bgWord = (bgWordPtr + currModelIndex + currLocalWordIdx);
					const float currWordPersistence = GetLocalWordPersistence(bgWord, frameIndex, descOffsetValue);
					if (currWordPersistence < currLastWordPersistence) {
						std::swap(bgWordPtr[currModelIndex + currLocalWordIdx], bgWordPtr[currModelIndex + currLocalWordIdx + 1]);
					}
					else if (currLastWordPersistence == FLT_MAX) {
						currLastWordPersistence = currWordPersistence;
					}
					else {
						break;
					}
					--currLocalWordIdx;
				}


			}
			else if (clsMatchingMethod) {
				// New Classify method	
				// Number of potential matched model
				int clsPotentialMatch = 0;
				while (currLocalWordIdx < WORDS_NO && clsPotentialMatch < clsMatchThreshold) {
					// Current bg word
					DescriptorStruct* bgWord = (bgWordPtr + currModelIndex + currLocalWordIdx);
					// Current bg word's persistence
					const float currWordPersistence = GetLocalWordPersistence(bgWord, frameIndex, descOffsetValue);
					float tempMatchLCDPDistance = 0.0f;
					float tempMatchRGBDistance = 0.0f;
					bool tempDarkPixel = false;
					bool tempRgbMatchPixel = false;
					if (((*bgWord).frameCount > 0)
						&& (!DescriptorMatching(bgWord, currWord, pxPointer, currLCDPThreshold, currRGBThreshold,
							tempMatchLCDPDistance, tempMatchRGBDistance, tempRgbMatchPixel, tempDarkPixel))) {
						(*bgWord).frameCount += 1;
						(*bgWord).q = frameIndex;
						clsPotentialMatch++;
					}
					else {
						if (tempRgbMatchPixel) {
							rgbMatchPixel = true;
						}
						else if (tempDarkPixel) {
							darkPixel = true;
						}
					}
					// Update MIN LCDP distance
					(*minLCDPDistance) = tempMatchLCDPDistance < (*minLCDPDistance) ? tempMatchLCDPDistance : (*minLCDPDistance);
					// Update MIN RGB distance
					(*minRGBDistance) = tempMatchRGBDistance < (*minRGBDistance) ? tempMatchRGBDistance : (*minRGBDistance);
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
					DescriptorStruct* bgWord = (bgWordPtr + currModelIndex + currLocalWordIdx);
					const float currWordPersistence = GetLocalWordPersistence(bgWord, frameIndex, descOffsetValue);
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
					(*currDynamicRate) = std::max(1.0f, (*currDynamicRate) - upDynamicRateDecrease);
				}
				// Classified as FG Pixels
				else {
					(*currFGMask) = 255;
					if (clsNbMatchSwitch) {
						// Compare with neighbour's model
						// Neighbour matching size (Max: 7x7)
						size_t nbMatchNo = std::floor((((*currDistThreshold) / 9) * 48));

						for (size_t nbIndex = 0; nbIndex < nbMatchNo; nbIndex++) {
							// Neighbour pixel pointer
							size_t nbPxPointer = pxInfoLUTPtr[pxPointer].nbIndex[nbIndex].dataIndex;
							// Current neighbour pixel's matching threshold
							int clsNBMatchThreshold = clsMatchThreshold;
							// Number of potential matched model
							int clsNBPotentialMatch = 0;
							// Neighbour pixel's model index
							const size_t nbModelIndex = pxInfoLUTPtr[pxPointer].nbIndex[nbIndex].modelIndex;
							// Current neighbour distance threshold
							float * nbDistThreshold = (float*)(resDistThreshold.data + (nbPxPointer * 4));
							// Neighbour LCD descriptor threshold
							const double nbLCDPThreshold = std::min(clsLCDPMaxThreshold, std::max(clsLCDPThreshold, (std::pow(2, double((*nbDistThreshold))) / 512)));
							// Neighbour RGB descriptor threshold
							const double nbRGBThreshold = std::max(clsRGBThreshold, floor(clsRGBThreshold*(*nbDistThreshold)));

							float nbPotentialPersistenceSum = 0.0f;
							float nbLastWordPersistence = FLT_MAX;
							size_t nbLocalWordIdx = 0;
							bool darkPixel = false;
							bool rgbMatchPixel = false;
							while (nbLocalWordIdx < WORDS_NO && clsNBPotentialMatch < clsNBMatchThreshold) {
								DescriptorStruct* nbWord = (bgWordPtr + nbModelIndex + nbLocalWordIdx);
								const float currWordPersistence = GetLocalWordPersistence(nbWord, frameIndex, descOffsetValue);
								float tempMatchLCDPDistance = 0.0f;
								float tempMatchRGBDistance = 0.0f;
								bool tempDarkPixel = false;
								bool tempRgbMatchPixel = false;
								if (((*nbWord).frameCount > 0)
									&& (!DescriptorMatching(nbWord, currWord, nbPxPointer, nbLCDPThreshold, nbRGBThreshold,
										tempMatchLCDPDistance, tempMatchRGBDistance, tempRgbMatchPixel, tempDarkPixel))) {
									(*nbWord).frameCount += 1;
									(*nbWord).q = frameIndex;
									nbPotentialPersistenceSum += currWordPersistence;
								}
								if (tempRgbMatchPixel) {
									rgbMatchPixel = true;
								}
								else if (tempDarkPixel) {
									darkPixel = true;
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
								DescriptorStruct* nbWord = (bgWordPtr + nbModelIndex + nbLocalWordIdx);
								const float currWordPersistence = GetLocalWordPersistence(nbWord, frameIndex, descOffsetValue);
								if (currWordPersistence > nbLastWordPersistence) {
									std::swap(bgWordPtr[nbModelIndex + nbLocalWordIdx], bgWordPtr[nbModelIndex + nbLocalWordIdx - 1]);
								}
								else {
									nbLastWordPersistence = currWordPersistence;
								}
								++nbLocalWordIdx;
							}
							// BG Pixels
							if (clsNBPotentialMatch >= clsNBMatchThreshold) {
								(*currFGMask) = 0;
								(*currDynamicRate) += upDynamicRateIncrease;
								break;
							}
						}
					}
					//// BG Pixels
					//if (!(*currFGMask)) {
					//	(*currDKMask) = (rgbMatchPixel | darkPixel) ? 255 : 0;
					//}
				}
				// UPDATE PROCESS
				// Random replace current frame's descriptor with the model
				if (upRandomReplaceSwitch) {
					const float randNumber = ((double)std::rand() / (RAND_MAX));
					
					bool checkTemp = ((1 / (*currUpdateRate)) >= randNumber);

					if (checkTemp) {
						if ((*currFGMask)) {
							// FG
							const float randNumber2 = ((double)std::rand() / (RAND_MAX));
							bool checkTemp2 = ((1 / (*currUpdateRate)) >= randNumber2);
							if (checkTemp2) {								
								DescriptorStruct* bgWord = (bgWordPtr + currModelIndex + WORDS_NO - 1);
								for (size_t channel = 0; channel < 3; channel++) {
									(*bgWord).rgb[channel] = (*currWord).rgb[channel];
								}
								for (size_t channel = 0; channel < 16; channel++) {
									(*bgWord).LCDP[channel] = (*currWord).LCDP[channel];
								}
							}							
						}
						else {
							// BG
							int randNum = rand() % WORDS_NO;
							DescriptorStruct* bgWord = (bgWordPtr + currModelIndex + randNum);
							for (size_t channel = 0; channel < 3; channel++) {
								(*bgWord).rgb[channel] = (*currWord).rgb[channel];
							}
							for (size_t channel = 0; channel < 16; channel++) {
								(*bgWord).LCDP[channel] = (*currWord).LCDP[channel];
							}
						}

					}
				}
			}
			// UPDATE PROCESS
			// Randomly update a selected neighbour descriptor with current descriptor
			if (upRandomUpdateNbSwitch) {
				const float randNumber = ((double)std::rand() / (RAND_MAX));
				const bool check1 = ((1 / *(currUpdateRate)) >= randNumber);
				if (check1) {
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

					if (!clsMatchingMethod) {
						// Current pixel's persistence threshold
						float * currPersistenceThreshold = (float*)(clsPersistenceThreshold.data + (pxPointer * 4));
						float potentialNBPersistenceSum = 0.0f;
						currLastWordPersistence = FLT_MAX;
						currLocalWordIdx = 0;
						bool * matchIndex = new bool[WORDS_NO];
						while (currLocalWordIdx < WORDS_NO && potentialNBPersistenceSum < (*currPersistenceThreshold)) {
							DescriptorStruct* bgWord = (bgWordPtr + startNBModelIndex + currLocalWordIdx);
							const float currWordPersistence = GetLocalWordPersistence(bgWord, frameIndex, descOffsetValue);
							float tempMatchLCDPDistance = 0.0f;
							float tempMatchRGBDistance = 0.0f;
							bool darkPixel;
							bool rgbMatchPixel = false;
							if (((*bgWord).frameCount > 0)
								&& (!DescriptorMatching(bgWord, currWord, pxPointer, currLCDPThreshold, currRGBThreshold,
									tempMatchLCDPDistance, tempMatchRGBDistance, rgbMatchPixel, darkPixel))) {
								*(matchIndex + currLocalWordIdx) = true;
								if (currWordPersistence < 1) {
									(*bgWord).frameCount += 1;
								}
								(*bgWord).q = frameIndex;
								potentialNBPersistenceSum += currWordPersistence;
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
							DescriptorStruct* bgWord = (bgWordPtr + startNBModelIndex + currLocalWordIdx);
							const float currWordPersistence = GetLocalWordPersistence(bgWord, frameIndex, descOffsetValue);
							if (currWordPersistence > currLastWordPersistence) {
								std::swap(bgWordPtr[startNBModelIndex + currLocalWordIdx], bgWordPtr[startNBModelIndex + currLocalWordIdx - 1]);
							}
							else {
								currLastWordPersistence = currWordPersistence;
							}
							++currLocalWordIdx;
						}
						if (potentialNBPersistenceSum < clsMinPersistenceThreshold) {
							DescriptorStruct* bgWord = (bgWordPtr + startNBModelIndex + WORDS_NO - 1);
							(*bgWord).frameCount = (*currWord).frameCount;
							(*bgWord).p = (*currWord).p;
							(*bgWord).q = (*currWord).q;
							for (size_t channel = 0; channel < 3; channel++) {
								(*bgWord).rgb[channel] = (*currWord).rgb[channel];
							}
							for (size_t channel = 0; channel < 16; channel++) {
								(*bgWord).LCDP[channel] = (*currWord).LCDP[channel];
							}
						}
						delete matchIndex;
					}
					else if (clsMatchingMethod) {
						// Current neighbour pixel's matching threshold
						int clsNBMatchThreshold = clsMatchThreshold;
						// Number of potential matched model
						int clsNBPotentialMatch = 0;
						currLastWordPersistence = FLT_MAX;
						currLocalWordIdx = 0;
						bool * matchIndex = new bool[WORDS_NO];
						while (currLocalWordIdx < WORDS_NO && clsNBPotentialMatch < clsNBMatchThreshold) {
							DescriptorStruct* nbBgWord = (bgWordPtr + startNBModelIndex + currLocalWordIdx);
							const float currWordPersistence = GetLocalWordPersistence(nbBgWord, frameIndex, descOffsetValue);
							float tempMatchLCDPDistance = 0.0f;
							float tempMatchRGBDistance = 0.0f;
							bool darkPixel;
							bool rgbMatchPixel = false;
							if (((*nbBgWord).frameCount > 0)
								&& (!DescriptorMatching(nbBgWord, currWord, pxPointer, currLCDPThreshold, currRGBThreshold,
									tempMatchLCDPDistance, tempMatchRGBDistance, rgbMatchPixel, darkPixel))) {
								*(matchIndex + currLocalWordIdx) = true;
								if (currWordPersistence < 1) {
									(*nbBgWord).frameCount += 1;
								}
								(*nbBgWord).q = frameIndex;
								clsNBPotentialMatch++;
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
							DescriptorStruct* bgWord = (bgWordPtr + startNBModelIndex + currLocalWordIdx);
							const float currWordPersistence = GetLocalWordPersistence(bgWord, frameIndex, descOffsetValue);
							if (currWordPersistence > currLastWordPersistence) {
								std::swap(bgWordPtr[startNBModelIndex + currLocalWordIdx], bgWordPtr[startNBModelIndex + currLocalWordIdx - 1]);
							}
							else {
								currLastWordPersistence = currWordPersistence;
							}
							++currLocalWordIdx;
						}
						delete matchIndex;
						if (clsNBPotentialMatch < clsNBMatchThreshold) {
							if ((*currFGMask)) {
								// FG
								//const float randNumber2 = ((double)std::rand() / (RAND_MAX / 2));
								//bool checkTemp2 = ((1 / (*currUpdateRate)) >= randNumber2);
								//if (checkTemp2) {
									DescriptorStruct* bgWord = (bgWordPtr + startNBModelIndex + WORDS_NO - 1);
									(*bgWord).frameCount = (*currWord).frameCount;
									(*bgWord).p = (*currWord).p;
									(*bgWord).q = (*currWord).q;
									for (size_t channel = 0; channel < 3; channel++) {
										(*bgWord).rgb[channel] = (*currWord).rgb[channel];
									}
									for (size_t channel = 0; channel < 16; channel++) {
										(*bgWord).LCDP[channel] = (*currWord).LCDP[channel];
									}
								//}
							}
							else {
								DescriptorStruct* bgWord = (bgWordPtr + startNBModelIndex + WORDS_NO - 1);
								(*bgWord).frameCount = (*currWord).frameCount;
								(*bgWord).p = (*currWord).p;
								(*bgWord).q = (*currWord).q;
								for (size_t channel = 0; channel < 3; channel++) {
									(*bgWord).rgb[channel] = (*currWord).rgb[channel];
								}
								for (size_t channel = 0; channel < 16; channel++) {
									(*bgWord).LCDP[channel] = (*currWord).LCDP[channel];
								}
							}							
						}
					}
				}
			}
			if (!clsMatchingMethod) {
				// Current pixel's persistence threshold
				float * currPersistenceThreshold = (float*)(clsPersistenceThreshold.data + (pxPointer * 4));
				// Update persistence threshold
				const float highestPersistence = GetLocalWordPersistence((bgWordPtr + currModelIndex), frameIndex, descOffsetValue);
				*(currPersistenceThreshold) = std::min(1.0f, std::max(clsMinPersistenceThreshold, highestPersistence));
			}

			if (upFeedbackSwitch) {
				// Last foreground mask
				uchar * lastFGMask = (uchar*)(resLastFGMask.data + pxPointer);
				if (!clsMatchingMethod) {
					// Current pixel's persistence threshold
					float * currPersistenceThreshold = (float*)(clsPersistenceThreshold.data + (pxPointer * 4));
					// FG
					if (*(currFGMask)) {
						(*currPxDistance) = ((1 - MIN_DISTANCE_ALPHA)*(*currPxDistance)) + ((MIN_DISTANCE_ALPHA)*std::max(((*currPersistenceThreshold) - currPotentialPersistenceSum) / (*currPersistenceThreshold), (*minLCDPDistance)));
					}
					// BG
					else {
						(*currPxDistance) = ((1 - MIN_DISTANCE_ALPHA)*(*currPxDistance)) + (MIN_DISTANCE_ALPHA*(*minLCDPDistance));
					}

					bool check1 = (*lastFGMask) | (((*currPxDistance) < UNSTABLE_REG_RATIO_MIN) && (*currFGMask));
					bool check2 = check1 && ((*currUpdateRate) < upLearningRateUpperCap);
					if (check2) {
						/*(*currUpdateRate) = std::min(FEEDBACK_T_UPPER, (*currUpdateRate) +
							(FEEDBACK_T_INCR / ((*currPxDistance)*(*currDistThreshold))));*/
						(*currUpdateRate) = std::min(upUpdateRateHighest, (*currUpdateRate) +
							(upUpdateRateIncrease / ((*currPxDistance)*(*currDynamicRate))));
					}
					check2 = !check1 && ((*currUpdateRate) >= upLearningRateLowerCap);
					if (check2) {
						/*(*currUpdateRate) = std::max(FEEDBACK_T_LOWER, (*currUpdateRate) -
							((FEEDBACK_T_DECR*(*currDistThreshold)) / (*currPxDistance)));*/
						(*currUpdateRate) = std::max(upUpdateRateLowest, (*currUpdateRate) -
							((upUpdateRateDecrease*(*currDynamicRate)) / (*currPxDistance)));
					}

					check1 = (*currDistThreshold) < (std::pow((1 + ((*currPxDistance) * 2)), 2));
					// FG
					if (check1) {
						//(*currDistThreshold) += FEEDBACK_R_VAR;
						(*currDistThreshold) += (*currDynamicRate);
					}
					else {
						//(*currDistThreshold) = std::max(1.0f, (*currDistThreshold) - FEEDBACK_R_VAR);
						(*currDistThreshold) = std::max(1.0f, (*currDistThreshold) - (1.0f / (*currDynamicRate)));
					}
				}
				else if (clsMatchingMethod) {
					(*currPxDistance) = ((1 - MIN_DISTANCE_ALPHA)*(*currPxDistance)) + (MIN_DISTANCE_ALPHA*(*minLCDPDistance));

					bool check1 = (*lastFGMask) | (((*currPxDistance) < UNSTABLE_REG_RATIO_MIN) && (*currFGMask));
					bool check2 = check1 && ((*currUpdateRate) < upLearningRateUpperCap);
					if (check2) {
						(*currUpdateRate) = std::min(upUpdateRateHighest, (*currUpdateRate) +
							(upUpdateRateIncrease / ((*currPxDistance)*(*currDynamicRate))));
					}
					check2 = !check1 && ((*currUpdateRate) >= upLearningRateLowerCap);
					if (check2) {
						(*currUpdateRate) = std::max(upUpdateRateLowest, (*currUpdateRate) -
							((upUpdateRateDecrease*(*currDynamicRate)) / (*currPxDistance)));
					}

					check1 = (*currDistThreshold) < (std::pow((1 + ((*currPxDistance) * 2)), 2));
					// FG
					if (check1) {
						//(*currDistThreshold) += FEEDBACK_R_VAR;
						(*currDistThreshold) += (*currDynamicRate);
					}
					else {
						//(*currDistThreshold) = std::max(1.0f, (*currDistThreshold) - FEEDBACK_R_VAR);
						(*currDistThreshold) = std::max(1.0f, (*currDistThreshold) - (1.0f / (*currDynamicRate)));
					}
				}

			}
		}
	}

	// POST PROCESSING
	if (postSwitch) {
		cv::Mat compensationResult;
		resCurrFGMask.copyTo(resLastRawFGMask);
		cv::Mat element = cv::getStructuringElement(0, cv::Size(5, 5));
		cv::Mat element2 = cv::getStructuringElement(cv::MORPH_RECT,
			cv::Size(2 * 3 + 1, 2 * 3 + 1));
		cv::Mat element3 = cv::getStructuringElement(cv::MORPH_RECT,
			cv::Size(2 * 5 + 1, 2 * 5 + 1));
		cv::Mat element4 = cv::getStructuringElement(cv::MORPH_RECT,
			cv::Size(1, 2 * 4 + 1));
		// 1st round
		cv::Mat tempCurrFGMask;
		resCurrFGMask.copyTo(tempCurrFGMask);
		cv::Mat reconstructLine = BorderLineReconst(resCurrFGMask);
		compensationResult = CompensationMotionHist(resT_1FGMask, resT_2FGMask, resCurrFGMask, postCompensationThreshold);
		cv::morphologyEx(resCurrFGMask, resFGMaskPreFlood, cv::MORPH_CLOSE, element);
		resFGMaskPreFlood.copyTo(resFGMaskFloodedHoles);
		cv::bitwise_or(resFGMaskFloodedHoles, reconstructLine, resFGMaskFloodedHoles);
		resFGMaskFloodedHoles = ContourFill(resFGMaskFloodedHoles);
		cv::erode(resFGMaskPreFlood, resFGMaskPreFlood, cv::Mat(), cv::Point(-1, -1), 3);
		cv::bitwise_or(resCurrFGMask, resFGMaskFloodedHoles, resCurrFGMask);
		cv::bitwise_or(resCurrFGMask, resFGMaskPreFlood, resCurrFGMask);
		cv::bitwise_or(resCurrFGMask, compensationResult, resCurrFGMask);
		// 2nd round
		//cv::morphologyEx(gradientResult, gradientResult, cv::MORPH_GRADIENT, element2);
		//cv::threshold(gradientResult, gradientResult, 80, 255, CV_THRESH_BINARY);
		//resDarkPixel = DarkPixelGenerator(inputImg);
		//cv::morphologyEx(resDarkPixel, resDarkPixel, cv::MORPH_DILATE, element);	
		//cv::bitwise_and(resDarkPixel, gradientResult, gradientResult);
		//cv::morphologyEx(gradientResult, gradientResult, cv::MORPH_DILATE, element3);
		//cv::bitwise_not(gradientResult, gradientResult);
		//cv::bitwise_and(tempCurrFGMask, gradientResult, tempCurrFGMask);
		//cv::morphologyEx(tempCurrFGMask, tempCurrFGMask, cv::MORPH_CLOSE, element2);
		//reconstructLine = BorderLineReconst(tempCurrFGMask);
		////compensationResult = CompensationMotionHist(resT_1FGMask, resT_2FGMask, resCurrFGMask, postCompensationThreshold);
		//cv::morphologyEx(tempCurrFGMask, resFGMaskPreFlood, cv::MORPH_CLOSE, element);
		//resFGMaskPreFlood.copyTo(resFGMaskFloodedHoles);
		//cv::bitwise_or(resFGMaskFloodedHoles, reconstructLine, resFGMaskFloodedHoles);
		//resFGMaskFloodedHoles = ContourFill(resFGMaskFloodedHoles);
		//cv::erode(resFGMaskPreFlood, resFGMaskPreFlood, cv::Mat(), cv::Point(-1, -1), 3);
		//cv::bitwise_or(tempCurrFGMask, resFGMaskFloodedHoles, tempCurrFGMask);
		//cv::bitwise_or(tempCurrFGMask, resFGMaskPreFlood, tempCurrFGMask);
		//cv::morphologyEx(tempCurrFGMask, tempCurrFGMask, cv::MORPH_CLOSE, element2);
		//cv::medianBlur(tempCurrFGMask, tempCurrFGMask, postMedianFilterSize);
		//cv::bitwise_and(resCurrFGMask, tempCurrFGMask, resCurrFGMask);

		cv::medianBlur(resCurrFGMask, resLastFGMask, postMedianFilterSize);
		reconstructLine = BorderLineReconst(resLastFGMask);
		cv::bitwise_or(reconstructLine, resLastFGMask, resLastFGMask);
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
	resDarkPixel = cv::Scalar_<uchar>::all(0);
	resLastImg = (inputImg + (resLastImg*(frameIndex - 1))) / frameIndex;
}

/*=====METHODS=====*/
/*=====DEFAULT methods=====*/
// Refreshes all samples based on the last analyzed frame - checked
void BackgroundSubtractorLCDP::RefreshModel(float refreshFraction, bool forceUpdateSwitch)
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
				DescriptorStruct * bgWord = (bgWordPtr + modelIndex + currModelIndex);
				for (size_t channel = 0; channel < 3; channel++) {
					(*bgWord).rgb[channel] = (*currWord).rgb[channel];
				}
				for (size_t channel = 0; channel < 15; channel++) {
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
// Descriptor Generator-Generate pixels' descriptor (RGB+LCDP) - checked
void BackgroundSubtractorLCDP::DescriptorGenerator(const cv::Mat inputFrame, const PxInfo *pxInfoPtr,
	DescriptorStruct * wordPtr)
{
	(*wordPtr).frameCount = 1;
	for (int channel = 0; channel < 3; channel++) {
		(*wordPtr).rgb[channel] = inputFrame.data[(*pxInfoPtr).bgrDataIndex + channel];
	}
	(*wordPtr).p = frameIndex;
	(*wordPtr).q = frameIndex;
	LCDGenerator(inputFrame, pxInfoPtr, wordPtr);
}
// Generate LCD Descriptor - checked
void BackgroundSubtractorLCDP::LCDGenerator(const cv::Mat inputFrame, const PxInfo *pxInfoPtr, DescriptorStruct *wordPtr)
{
	// Current pixel RGB intensity
	int B_CURR = inputFrame.data[(*pxInfoPtr).bgrDataIndex];
	int G_CURR = inputFrame.data[(*pxInfoPtr).bgrDataIndex + 1];
	int R_CURR = inputFrame.data[(*pxInfoPtr).bgrDataIndex + 2];

	const float * colorDiffRatio = (float*)(descColorDiffRatio.data + (((*pxInfoPtr).dataIndex) * 4));
	const int nbNo = *(descNbNo.data + (*pxInfoPtr).dataIndex);
	// Define Neighbour differences variables
	int R_NB, G_NB, B_NB;
	double ratioBNB_GCURR_MIN, ratioBNB_RCURR_MIN, ratioGNB_BCURR_MIN, ratioGNB_RCURR_MIN, ratioRNB_BCURR_MIN, ratioRNB_GCURR_MIN,
		ratioBNB_GCURR_MAX, ratioBNB_RCURR_MAX, ratioGNB_BCURR_MAX, ratioGNB_RCURR_MAX, ratioRNB_BCURR_MAX, ratioRNB_GCURR_MAX,
		ratioBNB_BCURR_MIN, ratioGNB_GCURR_MIN, ratioRNB_RCURR_MIN, ratioBNB_BCURR_MAX, ratioGNB_GCURR_MAX, ratioRNB_RCURR_MAX;

	if (!descRatioCalculationMethod) {
		// OLD ratio calculation method
		// Calculate thresholding ratio
		double ratioBNB_GCURR = std::max(3.0, double(*(colorDiffRatio)*G_CURR));
		double ratioBNB_RCURR = std::max(3.0, double(*(colorDiffRatio)*R_CURR));
		double ratioGNB_BCURR = std::max(3.0, double(*(colorDiffRatio)*B_CURR));
		double ratioGNB_RCURR = std::max(3.0, double(*(colorDiffRatio)*R_CURR));
		double ratioRNB_BCURR = std::max(3.0, double(*(colorDiffRatio)*B_CURR));
		double ratioRNB_GCURR = std::max(3.0, double(*(colorDiffRatio)*G_CURR));

		ratioBNB_GCURR_MIN = std::max(-255.0, -ratioBNB_GCURR);
		ratioBNB_RCURR_MIN = std::max(-255.0, -ratioBNB_RCURR);
		ratioGNB_BCURR_MIN = std::max(-255.0, -ratioGNB_BCURR);
		ratioGNB_RCURR_MIN = std::max(-255.0, -ratioGNB_RCURR);
		ratioRNB_BCURR_MIN = std::max(-255.0, -ratioRNB_BCURR);
		ratioRNB_GCURR_MIN = std::max(-255.0, -ratioRNB_GCURR);

		ratioBNB_GCURR_MAX = std::min(255.0, ratioBNB_GCURR);
		ratioBNB_RCURR_MAX = std::min(255.0, ratioBNB_RCURR);
		ratioGNB_BCURR_MAX = std::min(255.0, ratioGNB_BCURR);
		ratioGNB_RCURR_MAX = std::min(255.0, ratioGNB_RCURR);
		ratioRNB_BCURR_MAX = std::min(255.0, ratioRNB_BCURR);
		ratioRNB_GCURR_MAX = std::min(255.0, ratioRNB_GCURR);

		double ratioBNB_BCURR = std::max(3.0, double((*(colorDiffRatio) / 10)*B_CURR));
		double ratioGNB_GCURR = std::max(3.0, double((*(colorDiffRatio) / 10)*G_CURR));
		double ratioRNB_RCURR = std::max(3.0, double((*(colorDiffRatio) / 10)*R_CURR));

		ratioBNB_BCURR_MIN = std::max(-255.0, -ratioBNB_BCURR);
		ratioGNB_GCURR_MIN = std::max(-255.0, -ratioGNB_GCURR);
		ratioRNB_RCURR_MIN = std::max(-255.0, -ratioRNB_RCURR);

		ratioBNB_BCURR_MAX = std::min(255.0, ratioBNB_BCURR);
		ratioGNB_GCURR_MAX = std::min(255.0, ratioGNB_GCURR);
		ratioRNB_RCURR_MAX = std::min(255.0, ratioRNB_RCURR);
	}
	else if (descRatioCalculationMethod) {
		// New ratio calculation method
		double ratioBNB_GCURR = std::max(3.0, abs(double(*(colorDiffRatio)*(B_CURR - G_CURR))));
		double ratioBNB_RCURR = std::max(3.0, abs(double(*(colorDiffRatio)*(B_CURR - R_CURR))));
		double ratioGNB_BCURR = std::max(3.0, abs(double(*(colorDiffRatio)*(G_CURR - B_CURR))));
		double ratioGNB_RCURR = std::max(3.0, abs(double(*(colorDiffRatio)*(G_CURR - R_CURR))));
		double ratioRNB_BCURR = std::max(3.0, abs(double(*(colorDiffRatio)*(R_CURR - B_CURR))));
		double ratioRNB_GCURR = std::max(3.0, abs(double(*(colorDiffRatio)*(R_CURR - G_CURR))));

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

		double ratioBNB_BCURR = std::max(3.0, double(*(colorDiffRatio)*B_CURR));
		double ratioGNB_GCURR = std::max(3.0, double(*(colorDiffRatio)*G_CURR));
		double ratioRNB_RCURR = std::max(3.0, double(*(colorDiffRatio)*R_CURR));

		ratioBNB_BCURR_MIN = std::max(-255.0, -ratioBNB_BCURR);
		ratioGNB_GCURR_MIN = std::max(-255.0, -ratioGNB_GCURR);
		ratioRNB_RCURR_MIN = std::max(-255.0, -ratioRNB_RCURR);

		ratioBNB_BCURR_MAX = std::min(255.0, ratioBNB_BCURR);
		ratioGNB_GCURR_MAX = std::min(255.0, ratioGNB_GCURR);
		ratioRNB_RCURR_MAX = std::min(255.0, ratioRNB_RCURR);
	}

	double tempBNB_GCURR, tempBNB_RCURR, tempGNB_BCURR, tempGNB_RCURR, tempRNB_BCURR, tempRNB_GCURR, tempBNB_BCURR,
		tempGNB_GCURR, tempRNB_RCURR;

	for (int nbPixelIndex = 0; nbPixelIndex < nbNo; nbPixelIndex++) {
		// Obtain neighbourhood pixel's value
		B_NB = inputFrame.data[(*pxInfoPtr).nbIndex[nbPixelIndex].bgrDataIndex];
		G_NB = inputFrame.data[(*pxInfoPtr).nbIndex[nbPixelIndex].bgrDataIndex + 1];
		R_NB = inputFrame.data[(*pxInfoPtr).nbIndex[nbPixelIndex].bgrDataIndex + 2];
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

		(*wordPtr).LCDP[nbPixelIndex] = tempResult;
	}

}
// Calculate word persistence value  - checked
float BackgroundSubtractorLCDP::GetLocalWordPersistence(const DescriptorStruct* wordPtr, const size_t currFrameIndex,
	const size_t offsetValue) {
	return (float)((*wordPtr).frameCount) / (((*wordPtr).q - (*wordPtr).p) + ((currFrameIndex - (*wordPtr).q) * 2) + offsetValue);
}

/*=====LUT Methods=====*/
// Generate neighbourhood pixel offset value - checked
void BackgroundSubtractorLCDP::GenerateNbOffset(PxInfo * pxInfoPtr)
{
	const int currX = (*pxInfoPtr).coor_x;
	const int currY = (*pxInfoPtr).coor_y;
	for (int nbIndex = 0; nbIndex < 48; nbIndex++) {
		// Coordinate X value for neighbourhood pixel
		(*pxInfoPtr).nbIndex[nbIndex].coor_x = std::min(frameSizeZero.width, std::max(0, (currX + nbOffset[nbIndex].x)));
		// Coordinate X value for neighbourhood pixel
		(*pxInfoPtr).nbIndex[nbIndex].coor_y = std::min(frameSizeZero.height, std::max(0, (currY + nbOffset[nbIndex].y)));
		// Data index for neighbourhood pixel's pointer
		(*pxInfoPtr).nbIndex[nbIndex].dataIndex = (((*pxInfoPtr).nbIndex[nbIndex].coor_y*(frameSize.width)) + ((*pxInfoPtr).nbIndex[nbIndex].coor_x));
		// Data index for neighbourhood pixel's BGR pointer
		(*pxInfoPtr).nbIndex[nbIndex].bgrDataIndex = 3 * (*pxInfoPtr).nbIndex[nbIndex].dataIndex;
		// Model index for neighbourhood pixel
		(*pxInfoPtr).nbIndex[nbIndex].modelIndex = WORDS_NO * (*pxInfoPtr).nbIndex[nbIndex].dataIndex;
	}
}
// Generate LCD differences Lookup table (0: 100% Same -> 1: 100% Different) - checked
void BackgroundSubtractorLCDP::GenerateLCDDiffLUT() {
	// Total number of differences (Decimal)
	const int totalDiff = std::pow(2, (descDiffNo * 2));
	for (int diffIndex = 0; diffIndex < totalDiff; diffIndex++) {
		float countColour = 0;
		float countTexture = 0;
		int tempDiffIndex = diffIndex;
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
		if (countTexture > 0) {
			LCDDiffLUTPtr[diffIndex] = 1;
		}
		else if (countColour > 0) {
			LCDDiffLUTPtr[diffIndex] = 1;
		}
		else {
			LCDDiffLUTPtr[diffIndex] = 0;
		}
		//LCDDiffLUTPtr[diffIndex] = ((countTexture / 3) + (countColour / 6)) / 2;
	}
}

/*=====MATCHING Methods=====*/
// Descriptor matching (RETURN-True:Not match, False: Match) - checked
bool BackgroundSubtractorLCDP::DescriptorMatching(DescriptorStruct *bgWord, DescriptorStruct *currWord
	, const size_t pxPointer, const double LCDPThreshold, const double RGBThreshold, float &matchLCDPDistance, float &matchRGBDistance,
	bool &rgbMatchPixel, bool &darkPixel)
{
	// Initialize the result be false: match
	bool result = false;
	bool result2 = false;
	int countDistance = 0;
	// Match LCD descriptor
	if (clsLCDPDiffSwitch) {
		result = LCDPMatching(bgWord, currWord, pxPointer, LCDPThreshold, matchLCDPDistance);
		countDistance++;
	}
	// Match RGB descriptor
	if (clsRGBDiffSwitch) {
		result2 = RGBMatching((*bgWord).rgb, (*currWord).rgb, RGBThreshold, matchRGBDistance);
		//result2 = (clsAndOrSwitch) ? (result&tempRGBResult) : (result | tempRGBResult);
		rgbMatchPixel = !result2;
		countDistance++;
	}
	if (clsRGBBrightPxSwitch) {
		bool result3 = BrightRGBMatching((*bgWord).rgb, (*currWord).rgb, 30);
		darkPixel = (!result3);
		result2 = result2&result3;
	}
	result = result2 | result;

	return result;
}
// LCD Matching (RETURN-True:Not match, False: Match) - checked
bool BackgroundSubtractorLCDP::LCDPMatching(DescriptorStruct *bgWord, DescriptorStruct *currWord,
	const size_t pxPointer, const double LCDPThreshold, float &minDistance) {
	float tempDistance = 0.0f;
	for (size_t descIndex = 0; descIndex < (*(descNbNo.data + pxPointer)); descIndex++) {
		// Calculate the total number of bit that are different
		int Diff = std::abs((*bgWord).LCDP[descIndex] - (*currWord).LCDP[descIndex]);
		tempDistance += LCDDiffLUTPtr[Diff];
	}
	minDistance = tempDistance / (*(descNbNo.data + pxPointer));
	return (minDistance > LCDPThreshold) ? true : false;
}
// RGB Matching (RETURN-True:Not match, False: Match) - checked
bool BackgroundSubtractorLCDP::RGBMatching(const int bgRGB[], const int currRGB[], const double RGBThreshold, float &minDistance)
{
	bool result = false;
	int distance = UINT_MAX;
	for (int channel = 0; channel < 3; channel++) {
		const int tempDistance = std::abs(bgRGB[channel] - currRGB[channel]);
		distance = tempDistance < distance ? tempDistance : distance;
		if (tempDistance > RGBThreshold) {
			result = true;
		}
	}
	minDistance = distance / UINT_MAX;
	return result;
}
// Bright Pixel (RETURN-True:Not match, False: Match) match = dark pixel - checked
bool BackgroundSubtractorLCDP::BrightRGBMatching(const int bgRGB[], const int currRGB[], const double BrightThreshold) {
	bool result = false;
	for (int channel = 0; channel < 3; channel++) {
		if ((bgRGB[channel] - currRGB[channel]) > BrightThreshold) {
			return true;
		}
	}
	return result;
}

/*=====POST-PROCESSING Methods=====*/
// Compensation with Motion Hist - checked
cv::Mat BackgroundSubtractorLCDP::CompensationMotionHist(const cv::Mat T_1FGMask, const cv::Mat T_2FGMask, const cv::Mat currFGMask,
	const cv::Mat postCompensationThreshold) {
	cv::Mat compensationResult;
	compensationResult.create(frameSize, CV_8UC1);
	compensationResult = cv::Scalar_<uchar>::all(255);

	for (size_t modelIndex = 0; modelIndex < frameInitTotalPixel; modelIndex++) {
		if (!currFGMask.data[modelIndex]) {
			int totalFGMask = 0;
			for (size_t nbIndex = 0; nbIndex < 9; nbIndex++) {
				totalFGMask += T_1FGMask.data[pxInfoLUTPtr[modelIndex].nbIndex[nbIndex].dataIndex];
				totalFGMask += T_2FGMask.data[pxInfoLUTPtr[modelIndex].nbIndex[nbIndex].dataIndex];
				totalFGMask += currFGMask.data[pxInfoLUTPtr[modelIndex].nbIndex[nbIndex].dataIndex];
			}
			totalFGMask /= 255;
			compensationResult.data[modelIndex] = ((totalFGMask / 26) > postCompensationThreshold.data[modelIndex]) ? 255 : 0;
		}
	}

	return compensationResult;
}
// Contour filling the empty holes - checked
cv::Mat BackgroundSubtractorLCDP::ContourFill(const cv::Mat inputImg) {
	cv::Mat input;
	cv::Mat output;
	cv::copyMakeBorder(inputImg, input, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar());

	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::findContours(input, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);

	// fill external contours
	if (!contours.empty() && !hierarchy.empty())
	{
		int d = contours.size();
		for (int idx = 0; idx < contours.size(); idx++)
		{
			drawContours(input, contours, idx, cv::Scalar::all(255), CV_FILLED, 8);
		}
	}
	//step 3: remove the border
	input = input.rowRange(cv::Range(1, input.rows - 1));
	input = input.colRange(cv::Range(1, input.cols - 1));
	input.copyTo(output);
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
	const size_t maxLineHeight = maxHeight*0.4;
	const size_t maxLineWidth = maxWidth*0.4;
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
				const uchar currFGMask = *(resCurrFGMask.data + pxPointer);

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
// Dark Pixel generator
cv::Mat BackgroundSubtractorLCDP::DarkPixelGenerator(const cv::Mat inputImg) {
	cv::Mat output;
	output.create(frameSize, CV_8UC1);
	output = cv::Scalar_<uchar>::all(0);
	// Store the pixel's RGB values
	int currRgb[3];
	int lastRgb[3];
	size_t pxPointer = 0;
	for (size_t pxPointer = 0; pxPointer < frameInitTotalPixel; ++pxPointer) {
		// Current distance threshold
		float * distThreshold = (float*)(resDistThreshold.data + (pxPointer * 4));
		// RGB descriptor threshold
		const double RGBThreshold = std::max(clsRGBThreshold, floor(clsRGBThreshold*(*distThreshold)));
		bool result = false;
		for (int channel = 0; channel < 3; channel++) {
			currRgb[channel] = inputImg.data[(pxPointer * 3) + channel];
			lastRgb[channel] = resLastImg.data[(pxPointer * 3) + channel];
			if (double(lastRgb[channel] - currRgb[channel]) < (70)) {
				result = true;
			}
			/*if (std::abs(double(lastRgb[channel] - currRgb[channel])) <10) {
				result = true;
			}*/
		}
		float distanceDump;
		bool RGBMatch = !RGBMatching(lastRgb, currRgb, 10, distanceDump);
		bool DarkPixel = !result;
		if (RGBMatch | DarkPixel) {
			output.data[pxPointer] = 255;
		}
	}
	return output;
}
/*=====OTHERS Methods=====*/
// Save parameters
void BackgroundSubtractorLCDP::SaveParameter(std::string filename, std::string folderName) {
	std::ofstream myfile;
	myfile.open(folderName + "/parameter.txt", std::ios::app);
	myfile << "\n----VIDEO PARAMETER----\n";
	myfile << "VIDEO WIDTH:";
	myfile << frameSize.width;
	myfile << "\nVIDEO HEIGHT:";
	myfile << frameSize.height;
	myfile << "\n\n----METHOD THRESHOLD----\n<< << <-DESCRIPTOR DEFAULT PARAMETER-> >> >>";
	myfile << "\nTotal number of LCD differences per pixel:";
	myfile << descDiffNo;
	myfile << "\nTotal number of LCD descriptor's neighbour:";
	myfile << int(*(descNbNo.data));
	myfile << "\nLCD colour differences ratio:";
	float * colorDiffRatio = (float*)(descColorDiffRatio.data);
	myfile << *(colorDiffRatio);
	myfile << "\nPersistence's offset value:";
	myfile << descOffsetValue;
	myfile << "\nRatio calculation method:";
	myfile << descRatioCalculationMethod;
	myfile << "\n\n<<<<<-CLASSIFIER DEFAULT PARAMETER->>>>>";
	myfile << "\nRGB detection switch:";
	myfile << clsRGBDiffSwitch;
	myfile << "\nRGB dark pixel detection switch:";
	myfile << clsRGBBrightPxSwitch;
	myfile << "\nLCD detection switch:";
	myfile << clsLCDPDiffSwitch;
	myfile << "\nLCD detection AND (true) OR (false) switch:";
	myfile << clsAndOrSwitch;
	myfile << "\nDefault LCD differences threshold:";
	myfile << clsLCDPThreshold;
	myfile << "\nMaximum of LCD differences threshold:";
	myfile << clsLCDPMaxThreshold;
	myfile << "\nInitial matched persistence value threshold:";
	float * persistenceThreshold = (float*)(clsPersistenceThreshold.data);
	myfile << *(persistenceThreshold);
	myfile << "\nNeighbourhood matching switch:";
	myfile << clsNbMatchSwitch;
	myfile << "\nTotal number of pixel's neighbour:";
	myfile << int(*(clsNbNo.data));
	myfile << "\nClassify matching method:";
	myfile << clsMatchingMethod;
	myfile << "\nClassify matching threshold:";
	myfile << clsMatchThreshold;

	myfile << "\n\n<<<<<-UPDATE DEFAULT PARAMETER->>>>>";
	myfile << "\nRandom replace model switch:";
	myfile << upRandomReplaceSwitch;
	myfile << "\nInitial random replace model probability:";
	float * updateRate = (float*)(resUpdateRate.data);
	myfile << *(updateRate);
	myfile << "\nRandom update neighbourhood model switch:";
	myfile << upRandomUpdateNbSwitch;
	myfile << "\nInitial update neighbourhood model probability:";
	myfile << *(updateRate);
	myfile << "\nTotal number of neighbour undergo updates:";
	myfile << int(*(upNbNo.data));
	myfile << "\nFeedback loop switch:";
	myfile << upFeedbackSwitch;
	myfile << "\nInitial update rates, T(x):";
	myfile << *(updateRate);
	myfile << "\nInitial commonValue, R:";
	float * distThreshold = (float*)(resDistThreshold.data);
	myfile << *(distThreshold);
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
	myfile << "\nLocal distance threshold change factor:";
	myfile << FEEDBACK_R_VAR;
	myfile << "\nMaximum number of model:";
	myfile << WORDS_NO;

	myfile.close();

	myfile.open(filename + "/parameter.csv", std::ios::app);
	myfile << "," << frameSize.width << "," << frameSize.height << "," << descDiffNo;
	myfile << "," << int(*(descNbNo.data));
	myfile << "," << *(colorDiffRatio) << "," << descOffsetValue << "," << clsRGBDiffSwitch;
	myfile << "," << clsRGBBrightPxSwitch << "," << clsLCDPDiffSwitch << "," << clsAndOrSwitch << "," << clsLCDPThreshold;
	myfile << "," << clsLCDPMaxThreshold;
	myfile << "," << *(persistenceThreshold) << "," << clsMatchThreshold << "," << descRatioCalculationMethod << ",";
	myfile << clsNbMatchSwitch << "," << clsMatchingMethod << "," << upRandomReplaceSwitch;
	myfile << "," << *(updateRate) << "," << upRandomUpdateNbSwitch << "," << int(*(upNbNo.data));
	myfile << "," << upFeedbackSwitch;
	myfile << "," << *(distThreshold) << "," << upDynamicRateIncrease << "," << upDynamicRateDecrease << "," << upUpdateRateIncrease;
	myfile << "," << upUpdateRateDecrease << "," << upLearningRateLowerCap << "," << upLearningRateUpperCap;
	myfile << "," << FEEDBACK_R_VAR << "," << WORDS_NO;
	myfile.close();
}
// Debug pixel location
void BackgroundSubtractorLCDP::DebugPxLocation(int x, int y)
{
	debPxLocation = cv::Point(x, y);
}
