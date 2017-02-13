#include "BackgroundSubtractorLCDP.h"
#include "RandUtils.h"
#include <iostream>
#include <fstream>
#include <vector>

// Local define used to specify the desc dist threshold offset used for unstable regions
#define UNSTAB_DESC_DIST_OFFSET (3.00f)
// Parameters used to define 'unstable' regions, based on segm noise/bg dynamics and local dist threshold values
#define UNSTABLE_REG_RATIO_MIN  (0.10f)
#define UNSTABLE_REG_RDIST_MIN  (3.00f)
// Parameters used to scale dynamic learning rate adjustments  ('T(x)')
#define FEEDBACK_T_DECR  (0.2500f)
#define FEEDBACK_T_INCR  (0.5000f)
#define FEEDBACK_T_LOWER (2.0000f)
#define FEEDBACK_T_UPPER (256.00f)
// Parameters used to adjust the variation step size of 'v(x)'
//#define FEEDBACK_V_INCR  (1.00f)
//#define FEEDBACK_V_DECR  (0.01f)
#define FEEDBACK_V_INCR  (0.50f)
#define FEEDBACK_V_DECR  (0.40f)
// Parameter used to scale dynamic distance threshold adjustments ('R(x)')
#define FEEDBACK_R_VAR (0.01f)

// Local define used to specify the default frame size (320x240 = QVGA)
#define DEFAULT_FRAME_SIZE cv::Size(320,240)

BackgroundSubtractorLCDP::BackgroundSubtractorLCDP(cv::Size inputFrameSize, cv::Mat inputROIFrame, size_t inputWordsNo) :
	/*=====LOOK-UP TABLE=====*/
	// Internal pixel info LUT for all possible pixel indexes
	pxInfoLUTPtr(nullptr),
	// LCD differences LUT
	LCDDiffLUTPtr(nullptr),

	/*=====MODEL Parameters=====*/
	// Store the background's word and it's iterator
	bgWordPtr(nullptr),
	bgWordPtrIter(nullptr),
	// Store the currect frame's word and it's iterator
	currWordPtr(nullptr),
	currWordPtrIter(nullptr),
	// Total number of words per pixel
	WORDS_NO(inputWordsNo),
	// Frame index
	frameIndex(1),

	/*=====DESCRIPTOR Parameters=====*/
	// Total number of LCD differences per pixel
	descDiffNo(18),
	// Persistence's offset value;
	descOffsetValue(1000),

	/*=====CLASSIFIER Parameters=====*/
	// RGB detection switch
	clsRGBDiffSwitch(false),
	// RGB differences threshold
	clsRGBThreshold(10),
	// RGB bright pixel switch
	clsRGBBrightPxSwitch(true),
	// LCDP detection switch
	clsLCDPDiffSwitch(true),
	// LCDP differences threshold
	clsLCDPThreshold(32),
	// Maximum number of LCDP differences threshold
	clsLCDPMaxThreshold(72),
	// LCDP detection AND (true) OR (false) switch
	clsAndOrSwitch(true),
	// Neighbourhood matching switch
	clsNbMatchSwitch(true),

	/*=====FRAME Parameters=====*/
	// ROI frame
	frameRoi(inputROIFrame),
	// Size of input frame
	frameSize(inputFrameSize),
	// Total number of pixel of region of interest
	frameRoiTotalPixel(cv::countNonZero(inputROIFrame)),
	// Total number of pixel of input frame
	frameInitTotalPixel(inputFrameSize.area()),

	/*=====UPDATE Parameters=====*/
	// Random replace model switch
	upRandomReplaceSwitch(true),
	// Random update neighbourhood model switch
	upRandomUpdateNbSwitch(true),
	// Feedback loop switch
	upFeedbackSwitch(true)
{
	CV_Assert(WORDS_NO > 0);
}
BackgroundSubtractorLCDP::~BackgroundSubtractorLCDP() {
	delete[] pxInfoLUTPtr;
	delete[] LCDDiffLUTPtr;
	delete[] bgWordPtr;
	delete[] currWordPtr;
}

void BackgroundSubtractorLCDP::Initialize(const cv::Mat inputFrame, cv::Mat inputROI)
{
	const int totalDescDiffNo = std::pow(2, descDiffNo);
	/*=====LOOK-UP TABLE=====*/
	// Internal pixel info LUT for all possible pixel indexes
	pxInfoLUTPtr = new PxInfoBase[frameInitTotalPixel];
	memset(pxInfoLUTPtr, 0, sizeof(PxInfoBase)*frameInitTotalPixel);
	// LCD differences LUT
	LCDDiffLUTPtr = new unsigned int[totalDescDiffNo];
	memset(LCDDiffLUTPtr, 0, sizeof(unsigned int)*totalDescDiffNo);
	// Generate LCD difference Lookup table
	GenerateLCDDiffLUT();

	/*=====MODEL Parameters=====*/
	// Store the background's word and it's iterator
	bgWordPtr = new Descriptor[frameInitTotalPixel*WORDS_NO];
	memset(bgWordPtr, 0, sizeof(Descriptor)*frameInitTotalPixel*WORDS_NO);
	bgWordPtrIter = bgWordPtr;
	// Store the currect frame's word and it's iterator
	currWordPtr = new Descriptor[frameInitTotalPixel];
	memset(currWordPtr, 0, sizeof(Descriptor)*frameInitTotalPixel);
	currWordPtrIter = currWordPtr;

	// Model initialization check
	modelInitCheck = true;

	/*=====DESCRIPTOR Parameters=====*/
	// Size of neighbourhood 3(3x3)/5(5x5)
	descNbSize.create(frameSize, CV_8UC1);
	descNbSize = cv::Scalar_<uchar>(5);
	// Total number of neighbourhood pixel 8(3x3)/16(5x5)
	descNbNo.create(frameSize, CV_8UC1);
	descNbNo = cv::Scalar_<uchar>(16);
	// LCD colour differences ratio
	descColorDiffRatio.create(frameSize, CV_32FC1);
	descColorDiffRatio = cv::Scalar(0.05f);

	/*=====CLASSIFIER Parameters=====*/
	// Total number of neighbour 8(3x3)/16(5x5)
	clsNbNo.create(frameSize, CV_8UC1);
	clsNbNo = cv::Scalar_<uchar>(0);
	// Matched persistence value threshold
	clsPersistenceThreshold.create(frameSize, CV_32FC1);
	clsPersistenceThreshold = cv::Scalar((1.0f / descOffsetValue) / 4.0f);
	/*=====POST-PROCESS Parameters=====*/
	// Size of median filter
	postMedianFilterSize = 9;
	// The compensation motion history threshold
	compensationThreshold.create(frameSize, CV_32FC1);
	compensationThreshold = cv::Scalar(0.7f);

	if ((frameRoiTotalPixel >= (frameInitTotalPixel / 2)) && (frameInitTotalPixel >= DEFAULT_FRAME_SIZE.area())) {
		/*=====POST-PROCESS Parameters=====*/
		double tempMedianFilterSize = std::min(double(14), floor((frameRoiTotalPixel / DEFAULT_FRAME_SIZE.area()) + 0.5) + postMedianFilterSize);
		if ((int(tempMedianFilterSize) % 2) == 0)
			tempMedianFilterSize = tempMedianFilterSize - 1;
		// Current kernel size for median blur post-proc filtering
		postMedianFilterSize = tempMedianFilterSize;

		/*=====UPDATE Parameters=====*/
		// Specifies whether Tmin / Tmax scaling is enabled or not
		upLearningRateScalingSwitch = true;
		// Specifies the px update spread range
		upUse3x3Spread = !(frameInitTotalPixel > (DEFAULT_FRAME_SIZE.area() * 2));
		// Current learning rate caps
		upLearningRateLowerCap = FEEDBACK_T_LOWER;
		upLearningRateUpperCap = FEEDBACK_T_UPPER;
	}
	else {
		/*=====POST-PROCESS Parameters=====*/
		// Current kernel size for median blur post-proc filtering
		postMedianFilterSize = postMedianFilterSize;

		/*=====UPDATE Parameters=====*/
		// Specifies whether Tmin / Tmax scaling is enabled or not
		upLearningRateScalingSwitch = false;
		// Specifies the px update spread range
		upUse3x3Spread = true;
		// Current learning rate caps
		upLearningRateLowerCap = FEEDBACK_T_LOWER * 2;
		upLearningRateUpperCap = FEEDBACK_T_UPPER * 2;
	}

	/*=====UPDATE Parameters=====*/
	// Initial blinking accumulate level
	upBlinkAccLevel.create(frameSize, CV_32FC1);
	upBlinkAccLevel = cv::Scalar(1.0f);
	// Total number of neighbour undergo neighbourhood updates 8(3X3)/16(5X5)
	upNbNo.create(frameSize, CV_8UC1);
	upNbNo = cv::Scalar_<uchar>(8);

	upSamplesForMovingAvgs = 100;

	/*=====RESULTS=====*/
	// Per-pixel distance thresholds ('R(x)', but used as a relative value to determine both 
	// intensity and descriptor variation thresholds)
	resDistThreshold.create(frameSize, CV_32FC1);
	resDistThreshold = cv::Scalar(1.0f);
	// A lookup map used to keep track of unstable regions (based on segm. noise & local dist. thresholds)
	resUnstableRegionMask.create(frameSize, CV_8UC1);
	resUnstableRegionMask = cv::Scalar_<uchar>::all(0);
	// Per-pixel mean raw segmentation results (used to detect unstable segmentation regions)
	resMeanRawSegmRes_LT.create(frameSize, CV_32FC1);
	resMeanRawSegmRes_LT = cv::Scalar(0.0f);
	resMeanRawSegmRes_ST.create(frameSize, CV_32FC1);
	resMeanRawSegmRes_ST = cv::Scalar(0.0f);
	// Per-pixel mean raw segmentation results (used to detect unstable segmentation regions)
	resMeanFinalSegmRes_LT.create(frameSize, CV_32FC1);
	resMeanFinalSegmRes_LT = cv::Scalar(0.0f);
	resMeanFinalSegmRes_ST.create(frameSize, CV_32FC1);
	resMeanFinalSegmRes_ST = cv::Scalar(0.0f);
	// Per-pixel update rates('T(x)', which contains pixel - level 'sigmas')
	resUpdateRate.create(frameSize, CV_32FC1);
	resUpdateRate = cv::Scalar(upLearningRateLowerCap);
	// Per-pixel mean minimal distances from the model ('D_min(x)', used to control 
	// variation magnitude and direction of 'T(x)' and 'R(x)')
	resMeanMinDist_LT.create(frameSize, CV_32FC1);
	resMeanMinDist_LT = cv::Scalar(0.0f);
	resMeanMinDist_ST.create(frameSize, CV_32FC1);
	resMeanMinDist_ST = cv::Scalar(0.0f);
	// Per-pixel distance variation modulators ('v(x)', relative value used to modulate 'R(x)' 
	// and 'T(x)' variations)
	resVariationModulator.create(frameSize, CV_32FC1);
	resVariationModulator = cv::Scalar(10.0f);
	// Per-pixel blink detection map ('Z(x)')
	resBlinksFrame.create(frameSize, CV_8UC1);
	resBlinksFrame = cv::Scalar_<uchar>::all(0);

	// Minimum RGB distance
	resMinRGBDistance.create(frameSize, CV_32FC1);
	resMinRGBDistance = cv::Scalar(1.0f);
	// Minimum LCD distance
	resMinLCDPDistance.create(frameSize, CV_32FC1);
	resMinLCDPDistance = cv::Scalar(1.0f);
	// Current foreground mask
	resCurrFGMask.create(frameSize, CV_8UC1);
	resCurrFGMask = cv::Scalar_<uchar>::all(0);
	// Previous foreground mask
	resLastFGMask.create(frameSize, CV_8UC1);
	resLastFGMask = cv::Scalar_<uchar>::all(0);
	// The foreground mask generated by the method at [t-1] (without post-proc, used for blinking px detection)
	resLastRawFGMask.create(frameSize, CV_8UC1);
	resLastRawFGMask = cv::Scalar_<uchar>::all(0);
	// The blink mask generated by the method at [t-1] (without post-proc, used for blinking px detection)
	resLastRawFGBlinkMask.create(frameSize, CV_8UC1);
	resLastRawFGBlinkMask = cv::Scalar_<uchar>::all(0);
	// t-1 foreground mask
	resT_1FGMask.create(frameSize, CV_8UC1);
	resT_1FGMask = cv::Scalar_<uchar>::all(255);
	// t-2 foreground mask
	resT_2FGMask.create(frameSize, CV_8UC1);
	resT_2FGMask = cv::Scalar_<uchar>::all(255);
	// Last foreground dilated mask
	resLastFGMaskDilated.create(frameSize, CV_8UC1);
	resLastFGMaskDilated = cv::Scalar_<uchar>::all(0);
	// Last foreground dilated inverted mask
	resLastFGMaskDilatedInverted.create(frameSize, CV_8UC1);
	resLastFGMaskDilatedInverted = cv::Scalar_<uchar>::all(0);
	// Flooded holes foreground mask
	//resFGMaskFloodedHoles.create(frameSize, CV_8UC1);
	resFGMaskFloodedHoles.create(frameSize, CV_8UC1);
	resFGMaskFloodedHoles = cv::Scalar_<uchar>::all(0);
	// Pre flooded holes foreground mask
	resFGMaskPreFlood.create(frameSize, CV_8UC1);
	resFGMaskPreFlood = cv::Scalar_<uchar>::all(0);
	// Current raw foreground blinking mask
	resCurrRawFGBlinkMask.create(frameSize, CV_8UC1);
	resCurrRawFGBlinkMask = cv::Scalar_<uchar>::all(0);

	// Current pixel distance
	resCurrPxDistance.create(frameSize,CV_32FC1);
	resCurrPxDistance = cv::Scalar(0.0f);
	
	// Current pixel average distance
	resCurrAvgDistance.create(frameSize, CV_32FC1);
	resCurrAvgDistance = cv::Scalar(0.0f);

	size_t pxPointer = 0;
	size_t modelPointer = 0;

	for (size_t rowIndex = 0;rowIndex < frameSize.height;rowIndex++) {
		for (size_t colIndex = 0;colIndex < frameSize.width;colIndex++) {
			// Coordinate Y value
			pxInfoLUTPtr[pxPointer].coor_y = (int)rowIndex;
			// Coordinate X value
			pxInfoLUTPtr[pxPointer].coor_x = (int)colIndex;
			// Data index for pointer pixel
			pxInfoLUTPtr[pxPointer].dataIndex = pxPointer;
			// Data index for BGR data
			pxInfoLUTPtr[pxPointer].bgrDataIndex = pxPointer * 3;
			// Start index for pixel's model
			pxInfoLUTPtr[pxPointer].startModelIndex = modelPointer*WORDS_NO;
			++modelPointer;
			/*=====LUT Methods=====*/
			// Generate neighbourhood pixel offset value
			GenerateNbOffset(&pxInfoLUTPtr[pxPointer]);
			// Descriptor Generator-Generate pixels' descriptor (RGB+LCDP)
			DescriptorGenerator(inputFrame, &pxInfoLUTPtr[pxPointer], &currWordPtr[pxPointer]);
			pxPointer++;
		}
	}

	RefreshModel(1, 0);
}

// Program processing
void BackgroundSubtractorLCDP::Process(const cv::Mat inputImg, cv::Mat &outputImg)
{
	// PRE PROCESSING
	cv::GaussianBlur(inputImg, inputImg, preGaussianSize, 0, 0);

	// FG DETECTION PROCESS
	size_t modelPointer = 0;
	for (size_t pxPointer = 0; pxPointer < frameInitTotalPixel; ++pxPointer) {
		if (frameRoi.data[pxPointer]) {
			// Descriptor Generator-Generate pixels' descriptor (RGB+LCDP)
			DescriptorGenerator(inputImg, &pxInfoLUTPtr[pxPointer], &currWordPtr[pxPointer]);
			float * blinkAccLevel = (float*)(upBlinkAccLevel.data + (pxPointer * 4));
			float * distThreshold = (float*)(resDistThreshold.data + (pxPointer * 4));
			// Start index of the model of the current pixel
			const size_t startModelIndex = pxInfoLUTPtr[pxPointer].startModelIndex;
			float * currAvgDistance = (float*)(resCurrAvgDistance.data + (pxPointer * 4));
			// LCD descriptor threshold
			//double LCDPThreshold = clsLCDPThreshold<(*currAvgDistance)?((*currAvgDistance)>clsLCDPMaxThreshold? clsLCDPMaxThreshold :float((*currAvgDistance))):clsLCDPThreshold;
			const double LCDPThreshold = clsLCDPThreshold + (std::pow(1.5, floor((*distThreshold) + 0.5)));
			// RGB descriptor threshold
			const double RGBThreshold = floor(clsRGBThreshold*(*distThreshold));
			// Current pixel's foreground mask
			uchar * currFGMask = (resCurrFGMask.data + pxPointer);
			// Current pixel's descriptor
			Descriptor* currWord = &currWordPtr[pxPointer];
			// Current pixel's persistence threshold
			float * persistenceThreshold = (float*)(clsPersistenceThreshold.data + (pxPointer * 4));
			// Current pixel's min distance
			float * minLCDPDistance = (float*)(resMinLCDPDistance.data + (pxPointer * 4));
			float * minRGBDistance = (float*)(resMinRGBDistance.data + (pxPointer * 4));
			// Current pixel's update rate
			float * updateRate = (float*)(resUpdateRate.data + (pxPointer * 4));
			// Potential persistence value accumulator
			float potentialPersistenceSum = 0.0f;
			// Last word's persistence weight
			float lastWordWeight = FLT_MAX;
			// Starting word index
			size_t localWordIdx = 0;

			while (localWordIdx < WORDS_NO && potentialPersistenceSum < (*persistenceThreshold)) {
				Descriptor* bgWord = (bgWordPtr + startModelIndex + localWordIdx);
				const float currWordWeight = GetLocalWordWeight(bgWord, frameIndex, descOffsetValue);
				float tempRGBDistance;
				float tempLCDPDistance;
				if (((*bgWord).frameCount > 0)
					&& (!DescriptorMatching(bgWord, currWord, pxPointer, LCDPThreshold, RGBThreshold, tempRGBDistance, tempLCDPDistance))) {
					(*bgWord).frameCount += 1;
					(*bgWord).q = frameIndex;
					potentialPersistenceSum += currWordWeight;
				}
				// Update min distance
				(*minLCDPDistance) = tempLCDPDistance < (*minLCDPDistance) ? tempLCDPDistance : (*minLCDPDistance);
				(*minRGBDistance) = tempRGBDistance < (*minRGBDistance) ? tempRGBDistance : (*minRGBDistance);

				if (currWordWeight > lastWordWeight) {
					std::swap(bgWordPtr[startModelIndex + localWordIdx], bgWordPtr[startModelIndex + localWordIdx - 1]);
				}
				else
					lastWordWeight = currWordWeight;
				++localWordIdx;
			}

			// BG Pixels
			if (potentialPersistenceSum >= (*persistenceThreshold)) {
				(*currFGMask) = 0;

			}
			// FG Pixels
			else {
				(*currFGMask) = 255;
				if (clsNbMatchSwitch) {
					// Compare with neighbour's model
					//const uchar tempNbNo = *(clsNbNo.data + pxPointer);
					// ATTENTION IMPLEMENT AUTO NB NO
					size_t nbMatchNo = 0;
					if ((*blinkAccLevel) > upSamplesForMovingAvgs) {
						nbMatchNo = 16;
					}
					else if ((*blinkAccLevel) > 0) {
						nbMatchNo = 8;
					}


					for (size_t nbIndex = 0;nbIndex < nbMatchNo;nbIndex++) {
						const size_t startModelIndex = pxInfoLUTPtr[pxPointer].nbIndex[nbIndex].dataIndex;
						potentialPersistenceSum = 0.0f;
						lastWordWeight = FLT_MAX;
						localWordIdx = 0;

						while (localWordIdx < WORDS_NO && potentialPersistenceSum < (*persistenceThreshold)) {
							Descriptor* bgWord = (bgWordPtr + startModelIndex + localWordIdx);
							const float currWordWeight = GetLocalWordWeight(bgWord, frameIndex, descOffsetValue);
							float tempRGBDistance;
							float tempLCDPDistance;
							if (((*bgWord).frameCount > 0)
								&& (!DescriptorMatching(bgWord, currWord, pxPointer, LCDPThreshold, RGBThreshold, tempRGBDistance, tempLCDPDistance))) {
								(*bgWord).frameCount += 1;
								(*bgWord).q = frameIndex;
								potentialPersistenceSum += currWordWeight;
							}
							// Update min distance
							(*minLCDPDistance) = tempLCDPDistance < (*minLCDPDistance) ? tempLCDPDistance : (*minLCDPDistance);
							(*minRGBDistance) = tempRGBDistance < (*minRGBDistance) ? tempRGBDistance : (*minRGBDistance);
							if (currWordWeight > lastWordWeight) {
								std::swap(bgWordPtr[startModelIndex + localWordIdx], bgWordPtr[startModelIndex + localWordIdx - 1]);
							}
							else
								lastWordWeight = currWordWeight;
							++localWordIdx;
						}
						// BG Pixels
						if (potentialPersistenceSum >= (*persistenceThreshold)) {
							(*currFGMask) = 0;
							break;
						}
					}
				}
			}

			// UPDATE PROCESS
			// Replace current frame's descriptor with the modelres that having lowest persistence value among others
			Descriptor* bgWord = (bgWordPtr + startModelIndex + WORDS_NO - 1);
			(*bgWord).frameCount = (*currWord).frameCount;
			(*bgWord).p = (*currWord).p;
			(*bgWord).q = (*currWord).q;
			for (size_t channel = 0;channel < 3;channel++) {
				(*bgWord).rgb[channel] = (*currWord).rgb[channel];
			}
			for (size_t channel = 0;channel < 16;channel++) {
				(*bgWord).LCDP[channel] = (*currWord).LCDP[channel];
			}
			if ((*currFGMask)) {
				(*blinkAccLevel) += 1;
			}
			else {
				(*blinkAccLevel) -= 1;
			}
			// Random replace current frame's descriptor with the matched model
			if (upRandomReplaceSwitch) {
				const float randNumber = ((double)std::rand() / (RAND_MAX));
				const float randNumber2 = ((double)std::rand() / (RAND_MAX));
				bool checkTemp = ((1 / *(updateRate)) >= randNumber);
				bool checkTemp2 = ((1 / FEEDBACK_T_LOWER) >= randNumber2);
				const bool check1 = checkTemp && (!(*currFGMask)) && checkTemp2;
				if (check1) {
					int randNum = rand() % ((WORDS_NO - 1) + 1);
					Descriptor* bgWord = (bgWordPtr + startModelIndex + randNum);
					for (size_t channel = 0;channel < 3;channel++) {
						(*bgWord).rgb[channel] = (*currWord).rgb[channel];
					}
					for (size_t channel = 0;channel < 16;channel++) {
						(*bgWord).LCDP[channel] = (*currWord).LCDP[channel];
					}
				}
			}

			// Randomly update a selected neighbour descriptor with current descriptor
			if (upRandomUpdateNbSwitch) {
				const float randNumber = ((double)std::rand() / (RAND_MAX));
				const bool check1 = ((1 / *(updateRate)) >= randNumber) && (*currFGMask);
				if (check1) {
					int randNum = rand() % ((WORDS_NO - 1) + 1);
					int randNbNo = std::floor(rand() % (*(upNbNo.data + pxPointer)));
					cv::Point sampleCoor;

					getRandSamplePosition_3x3(sampleCoor, cv::Point(pxInfoLUTPtr[pxPointer].coor_x, pxInfoLUTPtr[pxPointer].coor_y), 0, frameSize);
					const size_t samplePxIndex = frameSize.width*sampleCoor.y + sampleCoor.x;
					// Start index of the model of the current pixel
					const size_t startModelIndex = pxInfoLUTPtr[samplePxIndex].startModelIndex;

					potentialPersistenceSum = 0.0f;
					lastWordWeight = FLT_MAX;
					localWordIdx = 0;
					bool * matchIndex = new bool[WORDS_NO];
					while (localWordIdx < WORDS_NO && potentialPersistenceSum < (*persistenceThreshold)) {
						Descriptor* bgWord = (bgWordPtr + startModelIndex + localWordIdx);
						const float currWordWeight = GetLocalWordWeight(bgWord, frameIndex, descOffsetValue);
						float tempRGBDistance;
						float tempLCDPDistance;
						if (((*bgWord).frameCount > 0)
							&& (!DescriptorMatching(bgWord, currWord, pxPointer, LCDPThreshold, RGBThreshold, tempRGBDistance, tempLCDPDistance))) {
							*(matchIndex + localWordIdx) = true;
							potentialPersistenceSum += currWordWeight;
						}
						++localWordIdx;
					}
					localWordIdx = 0;
					lastWordWeight = FLT_MAX;

					if (potentialPersistenceSum >= (*persistenceThreshold)) {
						// Start index of the model of the current pixel
						const size_t startModelIndex = pxInfoLUTPtr[pxPointer].startModelIndex;
						while (localWordIdx < WORDS_NO) {
							if (*(matchIndex + localWordIdx)) {
								Descriptor* bgWord = (bgWordPtr + startModelIndex + localWordIdx);
								/*const float currWordWeight = GetLocalWordWeight(bgWord, frameIndex, descOffsetValue);
								bgWord->frameCount = currWord->frameCount;
								bgWord->p = currWord->p;
								bgWord->q = currWord->q;*/
								for (size_t channel = 0;channel < 3;channel++) {
									(*bgWord).rgb[channel] = (*currWord).rgb[channel];
								}
								for (size_t channel = 0;channel < 16;channel++) {
									(*bgWord).LCDP[channel] = (*currWord).LCDP[channel];
								}

								/*if (currWordWeight>lastWordWeight) {
									std::swap(bgWordPtr[startModelIndex + localWordIdx], bgWordPtr[startModelIndex + localWordIdx - 1]);
								}
								else
									lastWordWeight = currWordWeight;*/
							}
							++localWordIdx;
						}
					}
				}
			}
			// Update persistence threshold
			const float highestPersistence = GetLocalWordWeight((bgWordPtr + startModelIndex), frameIndex, descOffsetValue);
			*(persistenceThreshold) = std::min(1.0f, highestPersistence / (*(distThreshold)));
			if (upFeedbackSwitch) {
				// Unstable region mask
				uchar * unstableRegionMask = (uchar*)(resUnstableRegionMask.data + pxPointer);
				// Mean raw segmentation result LT
				float * meanRawSegmRes_LT = (float*)(resMeanFinalSegmRes_LT.data + (pxPointer * 4));
				// Mean raw segmentation result ST
				float * meanRawSegmRes_ST = (float*)(resMeanFinalSegmRes_ST.data + (pxPointer * 4));
				// Mean min distance LT
				float * meanMinDist_LT = (float*)(resMeanMinDist_LT.data + (pxPointer * 4));
				// Mean min distance ST
				float * meanMinDist_ST = (float*)(resMeanMinDist_ST.data + (pxPointer * 4));
				// V(x)
				float * variationModulator = (float*)(resVariationModulator.data + (pxPointer * 4));
				// Update unstable region mask
				(*unstableRegionMask) = (((*distThreshold) > UNSTABLE_REG_RDIST_MIN) | (std::abs((*meanRawSegmRes_LT) - (*meanRawSegmRes_ST)) > UNSTABLE_REG_RATIO_MIN)) ? 1 : 0;
				// Last foreground mask
				uchar * lastFGMask = (uchar*)(resLastFGMask.data + pxPointer);
				// Blinking frame
				uchar * blinkFrame = (uchar*)(resBlinksFrame.data + pxPointer);

				// Current pixel distance
				float * currPxDistance = (float*)(resCurrPxDistance.data + (pxPointer * 4));

				// Update roll average factor
				const float rollAvgFactor_LT = 1.0f / std::min((float)frameIndex + 1.0f, (float)upSamplesForMovingAvgs);
				const float rollAvgFactor_ST = 1.0f / std::min((float)frameIndex, (float)upSamplesForMovingAvgs / 4.0f);


				// FG
				if (*(currFGMask)) {
					float minDistance = (*(persistenceThreshold)-potentialPersistenceSum) / *(persistenceThreshold);
					if (clsRGBDiffSwitch) {
						minDistance += ((*minLCDPDistance) + (*minRGBDistance)) / 2;
					}
					else {
						minDistance += (*minLCDPDistance);
					}
					const float percentDiff = ((*minLCDPDistance) - (*currAvgDistance)) / (*currAvgDistance);
					(*currAvgDistance) = percentDiff < 0.6 ? ((((*currAvgDistance)*(frameIndex - 1)) + (*minLCDPDistance)) / frameIndex) : (*currAvgDistance);
					(*currPxDistance) = (clsLCDPThreshold-minDistance)<0?(abs(minDistance-clsLCDPThreshold)/clsLCDPThreshold):(*currPxDistance);

					(*meanMinDist_LT) = ((*meanMinDist_LT) * (1 - rollAvgFactor_LT)) + (minDistance*rollAvgFactor_LT);
					(*meanMinDist_ST) = ((*meanMinDist_ST) * (1 - rollAvgFactor_ST)) + (minDistance*rollAvgFactor_ST);
					(*meanRawSegmRes_LT) = ((*meanRawSegmRes_LT) * (1 - rollAvgFactor_LT)) + rollAvgFactor_LT;
					(*meanRawSegmRes_ST) = ((*meanRawSegmRes_ST) * (1 - rollAvgFactor_ST)) + rollAvgFactor_ST;
				}
				// BG
				else {
					float minDistance;
					if (clsRGBDiffSwitch) {
						minDistance = ((*minLCDPDistance) + (*minRGBDistance)) / 2;
					}
					else {
						minDistance = (*minLCDPDistance);
					}
					//(*currPxDistance) = (clsLCDPThreshold - minDistance)>0 ? (abs(minDistance - clsLCDPThreshold) / clsLCDPThreshold) : (*currPxDistance);
					(*currAvgDistance) = ((((*currAvgDistance)*(frameIndex - 1)) + (*minLCDPDistance)) / frameIndex);
					(*meanMinDist_LT) = ((*meanMinDist_LT) * (1 - rollAvgFactor_LT)) + (minDistance*rollAvgFactor_LT);
					(*meanMinDist_ST) = ((*meanMinDist_ST) * (1 - rollAvgFactor_ST)) + (minDistance*rollAvgFactor_ST);
					(*meanRawSegmRes_LT) = ((*meanRawSegmRes_LT) * (1 - rollAvgFactor_LT));
					(*meanRawSegmRes_ST) = ((*meanRawSegmRes_ST) * (1 - rollAvgFactor_ST));
				}

				bool check1 = (*lastFGMask) | ((std::min((*meanMinDist_LT), (*meanMinDist_ST)) < UNSTABLE_REG_RATIO_MIN) && (*currFGMask));
				bool check2 = check1 && ((*updateRate) < upLearningRateUpperCap);
				if (check2) {
					(*updateRate) = std::min(FEEDBACK_T_UPPER, (*updateRate) +
						(FEEDBACK_T_INCR / ((std::max((*meanMinDist_LT), (*meanMinDist_ST))*(*variationModulator)))));
				}
				check2 = !check1 && ((*updateRate) >= upLearningRateLowerCap);
				if (check2) {
					(*updateRate) = std::max(FEEDBACK_T_LOWER, (*updateRate) -
						((FEEDBACK_T_DECR*(*variationModulator)) /
						(std::max((*meanMinDist_LT), (*meanMinDist_ST)))));
				}

				check1 = (std::max((*meanMinDist_LT), (*meanMinDist_ST)) > UNSTABLE_REG_RATIO_MIN) && (*blinkFrame);
				if (check1) {
					(*variationModulator) += FEEDBACK_V_INCR;
				}
				else
				{
					if ((*variationModulator) > FEEDBACK_V_DECR)
						if ((*lastFGMask)) {
							(*variationModulator) -= (FEEDBACK_V_DECR / 4);
						}
						else if ((*unstableRegionMask)) {
							(*variationModulator) -= (FEEDBACK_V_DECR / 2);
						}
						else {
							(*variationModulator) -= (FEEDBACK_V_DECR);
						}
				}
				(*variationModulator) = std::max(FEEDBACK_V_DECR, float((*variationModulator)));

				/*check1 = (*distThreshold) < (std::pow((1 + (std::min((*meanMinDist_LT), (*meanMinDist_ST)) * 2)), 2));
				if (check1) {
					(*distThreshold) = std::max(1.0f, (*distThreshold) - (FEEDBACK_R_VAR / (*variationModulator)));
				}
				else {
					(*distThreshold) = std::max(1.0f, (*distThreshold) - (FEEDBACK_R_VAR / (*variationModulator)));
				}*/
				// FG
				if (*(currFGMask)) {
					(*distThreshold) = std::max(1.0f, (*distThreshold) + (*currPxDistance));
				}
				else {
					(*distThreshold) = std::max(1.0f, (*distThreshold) - (*currPxDistance));
				}
			}
			modelPointer++;
		}


	}

	cv::Mat compensationResult;

	// POST PROCESSING
	//cv::Mat image_thresh;

	//cv::Mat tempCurrMask = resCurrFGMask;
	//resCurrFGMask.copyTo(outputImg);
	cv::bitwise_xor(resCurrFGMask, resLastRawFGMask, resCurrRawFGBlinkMask);
	cv::bitwise_or(resCurrRawFGBlinkMask, resLastRawFGBlinkMask, resBlinksFrame);
	resCurrRawFGBlinkMask.copyTo(resLastRawFGBlinkMask);
	resCurrFGMask.copyTo(resLastRawFGMask);
	cv::Mat element = cv::getStructuringElement(0, cv::Size(5, 5));
	//cv::morphologyEx(resCurrFGMask, resCurrFGMask, cv::MORPH_OPEN, cv::Mat());
	cv::morphologyEx(resCurrFGMask, resCurrFGMask, cv::MORPH_OPEN, element);
	cv::Mat borderLineReconstructResult = BorderLineReconst(resCurrFGMask);
	cv::bitwise_or(resCurrFGMask, borderLineReconstructResult, resCurrFGMask);
	compensationResult = CompensationMotionHist(resT_1FGMask, resT_2FGMask, resCurrFGMask, compensationThreshold);
	//cv::morphologyEx(resCurrFGMask, resFGMaskPreFlood, cv::MORPH_CLOSE, cv::Mat());
	cv::morphologyEx(resCurrFGMask, resFGMaskPreFlood, cv::MORPH_CLOSE, element);
	resFGMaskPreFlood.copyTo(resFGMaskFloodedHoles);
	resFGMaskFloodedHoles = ContourFill(resFGMaskFloodedHoles);
	//cv::floodFill(resFGMaskFloodedHoles, cv::Point(0, 0), UCHAR_MAX);
	//cv::bitwise_not(resFGMaskFloodedHoles, resFGMaskFloodedHoles);
	cv::erode(resFGMaskPreFlood, resFGMaskPreFlood, cv::Mat(), cv::Point(-1, -1), 3);
	cv::bitwise_or(resCurrFGMask, resFGMaskFloodedHoles, resCurrFGMask);
	//cv::bitwise_or(resCurrFGMask, mask, resCurrFGMask);
	cv::bitwise_or(resCurrFGMask, resFGMaskPreFlood, resCurrFGMask);
	cv::bitwise_or(resCurrFGMask, compensationResult, resCurrFGMask);

	//borderLineReconstructResult = BorderLineReconst(resCurrFGMask);
	//cv::bitwise_or(resCurrFGMask, borderLineReconstructResult, resCurrFGMask);

	//cv::floodFill(resCurrFGMask, cv::Point(0, 0), UCHAR_MAX);
	//cv::bitwise_not(resCurrFGMask, resCurrFGMask);

	cv::medianBlur(resCurrFGMask, resLastFGMask, postMedianFilterSize);
	cv::dilate(resLastFGMask, resLastFGMaskDilated, cv::Mat(), cv::Point(-1, -1), 3);
	cv::bitwise_and(resBlinksFrame, resLastFGMaskDilatedInverted, resBlinksFrame);
	cv::bitwise_not(resLastFGMaskDilated, resLastFGMaskDilatedInverted);
	cv::bitwise_and(resBlinksFrame, resLastFGMaskDilatedInverted, resBlinksFrame);

	resLastFGMask.copyTo(resCurrFGMask);

	resT_1FGMask.copyTo(resT_2FGMask);
	resLastFGMask.copyTo(resT_1FGMask);

	//tempCurrMask.copyTo(outputImg);
	resCurrFGMask.copyTo(outputImg);

	// Frame Index
	frameIndex++;
	// Minimum RGB distance
	resMinRGBDistance = cv::Scalar(1.0f);
	// Minimum LCD distance
	resMinLCDPDistance = cv::Scalar(1.0f);
}

/*=====METHODS=====*/
/*=====DEFAULT methods=====*/
// Refreshes all samples based on the last analyzed frame
void BackgroundSubtractorLCDP::RefreshModel(float refreshFraction, bool forceUpdateSwitch)
{
	const size_t noSampleBeRefresh = refreshFraction < 1.0f ? (size_t)(refreshFraction*WORDS_NO) : WORDS_NO;
	const size_t refreshStartPos = refreshFraction < 1.0f ? rand() % WORDS_NO : 0;
	size_t modelPointer = 0;
	for (size_t pxPointer = 0;pxPointer < frameInitTotalPixel;pxPointer++) {
		if (frameRoi.data[pxPointer]) {
			if (!forceUpdateSwitch || !resLastFGMask.data[pxPointer]) {
				// Start index of the model of the current pixel
				const size_t startModelIndex = pxInfoLUTPtr[pxPointer].startModelIndex;
				//// LCD descriptor threshold
				//const double LCDPThreshold = clsLCDPThreshold + (std::pow(1.5, floor(resDistThreshold.data[pxPointer] + 0.5)));
				//// RGB descriptor threshold
				//const double RGBThreshold = floor(clsRGBThreshold*resDistThreshold.data[pxPointer]);
				for (size_t currModelIndex = refreshStartPos; currModelIndex < refreshStartPos + noSampleBeRefresh; ++currModelIndex) {
					//Descriptor currPixelDesc = *(bgWordPtr + startModelIndex + currModelIndex);
					cv::Point sampleCoor;
					getRandSamplePosition(sampleCoor, cv::Point(pxInfoLUTPtr[pxPointer].coor_x, pxInfoLUTPtr[pxPointer].coor_y), 0, frameSize);
					const size_t samplePxIndex = frameSize.width*sampleCoor.y + sampleCoor.x;
					if (forceUpdateSwitch || !resLastFGMaskDilated.data[pxPointer]) {
						Descriptor * currWord = (currWordPtr + samplePxIndex);
						//bool foundUnited = false;
						//size_t localWordIdx;
						//for (localWordIdx = 0; localWordIdx < WORDS_NO; ++localWordIdx) {
						//	Descriptor* bgWord = (bgWordPtr + startModelIndex + localWordIdx);
						//	float tempRGBDistance;
						//	float tempLCDPDistance;
						//	if (((*bgWord).frameCount>0)
						//		&& (!DescriptorMatching(bgWord, currWord, LCDPThreshold, RGBThreshold, tempRGBDistance, tempLCDPDistance))) {
						//		(*bgWord).frameCount += 1;
						//		(*bgWord).q = frameIndex;
						//		break;
						//	}
						//	else if (!bgWord)
						//		foundUnited = true;
						//}
						//if (localWordIdx == WORDS_NO) {
						//	localWordIdx = WORDS_NO - 1;
						//	Descriptor * bgWord = foundUnited ? bgWordPtrIter++ : (bgWordPtr + startModelIndex + localWordIdx);
						Descriptor * bgWord = bgWordPtrIter++;
						for (size_t channel = 0;channel < 3;channel++) {
							(*bgWord).rgb[channel] = (*currWord).rgb[channel];
						}
						for (size_t channel = 0;channel < 15;channel++) {
							(*bgWord).LCDP[channel] = (*currWord).LCDP[channel];
						}
						(*bgWord).frameCount = 1;
						(*bgWord).p = frameIndex;
						(*bgWord).q = frameIndex;
						//}
						//while (localWordIdx > 0 && (!(bgWordPtr + startModelIndex + localWordIdx-1) ||
						//	GetLocalWordWeight((bgWordPtr + startModelIndex + localWordIdx), frameIndex, descOffsetValue) > GetLocalWordWeight((bgWordPtr + startModelIndex + localWordIdx-1), frameIndex, descOffsetValue))) {
						//	std::swap(bgWordPtr[startModelIndex + localWordIdx], bgWordPtr[startModelIndex + localWordIdx - 1]);
						//	--localWordIdx;
						//}
					}
				}
			}
			modelPointer++;
		}
	}
}

/*=====DESCRIPTOR Methods=====*/
// Descriptor Generator-Generate pixels' descriptor (RGB+LCDP)
void BackgroundSubtractorLCDP::DescriptorGenerator(const cv::Mat inputFrame, const PxInfoBase *pxInfoPtr,
	Descriptor * wordPtr)
{
	(*wordPtr).frameCount = 1;
	for (int channel = 0;channel < 3;channel++) {
		(*wordPtr).rgb[channel] = inputFrame.data[(*pxInfoPtr).bgrDataIndex + channel];
	}
	(*wordPtr).p = frameIndex;
	(*wordPtr).q = frameIndex;
	LCDGenerator(inputFrame, pxInfoPtr, wordPtr);
}
// Generate LCD Descriptor
void BackgroundSubtractorLCDP::LCDGenerator(const cv::Mat inputFrame, const PxInfoBase *pxInfoPtr, Descriptor *wordPtr)
{
	int R_CURR, G_CURR, B_CURR;
	int R_NB, G_NB, B_NB;
	int currPixelTotalRGB = 0;
	B_CURR = inputFrame.data[(*pxInfoPtr).bgrDataIndex];
	G_CURR = inputFrame.data[(*pxInfoPtr).bgrDataIndex + 1];
	R_CURR = inputFrame.data[(*pxInfoPtr).bgrDataIndex + 2];
	for (int channel = 0;channel < 3;channel++) {
		currPixelTotalRGB += inputFrame.data[(*pxInfoPtr).bgrDataIndex + channel];
	}
	const float * colorDiffRatio = (float*)(descColorDiffRatio.data + (((*pxInfoPtr).dataIndex) * 4));
	const int ratio = std::max(3, (int)(*(colorDiffRatio)*(currPixelTotalRGB / 3)));
	const int nRatio = -ratio;
	const int nbNo = descNbNo.data[(*pxInfoPtr).dataIndex];
	for (int nbPixelIndex = 0;nbPixelIndex < nbNo;nbPixelIndex++) {
		// Obtain neighbourhood pixel's value
		B_NB = inputFrame.data[(*pxInfoPtr).nbIndex[nbPixelIndex].bDataIndex];
		G_NB = inputFrame.data[(*pxInfoPtr).nbIndex[nbPixelIndex].gDataIndex];
		R_NB = inputFrame.data[(*pxInfoPtr).nbIndex[nbPixelIndex].rDataIndex];

		uint tempResult = 0;
		// B_NB - G_CURR
		const int tempBNB_GCURR = B_NB - G_CURR;
		tempResult += ((tempBNB_GCURR > ratio) ? 1 : ((tempBNB_GCURR < nRatio) ? 3 : 0));
		// B_NB - R_CURR
		const int tempBNB_RCURR = B_NB - R_CURR;
		tempResult += ((tempBNB_RCURR > ratio) ? 1 : ((tempBNB_RCURR < nRatio) ? 3 : 0)) << 2;
		// G_NB - B_CURR
		const int tempGNB_BCURR = G_NB - B_CURR;
		tempResult += ((tempGNB_BCURR > ratio) ? 1 : ((tempGNB_BCURR < nRatio) ? 3 : 0)) << 4;
		// G_NB - R_CURR
		const int tempGNB_RCURR = G_NB - R_CURR;
		tempResult += ((tempGNB_RCURR > ratio) ? 1 : ((tempGNB_RCURR < nRatio) ? 3 : 0)) << 6;
		// R_NB - B_CURR
		const int tempRNB_BCURR = R_NB - B_CURR;
		tempResult += ((tempRNB_BCURR > ratio) ? 1 : ((tempRNB_BCURR < nRatio) ? 3 : 0)) << 8;
		// R_NB - G_CURR
		const int tempRNB_GCURR = R_NB - G_CURR;
		tempResult += ((tempRNB_GCURR > ratio) ? 1 : ((tempRNB_GCURR < nRatio) ? 3 : 0)) << 10;
		// B_NB - B_CURR
		const int tempBNB_BCURR = B_NB - B_CURR;
		tempResult += ((tempBNB_BCURR > ratio) ? 1 : ((tempBNB_BCURR < nRatio) ? 3 : 0)) << 12;
		// G_NB - G_CURR
		const int tempGNB_GCURR = G_NB - G_CURR;
		tempResult += ((tempGNB_GCURR > ratio) ? 1 : ((tempGNB_GCURR < nRatio) ? 3 : 0)) << 14;
		// R_NB - R_CURR
		const int tempRNB_RCURR = R_NB - R_CURR;
		tempResult += ((tempRNB_RCURR > ratio) ? 1 : ((tempRNB_RCURR < nRatio) ? 3 : 0)) << 16;
		(*wordPtr).LCDP[nbPixelIndex] = tempResult;
	}

}
// Calculate word persistence value
float BackgroundSubtractorLCDP::GetLocalWordWeight(const Descriptor* wordPtr, const size_t currFrameIndex, const size_t offsetValue) {
	return (float)((*wordPtr).frameCount) / (((*wordPtr).q - (*wordPtr).p) + ((currFrameIndex - (*wordPtr).q) * 2) + offsetValue);
}

/*=====LUT Methods=====*/
// Generate neighbourhood pixel offset value
void BackgroundSubtractorLCDP::GenerateNbOffset(PxInfoBase * pxInfoPtr)
{
	const int maxWidth = frameSize.width - 1;
	const int maxHeight = frameSize.height - 1;
	const int x = (*pxInfoPtr).coor_x;
	const int y = (*pxInfoPtr).coor_y;
	for (int nbIndex = 0;nbIndex < 16;nbIndex++) {
		const int nbPixel_X = std::min(maxWidth, std::max(0, (x + nbOffset[nbIndex].x)));
		const int nbPixel_Y = std::min(maxHeight, std::max(0, (y + nbOffset[nbIndex].y)));
		(*pxInfoPtr).nbIndex[nbIndex].bDataIndex = (3 * ((nbPixel_Y*(maxWidth + 1)) + (nbPixel_X + 1))) - 3;
		(*pxInfoPtr).nbIndex[nbIndex].gDataIndex = (3 * ((nbPixel_Y*(maxWidth + 1)) + (nbPixel_X + 1))) - 2;
		(*pxInfoPtr).nbIndex[nbIndex].rDataIndex = (3 * ((nbPixel_Y*(maxWidth + 1)) + (nbPixel_X + 1))) - 1;
		(*pxInfoPtr).nbIndex[nbIndex].dataIndex = ((nbPixel_Y*(maxWidth + 1)) + (nbPixel_X + 1)) - 1;
	}
}
// Generate LCD difference Lookup table
void BackgroundSubtractorLCDP::GenerateLCDDiffLUT() {
	// Total number of bits (Decimal)
	const int totalDiff = std::pow(2, descDiffNo);
	for (int diffIndex = 0;diffIndex < totalDiff;diffIndex++) {
		unsigned int count = 0;
		int tempDiffIndex = diffIndex;
		// Checking bit by bit
		while (tempDiffIndex > 0)
		{
			// The second bit is 1
			if ((tempDiffIndex & 1) == 1) {
				count += 1;
				tempDiffIndex >>= 2;
			}
			// The second bit is 0
			else {
				tempDiffIndex >>= 1;
				// The first bit is 1
				if ((tempDiffIndex & 1) == 1)
					count += 1;
				tempDiffIndex >>= 1;
			}
		}
		*(LCDDiffLUTPtr + diffIndex) = count;
	}
}

/*=====MATCHING Methods=====*/
// Descriptor matching (RETURN-True:Not match, False: Match)
bool BackgroundSubtractorLCDP::DescriptorMatching(Descriptor *bgWord, Descriptor *currWord, const size_t pxPointer, const double LCDPThreshold, const double RGBThreshold,
	float &minRGBDistance, float &minLCDPDistance)
{
	bool result = false;

	if (clsLCDPDiffSwitch) {
		result = LCDPMatching(*(*bgWord).LCDP, *(*currWord).LCDP, pxPointer, LCDPThreshold, minLCDPDistance);
	}
	if (clsRGBDiffSwitch) {
		bool tempRGBResult = RGBMatching((*bgWord).rgb, (*currWord).rgb, RGBThreshold, minRGBDistance);
		result = (clsAndOrSwitch) ? (result&tempRGBResult) : (result | tempRGBResult);
	}
	if (clsRGBBrightPxSwitch) {
		result |= BrightRGBMatching((*bgWord).rgb, (*currWord).rgb, 0);
	}
	return result;
}
// LCD Matching (RETURN-True:Not match, False: Match)
bool BackgroundSubtractorLCDP::LCDPMatching(const unsigned bgLCD, const unsigned currLCD, const size_t pxPointer, const double LCDPThreshold, float &minDistance) {
	unsigned tempDistance = 0;
	for (size_t descIndex = 0;descIndex <= (*(descNbNo.data + pxPointer));descIndex++) {
		// Calculate the total number of bit that are different
		int Diff = std::abs(int(bgLCD - currLCD));
		tempDistance += *(LCDDiffLUTPtr + Diff);
	}
	minDistance = tempDistance / (descDiffNo / 2);
	return (tempDistance > LCDPThreshold) ? true : false;
}
// RGB Matching (RETURN-True:Not match, False: Match)
bool BackgroundSubtractorLCDP::RGBMatching(const unsigned bgRGB[], const unsigned currRGB[], const double RGBThreshold, float &minDistance)
{
	bool result = false;
	unsigned distance = UINT_MAX;
	for (int channel = 0;channel < 3;channel++) {
		const unsigned tempDistance = std::abs(double(bgRGB[channel] - currRGB[channel]));
		distance = tempDistance < distance ? tempDistance : distance;
		if (tempDistance > RGBThreshold) {
			result = true;
		}
	}
	minDistance = distance / UINT_MAX;
	return result;
}
// Bright Pixel (RETURN-True:Not match, False: Match)
bool BackgroundSubtractorLCDP::BrightRGBMatching(const unsigned bgRGB[], const unsigned currRGB[], const double BrightThreshold) {
	bool result = false;
	for (int channel = 0;channel < 3;channel++) {
		if ((currRGB[channel] - bgRGB[channel]) < BrightThreshold) {
			return true;
		}
	}
	return result;
}

/*=====POST-PROCESSING Methods=====*/
// Border line reconstruct
cv::Mat BackgroundSubtractorLCDP::BorderLineReconst(const cv::Mat inputMask)
{
	cv::Mat reconstructResult;
	reconstructResult.create(frameSize, CV_8UC1);
	reconstructResult = cv::Scalar_<uchar>(0);
	const size_t maxHeight = frameSize.height - 1;
	const size_t maxWidth = frameSize.width - 1;
	const size_t startIndexList_Y[4] = { 0,0,0,maxHeight };
	const size_t endIndexList_Y[4] = { 0,maxHeight,maxHeight,maxHeight };
	const size_t startIndexList_X[4] = { 0,0,maxWidth,0 };
	const size_t endIndexList_X[4] = { maxWidth,0,maxWidth,maxWidth };
	for (int line = 0;line < 4;line++) {
		uchar previousIndex = 0;
		size_t previousIndex_Y_start = startIndexList_Y[line];
		size_t previousIndex_Y_end = startIndexList_Y[line];
		size_t previousIndex_X_start = startIndexList_X[line];
		size_t previousIndex_X_end = startIndexList_X[line];
		bool previous = false;
		bool completeLine = false;
		for (int rowIndex = startIndexList_Y[line];rowIndex <= endIndexList_Y[line];rowIndex++) {
			//if (line == 1)
				for (int colIndex = startIndexList_X[line];colIndex <= endIndexList_X[line];colIndex++) {
					size_t pxPointer = (rowIndex*frameSize.width) + colIndex;
					const uchar currFGMask = *(resCurrFGMask.data + pxPointer);

					if ((currFGMask != previousIndex) && (currFGMask == 255)) {
						if (!previous) {
							previous = true;
							previousIndex = 255;
							if (((previousIndex_Y_start < previousIndex_Y_end) || (previousIndex_X_start < previousIndex_X_end)) && completeLine) {
								for (int recRowIndex = previousIndex_Y_start;recRowIndex <= previousIndex_Y_end;recRowIndex++) {
									for (int recColIndex = previousIndex_X_start;recColIndex <= previousIndex_X_end;recColIndex++) {
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
							if (ybalance > (frameSize.height / 2)) {
								previousIndex_Y_start = rowIndex;
								previousIndex_X_start = colIndex;
							}
							else if (xbalance > (frameSize.width / 2)) {
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
// Compensation with Motion Hist
cv::Mat BackgroundSubtractorLCDP::CompensationMotionHist(const cv::Mat T_1FGMask, const cv::Mat T_2FGMask, const cv::Mat currFGMask, const cv::Mat compensationThreshold) {
	cv::Mat compensationResult;
	compensationResult.create(frameSize, CV_8UC1);
	compensationResult = cv::Scalar_<uchar>::all(255);

	for (size_t modelIndex = 0;modelIndex < frameInitTotalPixel;modelIndex++) {
		if (!currFGMask.data[modelIndex]) {
			int totalFGMask = 0;

			for (size_t nbIndex = 0;nbIndex < 9;nbIndex++) {
				totalFGMask += T_1FGMask.data[pxInfoLUTPtr[modelIndex].nbIndex[nbIndex].dataIndex] / 255;
				totalFGMask += T_2FGMask.data[pxInfoLUTPtr[modelIndex].nbIndex[nbIndex].dataIndex] / 255;
				totalFGMask += currFGMask.data[pxInfoLUTPtr[modelIndex].nbIndex[nbIndex].dataIndex] / 255;
			}

			compensationResult.data[modelIndex] = ((totalFGMask / 26) > compensationThreshold.data[modelIndex]) ? 255 : 0;
		}
	}

	return compensationResult;
}

cv::Mat BackgroundSubtractorLCDP::ContourFill(const cv::Mat img) {
	cv::Mat input;
	cv::Mat output;
	cv::copyMakeBorder(img, input, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar());

	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::findContours(input, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	//
	// fill external contours
	// 
	if (!contours.empty() && !hierarchy.empty())
	{
		int d = contours.size();
		for (int idx = 0;idx < contours.size();idx++)
		{
			drawContours(input, contours, idx, cv::Scalar::all(255), CV_FILLED, 8);
		}
	}
	//step 3: remove the border
	input = input.rowRange(cv::Range(1, input.rows - 1));
	//cout<<m_with_border<<endl;
	input = input.colRange(cv::Range(1, input.cols - 1));
	input.copyTo(output);
	return output;
}

void BackgroundSubtractorLCDP::EditLCDPThreshold(const double inputLCDPThrehsold) {
	clsLCDPThreshold = inputLCDPThrehsold;
}
/*=====OTHERS Methods=====*/
// Save parameters
void BackgroundSubtractorLCDP::SaveParameter(std::string folderName) {
	std::ofstream myfile;
	myfile.open(folderName + "/parameter.txt", std::ios::app);
	myfile << "\n----METHOD----\n";
	myfile << "\nWITH FEEDBACK LOOP\n";
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
	myfile << "\nInitial model-observation similarity, d(LT):";
	myfile << int(*(resMeanFinalSegmRes_LT.data));
	myfile << "\nInitial model-observation similarity, d(ST):";
	myfile << int(*(resMeanFinalSegmRes_ST.data));
	myfile << "\nInitial segmentation noise accumulator, v:";
	float * variationModulator = (float*)(resVariationModulator.data);
	myfile << *(variationModulator);
	myfile << "\nInitial update rates, T(x):";
	myfile << *(updateRate);
	myfile << "\nInitial commonValue, R:";
	float * distThreshold = (float*)(resDistThreshold.data);
	myfile << *(distThreshold);
	myfile << "\nInitial blinking accumulate level:";
	float * blinkAccLevel = (float*)(upBlinkAccLevel.data);
	myfile << *(blinkAccLevel);
	myfile << "\nSegmentation noise accumulator increase value:";
	myfile << FEEDBACK_V_DECR;
	myfile << "\nSegmentation noise accumulator decrease value:";
	myfile << FEEDBACK_V_INCR;
	myfile << "\nLocal update rate change factor (Desc):";
	myfile << FEEDBACK_T_DECR;
	myfile << "\nLocal update rate change factor (Incr):";
	myfile << FEEDBACK_T_INCR;
	myfile << "\nLocal update rate (Lower):";
	myfile << upLearningRateLowerCap;
	myfile << "\nLocal update rate (Upper):";
	myfile << upLearningRateUpperCap;
	myfile << "\nLocal distance threshold change factor:";
	myfile << FEEDBACK_R_VAR;
	myfile << "\nSamples for moving averages:";
	myfile << upSamplesForMovingAvgs;
	myfile << "\nMaximum number of model:";
	myfile << WORDS_NO;

	myfile.close();
}
