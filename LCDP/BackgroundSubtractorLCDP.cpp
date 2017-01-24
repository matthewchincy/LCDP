#include "BackgroundSubtractorLCDP.h"
#include "RandUtils.h"

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
#define FEEDBACK_V_INCR  (1.00f)
#define FEEDBACK_V_DECR  (0.01f)
// Parameter used to scale dynamic distance threshold adjustments ('R(x)')
#define FEEDBACK_R_VAR = (0.01f)

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
	clsLCDPThreshold(36),
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
	delete bgWordPtrIter;
	delete[] currWordPtr;
	delete currWordPtrIter;
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
	bgWordPtr = new Descriptor[frameRoiTotalPixel*WORDS_NO];
	memset(bgWordPtr, 0, sizeof(Descriptor)*frameRoiTotalPixel*WORDS_NO);
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
	clsPersistenceThreshold = cv::Scalar((1 / descOffsetValue) / 4);
	
	/*=====POST-PROCESS Parameters=====*/
	// Size of median filter
	postMedianFilterSize = 9;
	// The compensation motion history threshold
	compensationThreshold.create(frameSize, CV_32FC1);
	compensationThreshold = cv::Scalar(0.7f);

	if ((frameRoiTotalPixel >= (frameInitTotalPixel / 2)) && (frameInitTotalPixel >= DEFAULT_FRAME_SIZE.area())) {
		/*=====POST-PROCESS Parameters=====*/
		double tempMedianFilterSize = std::min(double(14), floor((frameRoiTotalPixel / DEFAULT_FRAME_SIZE.area()) + 0.5) + postMedianFilterSize);
		// Current kernel size for median blur post-proc filtering
		postMedianFilterSize = tempMedianFilterSize;

		/*=====UPDATE Parameters=====*/
		// Specifies whether Tmin / Tmax scaling is enabled or not
		upLearningRateScalingSwitch = true;
		// Specifies the px update spread range
		upUse3x3Spread = !(frameInitTotalPixel> (DEFAULT_FRAME_SIZE.area() * 2));
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
	resUnstableRegionMask = cv::Scalar_<uchar>(0);
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
	resBlinksFrame = cv::Scalar_<uchar>(0);

	// Minimum RGB distance
	resMinRGBDistance.create(frameSize, CV_32FC1);
	resMinRGBDistance = cv::Scalar(1.0f);
	// Minimum LCD distance
	resMinLCDPDistance.create(frameSize, CV_32FC1);
	resMinLCDPDistance = cv::Scalar(1.0f);
	// Current foreground mask
	resCurrFGMask.create(frameSize, CV_8UC1);
	resCurrFGMask = cv::Scalar_<uchar>(0);
	// Previous foreground mask
	resLastFGMask.create(frameSize, CV_8UC1);
	resLastFGMask = cv::Scalar_<uchar>(0);
	// The foreground mask generated by the method at [t-1] (without post-proc, used for blinking px detection)
	resLastRawFGMask.create(frameSize, CV_8UC1);
	resLastRawFGMask = cv::Scalar_<uchar>(0);
	// The blink mask generated by the method at [t-1] (without post-proc, used for blinking px detection)
	resLastRawFGBlinkMask.create(frameSize, CV_8UC1);
	resLastRawFGBlinkMask = cv::Scalar_<uchar>(0);
	// t-1 foreground mask
	resT_1FGMask.create(frameSize, CV_8UC1);
	resT_1FGMask = cv::Scalar_<uchar>(0);
	// t-2 foreground mask
	resT_2FGMask.create(frameSize, CV_8UC1);
	resT_2FGMask = cv::Scalar_<uchar>(0);
	// Last foreground dilated mask
	resLastFGMaskDilated.create(frameSize, CV_8UC1);
	resLastFGMaskDilated = cv::Scalar_<uchar>(0);
	// Last foreground dilated inverted mask
	resLastFGMaskDilatedInverted.create(frameSize, CV_8UC1);
	resLastFGMaskDilatedInverted = cv::Scalar_<uchar>(0);
	// Flooded holes foreground mask
	resFGMaskFloodedHoles.create(frameSize, CV_8UC1);
	resFGMaskFloodedHoles = cv::Scalar_<uchar>(0);
	// Pre flooded holes foreground mask
	resFGMaskPreFlood.create(frameSize, CV_8UC1);
	resFGMaskPreFlood = cv::Scalar_<uchar>(0);
	// Current raw foreground blinking mask
	resCurrRawFGBlinkMask.create(frameSize, CV_8UC1);
	resCurrRawFGBlinkMask = cv::Scalar_<uchar>(0);	
	
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
			// Check ROI
			if (frameRoi.data[pxPointer]) {
				// Start index for pixel's model
				pxInfoLUTPtr[pxPointer].startModelIndex = modelPointer*WORDS_NO;
				++modelPointer;
			}
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

void BackgroundSubtractorLCDP::Process(const cv::Mat INPUT_IMG, cv::Mat &OUTPUT_IMG)
{
	// PRE PROCESSING
	cv::GaussianBlur(INPUT_IMG, INPUT_IMG, preGaussianSize, 0, 0);

	// FG DETECTION PROCESS
	for (size_t nModelIter = 0; nModelIter < frameInitTotalPixel; ++nModelIter) {
		// Start index of the model of the current pixel
		const size_t nLocalDictIdx = nModelIter*WORDS_NO;
		// LCD descriptor threshold
		const double LCDPThreshold = clsLCDPThreshold + (std::pow(1.5, floor(resDistThreshold.data[nModelIter] + 0.5)));
		// RGB descriptor threshold
		const double RGBThreshold = floor(clsRGBThreshold*resDistThreshold.data[nModelIter]);
		Descriptor* currWord = &currWordPtr[nModelIter];
		float potentialPersistenceSum = 0.0f;
		float lastWordWeight = FLT_MAX;
		size_t nLocalWordIdx = 0;
		while (nLocalWordIdx<WORDS_NO && potentialPersistenceSum<clsPersistenceThreshold) {
			Descriptor* bgWord = &bgWordPtr[nLocalDictIdx + nLocalWordIdx];
			const float currWordWeight = GetLocalWordWeight(bgWord, frameIndex, descOffsetValue);
			if (bgWord
				&& (!DescriptorMatching(bgWord, currWord, LCDPThreshold, RGBThreshold))) {
				bgWord->frameCount += 1;
				bgWord->q = frameIndex;
				potentialPersistenceSum += currWordWeight
			}
			if (currWordWeight>lastWordWeight) {
				std::swap(bgWordPtr[nLocalDictIdx + nLocalWordIdx], bgWordPtr[nLocalDictIdx + nLocalWordIdx - 1]);
			}
			else
				lastWordWeight = currWordWeight;
			++nLocalWordIdx;
		}

		// BG Pixels
		if (potentialPersistenceSum >= clsPersistenceThreshold) {
			resCurrFGMask.data[nModelIter] = 0;
		}
		// FG Pixels
		else {
			resCurrFGMask.data[nModelIter] = 255;

		}
		// UPDATE PROCESS

	}

	cv::Mat compensationResult;


	// POST PROCESSING
	cv::bitwise_xor(resCurrFGMask, resLastRawFGMask, resCurrRawFGBlinkMask);
	cv::bitwise_or(resCurrRawFGBlinkMask, resLastRawFGBlinkMask, resBlinksFrame);
	resCurrRawFGBlinkMask.copyTo(resLastRawFGBlinkMask);
	resCurrFGMask.copyTo(resLastRawFGMask);
	cv::morphologyEx(resCurrFGMask, resCurrFGMask, cv::MORPH_OPEN, cv::Mat());
	cv::Mat borderLineReconstructResult = BorderLineReconst();
	CompensationMotionHist(resT_1FGMask, resT_2FGMask,resCurrFGMask,&compensationResult);
	cv::bitwise_or(resCurrFGMask, borderLineReconstructResult, resCurrFGMask);
	cv::morphologyEx(resCurrFGMask, resFGMaskPreFlood, cv::MORPH_CLOSE, cv::Mat());
	resFGMaskPreFlood.copyTo(resFGMaskFloodedHoles);
	cv::floodFill(resFGMaskFloodedHoles, cv::Point(0, 0), UCHAR_MAX);
	cv::bitwise_not(resFGMaskFloodedHoles, resFGMaskFloodedHoles);
	cv::erode(resFGMaskPreFlood, resFGMaskPreFlood, cv::Mat(), cv::Point(-1, -1), 3);
	cv::bitwise_or(resCurrFGMask, resFGMaskFloodedHoles, resCurrFGMask);
	cv::bitwise_or(resCurrFGMask, resFGMaskPreFlood, resCurrFGMask);
	cv::bitwise_and(resCurrFGMask, compensationResult, resCurrFGMask);

	borderLineReconstructResult = BorderLineReconst();
	cv::bitwise_or(resCurrFGMask, borderLineReconstructResult, resCurrFGMask);
	cv::floodFill(resCurrFGMask, cv::Point(0, 0), UCHAR_MAX);
	cv::bitwise_not(resCurrFGMask, resCurrFGMask);

	cv::medianBlur(resCurrFGMask, resLastFGMask, postMedianFilterSize);
	cv::dilate(resLastFGMask, resLastFGMaskDilated, cv::Mat(), cv::Point(-1, -1), 3);
	cv::bitwise_and(resBlinksFrame, resLastFGMaskDilatedInverted, resBlinksFrame);
	cv::bitwise_not(resLastFGMaskDilated, resLastFGMaskDilatedInverted);
	cv::bitwise_and(resBlinksFrame, resLastFGMaskDilatedInverted, resBlinksFrame);
	resLastFGMask.copyTo(resCurrFGMask);

	resT_1FGMask.copyTo(resT_2FGMask);
	resLastFGMask.copyTo(resT_1FGMask);
	// Frame Index
	frameIndex++;
}

/*=====METHODS=====*/
/*=====DEFAULT methods=====*/
// Refreshes all samples based on the last analyzed frame
void BackgroundSubtractorLCDP::RefreshModel(float refreshFraction, bool forceUpdateSwitch)
{
	const size_t noSampleBeRefresh = refreshFraction<1.0f ? (size_t)(refreshFraction*WORDS_NO) : WORDS_NO;
	const size_t refreshStartPos = refreshFraction<1.0f ? rand() % WORDS_NO : 0;
	for (size_t modelPointer = 0; modelPointer < frameRoiTotalPixel; ++modelPointer) {
		const size_t pixelPointer = pxInfoLUTPtr[modelPointer].dataIndex;
		if (!forceUpdateSwitch || !resLastFGMask.data[pixelPointer]) {
			// Start index of the model of the current pixel
			const size_t startModelIndex = pxInfoLUTPtr[modelPointer].startModelIndex;
			// LCD descriptor threshold
			const double LCDPThreshold = clsLCDPThreshold + (std::pow(1.5, floor(resDistThreshold.data[pixelPointer] + 0.5)));
			// RGB descriptor threshold
			const double RGBThreshold = floor(clsRGBThreshold*resDistThreshold.data[pixelPointer]);

			for (size_t currModelIndex = refreshStartPos; currModelIndex < refreshStartPos + noSampleBeRefresh; ++currModelIndex) {
				Descriptor currPixelDesc = *(bgWordPtr + startModelIndex + currModelIndex);
				cv::Point sampleCoor;
				getRandSamplePosition(sampleCoor, cv::Point(pxInfoLUTPtr[modelPointer].coor_x, pxInfoLUTPtr[modelPointer].coor_y), 0, frameSize);
				const size_t samplePxIndex = frameSize.width*sampleCoor.y + sampleCoor.x;
				if (forceUpdateSwitch || !resLastFGMaskDilated.data[samplePxIndex]) {
					Descriptor* currWord = &currWordPtr[samplePxIndex];
					bool bFoundUninitd = false;
					size_t nLocalWordIdx;
					for (nLocalWordIdx = 0; nLocalWordIdx < WORDS_NO; ++nLocalWordIdx) {
						Descriptor* bgWord = &bgWordPtr[nLocalDictIdx + nLocalWordIdx];
						if (bgWord
							&& (!DescriptorMatching(bgWord, currWord, LCDPThreshold, RGBThreshold))) {
							bgWord->frameCount += 1;
							bgWord->q = frameIndex;
							break;
						}
						else if (!bgWord)
							bFoundUninitd = true;
					}
					if (nLocalWordIdx == WORDS_NO) {
						nLocalWordIdx = WORDS_NO - 1;
						Descriptor* bgWord = bFoundUninitd ? bgWordPtrIter++ : &bgWordPtr[nLocalDictIdx + nLocalWordIdx];
						*(bgWord->LCDP) = *(currWord->LCDP);
						bgWord->frameCount = 1;
						bgWord->p = frameIndex;
						bgWord->q = frameIndex;
						*(bgWord->rgb) = *(currWord->rgb);
					}
					while (nLocalWordIdx>0 && (!&bgWordPtr[nLocalDictIdx + nLocalWordIdx - 1] || GetLocalWordWeight(&bgWordPtr[nLocalDictIdx + nLocalWordIdx], frameIndex, descOffsetValue)>GetLocalWordWeight(&bgWordPtr[nLocalDictIdx + nLocalWordIdx - 1], frameIndex, descOffsetValue))) {
						std::swap(bgWordPtr[nLocalDictIdx + nLocalWordIdx], bgWordPtr[nLocalDictIdx + nLocalWordIdx - 1]);
						--nLocalWordIdx;
					}
				}
			}
		}
	}
}

/*=====DESCRIPTOR Methods=====*/
// Descriptor Generator-Generate pixels' descriptor (RGB+LCDP)
void BackgroundSubtractorLCDP::DescriptorGenerator(const cv::Mat inputFrame,const PxInfoBase *pxInfoPtr,
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
	const float colorDiffRatio = descColorDiffRatio.data[(*pxInfoPtr).dataIndex];
	const int ratio = std::max(3, (int)(colorDiffRatio*(currPixelTotalRGB / 3)));
	const int nRatio = -ratio;
	const int nbNo = descNbNo[(*pxInfoPtr).dataIndex];
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
	return (float)((*wordPtr).frameCount) / (((*wordPtr).q - (*wordPtr).p) + (currFrameIndex - (*wordPtr).q)*offsetValue);
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
		while (tempDiffIndex>0)
		{
			// The second bit is 1
			if ((tempDiffIndex & 1)==1) {
				count += 1;
				tempDiffIndex >>= 2;
			}
			// The second bit is 0
			else {
				tempDiffIndex >>= 1;
				// The first bit is 1
				if ((tempDiffIndex & 1)==1)
					count += 1;
				tempDiffIndex >>= 1;
			}			
		}
		*(LCDDiffLUTPtr+diffIndex) = count;
	}	
}

/*=====MATCHING Methods=====*/
// Descriptor matching (RETURN-True:Not match, False: Match)
bool BackgroundSubtractorLCDP::DescriptorMatching(Descriptor *bgWord, Descriptor *currWord, const double LCDPThreshold, const double RGBThreshold,
	unsigned &minRGBDistance, unsigned &minLCDPDistance)
{
	bool result = false;

	if (clsLCDPDiffSwitch) {
		result = LCDPMatching(*(*bgWord).LCDP, *(*currWord).LCDP, LCDPThreshold, minLCDPDistance);
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
bool BackgroundSubtractorLCDP::LCDPMatching(const unsigned bgLCD, const unsigned currLCD, const double LCDPThreshold, unsigned &minDistance) {
	// Calculate the total number of bit that are different
	int Diff = std::abs(int(bgLCD - currLCD));
	minDistance = *(LCDDiffLUT + Diff);
	return (minDistance>LCDPThreshold) ? true : false;
}
// RGB Matching (RETURN-True:Not match, False: Match)
bool BackgroundSubtractorLCDP::RGBMatching(const unsigned bgRGB[], const unsigned currRGB[], const double RGBThreshold, unsigned &minDistance)
{
	bool result = false;
	unsigned distance = UINT_MAX;
	for (int channel = 0;channel < 3;channel++) {
		const unsigned tempDistance = std::abs(bgRGB[channel] - currRGB[channel]);
		distance = tempDistance < distance ? tempDistance : distance;
		if (tempDistance > RGBThreshold) {
			result = true;
		}
	}
	minDistance = distance;
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
		for (int rowIndex = startIndexList_Y[line];startIndexList_Y[line] <= endIndexList_Y[line];rowIndex++) {
			for (int colIndex = startIndexList_Y[line];startIndexList_X[line] <= endIndexList_X[line];colIndex++) {
				if ((resCurrFGMask.at<uchar>(rowIndex, colIndex) != previousIndex) && (resCurrFGMask.at<uchar>(rowIndex, colIndex) == 255)) {
					if (!previous) {
						previous = true;
						previousIndex = 255;
						if (((previousIndex_Y_start < previousIndex_Y_end) || (previousIndex_X_start < previousIndex_X_end)) && completeLine) {
							for (int recRowIndex = previousIndex_Y_start;previousIndex_Y_start <= previousIndex_Y_end;recRowIndex++) {
								for (int recColIndex = previousIndex_X_start;previousIndex_X_start <= previousIndex_X_end;recColIndex++) {
									reconstructResult.at<uchar>(recRowIndex, recColIndex) = 255;
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
				else if ((resCurrFGMask.at<uchar>(rowIndex, colIndex) != previousIndex) && (resCurrFGMask.at<uchar>(rowIndex, colIndex) == 0)) {
					if (previous) {
						previous = false;
						previousIndex = 0;
						completeLine = true;
					}
					previousIndex_Y_end = rowIndex;
					previousIndex_X_end = colIndex;
				}
				else if (resCurrFGMask.at<uchar>(rowIndex, colIndex) == 255) {
					if (previous) {
						previousIndex_Y_start = rowIndex;
						previousIndex_X_start = colIndex;
					}
				}
				else if (resCurrFGMask.at<uchar>(rowIndex, colIndex) == 0) {
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
cv::Mat BackgroundSubtractorLCDP::CompensationMotionHist(const cv::Mat T_1FGMask, const cv::Mat T_2FGMask, const cv::Mat currFGMask, const float currCompensationThreshold) {
	cv::Mat compensationResult;
	compensationResult.create(frameSize, CV_8UC1);
	compensationResult = cv::Scalar_<uchar>(0);

	for (size_t modelIndex = 0;modelIndex < frameInitTotalPixel;modelIndex++) {
		if (!currFGMask.data[modelIndex]) {
			int totalFGMask = 0;

			for (size_t nbIndex = 0;nbIndex < 9;nbIndex++) {
				totalFGMask += T_1FGMask.data[pxInfoLUTPtr[modelIndex].nbIndex[nbIndex].dataIndex] / 255;
				totalFGMask += T_2FGMask.data[pxInfoLUTPtr[modelIndex].nbIndex[nbIndex].dataIndex] / 255;
				totalFGMask += currFGMask.data[pxInfoLUTPtr[modelIndex].nbIndex[nbIndex].dataIndex] / 255;
			}

			compensationResult.data[modelIndex] = ((totalFGMask / 26) > currCompensationThreshold) ? 255 : 0;
		}
	}


}
