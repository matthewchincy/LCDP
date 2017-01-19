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
	// Total number of words per pixel
	WORDS_NO(inputWordsNo),
	// Store the background's word and it's iterator
	bgWordPtr(nullptr),
	bgWordPtrIter(nullptr),
	// Store the currect frame's word and it's iterator
	currWordPtr(nullptr),
	currWordPtrIter(nullptr),
	// Frame index
	frameIndex(0),
	// Size of neighbourhood 3(3x3)/5(5x5)
	descNbSize(5),
	// Total number of LCD's neighbour 8(3x3)/16(5x5)
	descNbNo(16),
	// LCD colour differences ratio
	descColorDiffRatio(0.05),
	// Total number of LCD differences per pixel
	descDiffNo(9),
	// Persistence's offset value;
	descOffsetValue(1000),
	/*=====CLASSIFIER Parameters=====*/
	// RGB differences threshold
	clsRGBThreshold(10),
	// RGB detection switch
	clsRGBDiffSwitch(false),
	// RGB bright pixel switch
	clsRGBBrightPxSwitch(true),
	// LCDP differences threshold
	clsLCDPThreshold(36),
	// Maximum number of LCDP differences threshold
	clsLCDPMaxThreshold(72),
	// LCDP detection switch
	clsLCDPDiffSwitch(true),
	// LCDP detection AND (true) OR (false) switch
	clsAndOrSwitch(true),
	// Neighbourhood matching switch
	clsNbMatchSwitch(true),
	// Total number of neighbour 8(3x3)/16(5x5)
	clsNbNo(16),
	/*=====FRAME Parameters=====*/
	// ROI frame
	frameRoi(inputROIFrame),
	// Size of region of interest
	frameRoiSize(inputFrameSize),
	// Size of input frame
	frameInitSize(inputFrameSize),
	// Total number of pixel of region of interest
	frameRoiTotalPixel(inputFrameSize.area()),
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

}

void BackgroundSubtractorLCDP::Initialize(const cv::Mat img_input, cv::Mat roi_input)
{

	bgWordPtr = new Descriptor[frameRoiTotalPixel*WORDS_NO];
	memset(bgWordPtr, 0, sizeof(Descriptor)*frameRoiTotalPixel*WORDS_NO);
	bgWordPtrIter = bgWordPtr;
	currWordPtr = new Descriptor[frameRoiTotalPixel];
	memset(currWordPtr, 0, sizeof(Descriptor)*frameRoiTotalPixel);
	currWordPtrIter = currWordPtr;

	modelInitCheck = true;

	// Frame Index
	frameIndex = 1;

	/*=====CLASSIFIER Parameters=====*/
	// Total number of neighbour 8(3x3)/16(5x5)
	clsNbNo =16;

	/*=====POST-PROCESS Parameters=====*/
	// Size of median filter
	postMedianFilterSize = 9;
	if ((frameRoiTotalPixel >= (frameInitTotalPixel / 2)) && (frameInitTotalPixel >= DEFAULT_FRAME_SIZE.area())) {
		// Specifies whether Tmin / Tmax scaling is enabled or not
		upLearningRateScalingSwitch = true;
		// Specifies the px update spread range
		upUse3x3Spread = !(frameInitTotalPixel> (DEFAULT_FRAME_SIZE.area() * 2));
		size_t tempMedianFilterSize = std::min(double(14), floor((frameRoiTotalPixel / DEFAULT_FRAME_SIZE.area()) + 0.5) + postMedianFilterSize);
		// Current kernel size for median blur post-proc filtering
		postMedianFilterSize = tempMedianFilterSize;
		// Current learning rate caps
		upLearningRateLowerCap = FEEDBACK_T_LOWER;
		upLearningRateUpperCap = FEEDBACK_T_UPPER;
	}
	else {
		// Specifies whether Tmin / Tmax scaling is enabled or not
		upLearningRateScalingSwitch = false;
		// Specifies the px update spread range
		upUse3x3Spread = true;
		// Current kernel size for median blur post-proc filtering
		postMedianFilterSize = postMedianFilterSize;
		// Current learning rate caps
		upLearningRateLowerCap = FEEDBACK_T_LOWER * 2;
		upLearningRateUpperCap = FEEDBACK_T_UPPER * 2;
	}

	/*=====UPDATE Parameters=====*/
	// Initial blinking accumulate level
	upBlinkAccLevel.create(frameInitSize, CV_32FC1);
	upBlinkAccLevel = cv::Scalar(1.0f);	
	// Total number of neighbour undergo neighbourhood updates 8(3X3)/16(5X5)
	upNbNo = 8;

	upSamplesForMovingAvgs = 100;

	/*=====RESULTS=====*/
	// Per-pixel distance thresholds ('R(x)', but used as a relative value to determine both 
	// intensity and descriptor variation thresholds)
	resDistThreshold.create(frameInitSize, CV_32FC1);
	resDistThreshold = cv::Scalar(1.0f);
	// A lookup map used to keep track of unstable regions (based on segm. noise & local dist. thresholds)
	resUnstableRegionMask.create(frameInitSize, CV_8UC1);
	resUnstableRegionMask = cv::Scalar_<uchar>(0);
	// Per-pixel mean raw segmentation results (used to detect unstable segmentation regions)
	resMeanRawSegmRes_LT.create(frameInitSize, CV_32FC1);
	resMeanRawSegmRes_LT = cv::Scalar(0.0f);
	resMeanRawSegmRes_ST.create(frameInitSize, CV_32FC1);
	resMeanRawSegmRes_ST = cv::Scalar(0.0f);
	// Per-pixel mean raw segmentation results (used to detect unstable segmentation regions)
	resMeanFinalSegmRes_LT.create(frameInitSize, CV_32FC1);
	resMeanFinalSegmRes_LT = cv::Scalar(0.0f);
	resMeanFinalSegmRes_ST.create(frameInitSize, CV_32FC1);
	resMeanFinalSegmRes_ST = cv::Scalar(0.0f);
	// Per-pixel update rates('T(x)', which contains pixel - level 'sigmas')
	resUpdateRate.create(frameInitSize, CV_32FC1);
	resUpdateRate = cv::Scalar(upLearningRateLowerCap);
	// Per-pixel mean minimal distances from the model ('D_min(x)', used to control 
	// variation magnitude and direction of 'T(x)' and 'R(x)')
	resMeanMinDist_LT.create(frameInitSize, CV_32FC1);
	resMeanMinDist_LT = cv::Scalar(0.0f);
	resMeanMinDist_ST.create(frameInitSize, CV_32FC1);
	resMeanMinDist_ST = cv::Scalar(0.0f);
	// Per-pixel distance variation modulators ('v(x)', relative value used to modulate 'R(x)' 
	// and 'T(x)' variations)
	resVariationModulator.create(frameInitSize, CV_32FC1);
	resVariationModulator = cv::Scalar(10);
	// Per-pixel blink detection map ('Z(x)')
	resBlinksFrame.create(frameInitSize, CV_8UC1);
	resBlinksFrame = cv::Scalar_<uchar>(0);

	// Minimum RGB distance
	resMinRGBDistance.create(frameInitSize, CV_32FC1);
	resMinRGBDistance = cv::Scalar(1);
	// Minimum LCD distance
	resMinLCDPDistance.create(frameInitSize, CV_32FC1);
	resMinLCDPDistance = cv::Scalar(1);
	// Current foreground mask
	resCurrFGMask.create(frameInitSize, CV_8UC1);
	resCurrFGMask = cv::Scalar_<uchar>(0);
	// Previous foreground mask
	resLastFGMask.create(frameInitSize, CV_8UC1);
	resLastFGMask = cv::Scalar_<uchar>(0);
	// The foreground mask generated by the method at [t-1] (without post-proc, used for blinking px detection)
	resLastRawFGMask.create(frameInitSize, CV_8UC1);
	resLastRawFGMask = cv::Scalar_<uchar>(0);
	// The blink mask generated by the method at [t-1] (without post-proc, used for blinking px detection)
	resLastRawFGBlinkMask.create(frameInitSize, CV_8UC1);
	resLastRawFGBlinkMask = cv::Scalar_<uchar>(0);
	// t-1 foreground mask
	resT_1FGMask.create(frameInitSize, CV_8UC1);
	resT_1FGMask = cv::Scalar_<uchar>(0);
	// t-2 foreground mask
	resT_2FGMask.create(frameInitSize, CV_8UC1);
	resT_2FGMask = cv::Scalar_<uchar>(0);
	resLastFGMaskDilated.create(frameInitSize, CV_8UC1);
	resLastFGMaskDilated = cv::Scalar_<uchar>(0);
	resLastFGMaskDilatedInverted.create(frameInitSize, CV_8UC1);
	resLastFGMaskDilatedInverted = cv::Scalar_<uchar>(0);
	resFGMaskFloodedHoles.create(frameInitSize, CV_8UC1);
	resFGMaskFloodedHoles = cv::Scalar_<uchar>(0);
	resFGMaskPreFlood.create(frameInitSize, CV_8UC1);
	resFGMaskPreFlood = cv::Scalar_<uchar>(0);
	resCurrRawFGBlinkMask.create(frameInitSize, CV_8UC1);
	resCurrRawFGBlinkMask = cv::Scalar_<uchar>(0);
	 
	// Declaring the neighbourhood offset value
	nbOffset[0] = cv::Point(-1, -1);
	nbOffset[1] = cv::Point(0, -1);
	nbOffset[2] = cv::Point(1, -1);
	nbOffset[3] = cv::Point(-1, 0);
	nbOffset[4] = cv::Point(1, 0);
	nbOffset[5] = cv::Point(-1, 1);
	nbOffset[6] = cv::Point(0, 1);
	nbOffset[7] = cv::Point(1, 1);
	nbOffset[8] = cv::Point(-2, -2);
	nbOffset[9] = cv::Point(0, -2);
	nbOffset[10] = cv::Point(2, -2);
	nbOffset[11] = cv::Point(-2, 0);
	nbOffset[12] = cv::Point(2, 0);
	nbOffset[13] = cv::Point(-2, 2);
	nbOffset[14] = cv::Point(0, 2);
	nbOffset[15] = cv::Point(2, 2);
	pxInfoLUT = new PxInfoBase[frameInitTotalPixel];
	size_t nPxIter = 0;
	size_t nModelIter = 0;
	for (size_t rowIndex = 0;rowIndex < frameInitSize.height;rowIndex++) {
		for (size_t colIndex = 0;colIndex <frameInitSize.width;colIndex++) {
			// Check ROI
			if (frameRoi.data[nPxIter]) {
				pxInfoLUT[nPxIter].imgCoord_Y = (int)rowIndex;
				pxInfoLUT[nPxIter].imgCoord_X = (int)colIndex;
				pxInfoLUT[nPxIter].nModelIdx = nModelIter;
				GenerateNbOffset(&pxInfoLUT[nPxIter]);
				// RGB iteration
				const size_t nPxRGBIter = nPxIter * 3;
				// Descriptor iteration
				const size_t nDescRGBIter = nPxRGBIter * 2;
				DescriptorGenerator(img_input, nPxRGBIter, &pxInfoLUT[nPxIter], currWordPtrIter++);
				++nModelIter;
			}
			nPxIter++;			
		}
	}

	RefreshModel(1, 0);
}

// Refresh model - fill up the background model
void BackgroundSubtractorLCDP::RefreshModel(float refreshFraction, bool forceFGUpdateSwitch)
{
	const size_t noSampleBeRefresh = refreshFraction<1.0f ? (size_t)(refreshFraction*WORDS_NO) : WORDS_NO;
	const size_t refreshStartPos = refreshFraction<1.0f ? rand() % WORDS_NO : 0;
	for (size_t nModelIter = 0; nModelIter < frameInitTotalPixel; ++nModelIter) {
		if (!forceFGUpdateSwitch||!resLastFGMask.data[nModelIter]) {
			for (size_t nCurrModelIdx = refreshStartPos; nCurrModelIdx < refreshStartPos + noSampleBeRefresh; ++nCurrModelIdx) {
				int nSampleImgCoord_Y, nSampleImgCoord_X;
				getRandSamplePosition(nSampleImgCoord_X, nSampleImgCoord_Y, pxInfoLUT[nModelIter].imgCoord_X, pxInfoLUT[nModelIter].imgCoord_Y, 0, frameInitSize);
				const size_t nSamplePxIdx = frameInitSize.width*nSampleImgCoord_Y + nSampleImgCoord_X;
				if (forceFGUpdateSwitch || !resLastFGMaskDilated.data[nSamplePxIdx]) {
					const size_t nSamplePxRGBIdx = nSamplePxIdx * 3;

				}
			}
		}
	}
}


void BackgroundSubtractorLCDP::Process(const cv::Mat INPUT_IMG, cv::Mat &OUTPUT_IMG)
{
	// PRE PROCESSING
	cv::GaussianBlur(INPUT_IMG, INPUT_IMG, preGaussianSize, 0, 0);




	// POST PROCESSING
	cv::bitwise_xor(resCurrFGMask, resLastRawFGMask, resCurrRawFGBlinkMask);
	cv::bitwise_or(resCurrRawFGBlinkMask, resLastRawFGBlinkMask, resBlinksFrame);
	resCurrRawFGBlinkMask.copyTo(resLastRawFGBlinkMask);
	resCurrFGMask.copyTo(resLastRawFGMask);
	cv::morphologyEx(resCurrFGMask, resCurrFGMask, cv::MORPH_OPEN, cv::Mat());
	cv::Mat borderLineReconstructResult = BorderLineReconst();
	cv::Mat compensationResult = CompensationMotionHist();
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

// Compensation with Motion Hist
void BackgroundSubtractorLCDP::CompensationMotionHist(const cv::Mat T_1FGMask, const cv::Mat T_2FGMask, const cv::Mat currFGMask,cv::Mat * compensationResult) {
	
	*compensationResult = cv::Scalar_<uchar>(0);

	for (size_t modelIndex = 0;modelIndex < frameInitTotalPixel;modelIndex++) {
		int totalFGMask=0;
		
		for (size_t nbIndex = 0;nbIndex < 9;nbIndex++) {
			//totalFGMask+= T_1FGMask.data[pxInfoLUT[modelIndex].pxIndex[nbIndex]]
		}
	}
	

}

// Generate LCD Descriptor
void BackgroundSubtractorLCDP::LCDGenerator(const cv::Mat inputFrame, const PxInfoBase *pxInfoLUT, Descriptor *tempWord)
{
	size_t nModelIndex = (*pxInfoLUT).nModelIdx;
	int R_CURR, G_CURR, B_CURR;
	int R_NB, G_NB, B_NB;
	int currPixelTotalRGB = 0;
	B_CURR = inputFrame.data[nModelIndex];
	G_CURR = inputFrame.data[nModelIndex + 1];
	R_CURR = inputFrame.data[nModelIndex + 2];
	for (int channel = 0;channel < 3;channel++) {
		currPixelTotalRGB += inputFrame.data[nModelIndex + channel];
	}
	const int ratio = std::max(3, (int)(descColorDiffRatio*(currPixelTotalRGB / 3)));
	const int nRatio = -ratio;
	for (int nbPixelIndex = 0;nbPixelIndex < descNbNo;nbPixelIndex++) {
		// Obtain neighbourhood pixel's value
		B_NB =inputFrame.data[(*pxInfoLUT).pxIndex[nbPixelIndex].B];
		G_NB = inputFrame.data[(*pxInfoLUT).pxIndex[nbPixelIndex].G];
		R_NB = inputFrame.data[(*pxInfoLUT).pxIndex[nbPixelIndex].R];

		uint tempResult = 0;
		// B_NB - G_CURR
		const int tempBNB_GCURR = B_NB - G_CURR;
		tempResult += ((tempBNB_GCURR > ratio) ? 1 : ((tempBNB_GCURR < nRatio) ? 3 : 0));
		// B_NB - R_CURR
		const int tempBNB_RCURR = B_NB - R_CURR;
		tempResult += ((tempBNB_RCURR > ratio) ? 1 : ((tempBNB_RCURR < nRatio) ? 3 : 0))<<2;
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
		(*tempWord).LCDP[nbPixelIndex] = tempResult;
	}

}

bool BackgroundSubtractorLCDP::DescriptorMatching()
{
	return false;
}

// Generate neighbourhood pixel offset value
void BackgroundSubtractorLCDP::GenerateNbOffset(PxInfoBase* pxInfoLUT)
{
	const int maxWidth = frameInitSize.width-1;
	const int maxHeight = frameInitSize.height - 1;
	const int x = (*pxInfoLUT).imgCoord_X;
	const int y = (*pxInfoLUT).imgCoord_Y;
	for (int nbIndex = 0;nbIndex < 16;nbIndex++) {
		const int nbPixel_X = std::min(maxWidth, std::max(0, (x + nbOffset[nbIndex].x)));
		const int nbPixel_Y = std::min(maxHeight, std::max(0, (y + nbOffset[nbIndex].y)));
		(*pxInfoLUT).pxIndex[nbIndex].B = (3 * ((nbPixel_Y*(maxWidth + 1)) + (nbPixel_X + 1))) - 3;
		(*pxInfoLUT).pxIndex[nbIndex].G = (3 * ((nbPixel_Y*(maxWidth + 1)) + (nbPixel_X + 1))) - 2;
		(*pxInfoLUT).pxIndex[nbIndex].R = (3 * ((nbPixel_Y*(maxWidth + 1)) + (nbPixel_X + 1)))-1;
	}
}

// Generate pixels' descriptor (RGB+LCDP)
void BackgroundSubtractorLCDP::DescriptorGenerator(const cv::Mat inputFrame,const size_t nPxRGBIter ,const PxInfoBase *pxInfoLUT, Descriptor * tempWord)
{
	(*tempWord).frameCount = 1;
	for (int channel = 0;channel < 3;channel++) {
		(*tempWord).rgb[channel] = inputFrame.data[nPxRGBIter + channel];
	}
	(*tempWord).p = frameIndex;
	(*tempWord).q = frameIndex;
	LCDGenerator(inputFrame, pxInfoLUT, tempWord);
}

// Border line reconstruct
cv::Mat BackgroundSubtractorLCDP::BorderLineReconst()
{
	cv::Mat reconstructResult;
	/*reconstructResult.create(frameInitSize, CV_8UC1);
	reconstructResult = cv::Scalar_<uchar>(0);
	size_t height = frameInitSize.height-1;
	size_t width = frameInitSize.width-1;
	size_t startIndexList_Y[4] = { 0,0,0,height };
	size_t endIndexList_Y[4] = { 0,height,height,height };
	size_t startIndexList_X[4] = { 0,0,width,0 };
	size_t endIndexList_X[4] = { width,0,width,width };
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
				if ((result.currFGMask.at<uchar>(rowIndex, colIndex) != previousIndex)&& (result.currFGMask.at<uchar>(rowIndex, colIndex) == 255)) {
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
				}else if((result.currFGMask.at<uchar>(rowIndex, colIndex) != previousIndex) && (result.currFGMask.at<uchar>(rowIndex, colIndex) == 0)) {
					if (previous) {
						previous = false;
						previousIndex = 0;
						completeLine = true;
					}
					previousIndex_Y_end = rowIndex;
					previousIndex_X_end = colIndex;
				}
				else if(result.currFGMask.at<uchar>(rowIndex, colIndex) == 255) {
					if (previous) {
						previousIndex_Y_start = rowIndex;
						previousIndex_X_start = colIndex;
					}
				}
				else if (result.currFGMask.at<uchar>(rowIndex, colIndex) == 0) {
					if (!previous) {
						size_t ybalance = previousIndex_Y_end - previousIndex_Y_start;
						size_t xbalance = previousIndex_X_end - previousIndex_X_start;
						if (ybalance > (methodParam.frame.initFrameSize.height / 2)) {
							previousIndex_Y_start = rowIndex;
							previousIndex_X_start = colIndex;
						}else if (xbalance > (methodParam.frame.initFrameSize.width / 2)) {
							previousIndex_Y_start = rowIndex;
							previousIndex_X_start = colIndex;
						}
						previousIndex_Y_end = rowIndex;
						previousIndex_X_end = colIndex;
					}
				}
			}
		}
	}*/
	return reconstructResult;
}