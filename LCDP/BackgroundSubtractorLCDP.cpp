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

BackgroundSubtractorLCDP::BackgroundSubtractorLCDP(int FRAME_HEIGHT, int FRAME_WIDTH, size_t maxWordsNo) :
	wordsNo(maxWordsNo) {
	CV_Assert(maxWordsNo > 0);
}
BackgroundSubtractorLCDP::~BackgroundSubtractorLCDP() {

}


void BackgroundSubtractorLCDP::Initialize(cv::Mat img_input, cv::Mat roi_input)
{
	int totalPixel = img_input.cols*img_input.rows;

	bgWord = new Descriptor[totalPixel*wordsNo];
	memset(bgWord, 0, sizeof(Descriptor)*totalPixel*wordsNo);
	bgWordListIter = bgWord;
	lastWord = new Descriptor[totalPixel];
	memset(lastWord, 0, sizeof(Descriptor)*totalPixel);
	lastWordListIter = lastWord;

	modelInitCheck = true;

	// Frame Index
	frameIndex = 1;

	// PRE-PROCESS Parameters
	// Size of gaussian filter
	methodParam.preProcess.gaussianSize = 5;

	// GENERATE DESCRIPTOR Parameters
	// LCDP Parameters
	// Size of neighbourhood 3(3x3)/5(5x5)
	methodParam.descriptor.LCDP.nbSize = 5;
	// Total number of LCD's neighbour 8(3x3)/16(5x5)
	methodParam.descriptor.LCDP.nbNo = 16;
	// Size of the extention on the frame
	methodParam.descriptor.LCDP.extendSize = cv::Size2d(floor(methodParam.descriptor.LCDP.nbSize / 2), floor(methodParam.descriptor.LCDP.nbSize / 2));
	// LCD colour differences ratio
	methodParam.descriptor.LCDP.T_colorDiffRatio = 0.05;
	// Total number of LCD differences per pixel
	methodParam.descriptor.LCDP.diffNo = 18;
	// Persistence's offset value;
	methodParam.descriptor.offsetValue = 1000;

	// CLASSIFIER Parameters
	// RGB Classifier Parameters
	// RGB differences threshold
	methodParam.classifier.rgb.T_RGB = 10;
	// RGB detection switch
	methodParam.classifier.rgb.RGBDiffSwitch = false;
	// RGB bright pixel switch
	methodParam.classifier.rgb.RGBBrightPxSwitch = true;
	// LCDP Classifier Parameters
	// LCDP differences threshold
	methodParam.classifier.LCDP.T_LCDP = 36;
	// Maximum number of LCDP differences threshold
	methodParam.classifier.LCDP.Max_T_LCDP = 72;
	// LCDP detection switch
	methodParam.classifier.LCDP.LCDPDiffSwitch = true;
	// LCDP detection AND (true) OR (false) switch
	methodParam.classifier.LCDP.AndOrSwitch = true;
	// Neighbourhood matching switch
	methodParam.classifier.nbMatchSwitch = true;
	// Total number of neighbour 8(3x3)/16(5x5)
	methodParam.classifier.nbNo = 16;

	// FRAME parameters
	// Size of input frame
	methodParam.frame.initFrameSize = cv::Size(img_input.cols, img_input.rows);
	// Total number of pixel of input frame
	methodParam.frame.initTotalPixel = methodParam.frame.initFrameSize.area();
	// Size of region of interest
	methodParam.frame.roiSize = cv::Size(roi_input.cols, roi_input.rows);
	// Total number of pixel of region of interest
	methodParam.frame.roiTotalPixel = methodParam.frame.roiSize.area();

	// POST-PROCESS Parameters
		// Size of median filter
	methodParam.postProcess.medianFilterSize = 9;
	if ((methodParam.frame.roiTotalPixel >= (methodParam.frame.initTotalPixel / 2)) && (methodParam.frame.initTotalPixel >= DEFAULT_FRAME_SIZE.area())) {
		// Specifies whether Tmin / Tmax scaling is enabled or not
		methodParam.update.learningRateScalingSwitch = true;
		// Specifies the px update spread range
		methodParam.update.use3x3Spread = !(methodParam.frame.initTotalPixel > (DEFAULT_FRAME_SIZE.area() * 2));
		size_t tempMedianFilterSize = std::min(double(14), floor((methodParam.frame.roiTotalPixel / DEFAULT_FRAME_SIZE.area()) + 0.5) + methodParam.postProcess.medianFilterSize);
		// Current kernel size for median blur post-proc filtering
		methodParam.postProcess.medianFilterSize = tempMedianFilterSize;
		// Current learning rate caps
		methodParam.update.learningRateLowerCap = FEEDBACK_T_LOWER;
		methodParam.update.learningRateUpperCap = FEEDBACK_T_UPPER;
	}
	else {
		// Specifies whether Tmin / Tmax scaling is enabled or not
		methodParam.update.learningRateScalingSwitch = false;
		// Specifies the px update spread range
		methodParam.update.use3x3Spread = true;
		// Current kernel size for median blur post-proc filtering
		methodParam.postProcess.medianFilterSize = methodParam.postProcess.medianFilterSize;
		// Current learning rate caps
		methodParam.update.learningRateLowerCap = FEEDBACK_T_LOWER * 2;
		methodParam.update.learningRateUpperCap = FEEDBACK_T_UPPER * 2;
	}

	// UPDATE parameters
	// Initial blinking accumulate level
	methodParam.update.blinkAccLevel.create(methodParam.frame.initFrameSize, CV_32FC1);
	methodParam.update.blinkAccLevel = cv::Scalar(1.0f);
	// Random replace model switch
	methodParam.update.randomReplaceSwitch = true;
	// Random update neighbourhood model switch
	methodParam.update.randomUpdateNbSwitch = true;
	// Total number of neighbour undergo neighbourhood updates 8(3X3)/16(5X5)
	methodParam.update.nbNo = 8;
	// Feedback loop switch
	methodParam.update.feedbackSwitch = true;

	methodParam.update.samplesForMovingAvgs = 100;


	// Per-pixel distance thresholds ('R(x)', but used as a relative value to determine both 
	// intensity and descriptor variation thresholds)
	result.model.distThreshold.create(methodParam.frame.initFrameSize, CV_32FC1);
	result.model.distThreshold = cv::Scalar(1.0f);
	// A lookup map used to keep track of unstable regions (based on segm. noise & local dist. thresholds)
	result.model.unstableRegionMask.create(methodParam.frame.initFrameSize, CV_8UC1);
	result.model.unstableRegionMask = cv::Scalar_<uchar>(0);
	// Per-pixel mean raw segmentation results (used to detect unstable segmentation regions)
	result.model.meanRawSegmRes_LT.create(methodParam.frame.initFrameSize, CV_32FC1);
	result.model.meanRawSegmRes_LT = cv::Scalar(0.0f);
	result.model.meanRawSegmRes_ST.create(methodParam.frame.initFrameSize, CV_32FC1);
	result.model.meanRawSegmRes_ST = cv::Scalar(0.0f);
	// Per-pixel mean raw segmentation results (used to detect unstable segmentation regions)
	result.model.meanFinalSegmRes_LT.create(methodParam.frame.initFrameSize, CV_32FC1);
	result.model.meanFinalSegmRes_LT = cv::Scalar(0.0f);
	result.model.meanFinalSegmRes_ST.create(methodParam.frame.initFrameSize, CV_32FC1);
	result.model.meanFinalSegmRes_ST = cv::Scalar(0.0f);
	// Per-pixel update rates('T(x)', which contains pixel - level 'sigmas')
	result.model.updateRate.create(methodParam.frame.initFrameSize, CV_32FC1);
	result.model.updateRate = cv::Scalar(methodParam.update.learningRateLowerCap);
	// Per-pixel mean minimal distances from the model ('D_min(x)', used to control 
	// variation magnitude and direction of 'T(x)' and 'R(x)')
	result.model.meanMinDist_LT.create(methodParam.frame.initFrameSize, CV_32FC1);
	result.model.meanMinDist_LT = cv::Scalar(0.0f);
	result.model.meanMinDist_ST.create(methodParam.frame.initFrameSize, CV_32FC1);
	result.model.meanMinDist_ST = cv::Scalar(0.0f);
	// Per-pixel distance variation modulators ('v(x)', relative value used to modulate 'R(x)' 
	// and 'T(x)' variations)
	result.model.variationModulator.create(methodParam.frame.initFrameSize, CV_32FC1);
	result.model.variationModulator = cv::Scalar(10);
	// Per-pixel blink detection map ('Z(x)')
	result.model.blinksFrame.create(methodParam.frame.initFrameSize, CV_8UC1);
	result.model.blinksFrame = cv::Scalar_<uchar>(0);

	// Minimum RGB distance
	result.minRGBDistance.create(methodParam.frame.initFrameSize, CV_32FC1);
	result.minRGBDistance = cv::Scalar(1);
	// Minimum LCD distance
	result.minLCDPDistance.create(methodParam.frame.initFrameSize, CV_32FC1);
	result.minLCDPDistance = cv::Scalar(1);
	// Current foreground mask
	result.currFGMask.create(methodParam.frame.initFrameSize, CV_8UC1);
	result.currFGMask = cv::Scalar_<uchar>(0);
	// Previous foreground mask
	result.lastFGMask.create(methodParam.frame.initFrameSize, CV_8UC1);
	result.lastFGMask = cv::Scalar_<uchar>(0);
	// The foreground mask generated by the method at [t-1] (without post-proc, used for blinking px detection)
	result.lastRawFGMask.create(methodParam.frame.initFrameSize, CV_8UC1);
	result.lastRawFGMask = cv::Scalar_<uchar>(0);
	// The blink mask generated by the method at [t-1] (without post-proc, used for blinking px detection)
	result.lastRawFGBlinkMask.create(methodParam.frame.initFrameSize, CV_8UC1);
	result.lastRawFGBlinkMask = cv::Scalar_<uchar>(0);
	// t-1 foreground mask
	result.t_1FGMask.create(methodParam.frame.initFrameSize, CV_8UC1);
	result.t_1FGMask = cv::Scalar_<uchar>(0);
	// t-2 foreground mask
	result.t_2FGMask.create(methodParam.frame.initFrameSize, CV_8UC1);
	result.t_2FGMask = cv::Scalar_<uchar>(0);
	result.lastFGMaskDilated.create(methodParam.frame.initFrameSize, CV_8UC1);
	result.lastFGMaskDilated = cv::Scalar_<uchar>(0);
	result.lastFGMaskDilatedInverted.create(methodParam.frame.initFrameSize, CV_8UC1);
	result.lastFGMaskDilatedInverted = cv::Scalar_<uchar>(0);
	result.FGMaskFloodedHoles.create(methodParam.frame.initFrameSize, CV_8UC1);
	result.FGMaskFloodedHoles = cv::Scalar_<uchar>(0);
	result.FGMaskPreFlood.create(methodParam.frame.initFrameSize, CV_8UC1);
	result.FGMaskPreFlood = cv::Scalar_<uchar>(0);
	result.currRawFGBlinkMask.create(methodParam.frame.initFrameSize, CV_8UC1);
	result.currRawFGBlinkMask = cv::Scalar_<uchar>(0);
	pxInfoLUT = new PxInfoBase[methodParam.frame.initTotalPixel];
	size_t nPxIter = 0;
	size_t nModelIter = 0;
	for (size_t rowIndex = 0;rowIndex < methodParam.frame.initFrameSize.height;rowIndex++) {
		for (size_t colIndex = 0;colIndex < methodParam.frame.initFrameSize.width;colIndex++) {
			++nPxIter;
			pxInfoLUT[nModelIter] = nPxIter;
			pxInfoLUT[nPxIter].imgCoord_Y = (int)nPxIter / methodParam.frame.initFrameSize.width;
			pxInfoLUT[nPxIter].imgCoord_X = (int)nPxIter%methodParam.frame.initFrameSize.width;
			pxInfoLUT[nPxIter].nModelIdx = nModelIter;
			const size_t nPxRGBIter = nPxIter * 3;
			const size_t nDescRGBIter = nPxRGBIter * 2;
			Descriptor tempWord = *lastWordListIter++;
			DescriptorGenerator(img_input.at<cv::Vec3d>(rowIndex, colIndex), tempWord);
			++nModelIter;
		}
	}

	RefreshModel(1, 0);
}

void BackgroundSubtractorLCDP::RefreshModel(float refreshFraction, bool forceFGUpdateSwitch)
{
	const size_t noSampleBeRefresh = refreshFraction<1.0f ? (size_t)(refreshFraction*wordsNo) : wordsNo;
	const size_t refreshStartPos = refreshFraction<1.0f ? rand() % wordsNo : 0;
	for (size_t nModelIter = 0; nModelIter < methodParam.frame.initTotalPixel; ++nModelIter) {
		if (!forceFGUpdateSwitch||!result.lastFGMask.data[nModelIter]) {
			for (size_t nCurrModelIdx = refreshStartPos; nCurrModelIdx < refreshStartPos + noSampleBeRefresh; ++nCurrModelIdx) {
				int nSampleImgCoord_Y, nSampleImgCoord_X;
				getRandSamplePosition(nSampleImgCoord_X, nSampleImgCoord_Y, pxInfoLUT[].imgCoord_X, pxInfoLUT[nPxIter].imgCoord_Y, methodParam.descriptor.LCDP.nbSize/ 2, methodParam.frame.initFrameSize);
				const size_t nSamplePxIdx = methodParam.frame.initFrameSize.width*nSampleImgCoord_Y + nSampleImgCoord_X;
			}
		}
	}
}

void BackgroundSubtractorLCDP::DescriptorGenerator(cv::Vec3d pixelInput, Descriptor &tempWord)
{
	tempWord.frameCount = 1;
	tempWord.rgb = pixelInput;
	tempWord.p = frameIndex;
	tempWord.q = frameIndex;
	tempWord.LCDP = 

}



void BackgroundSubtractorLCDP::Process(const cv::Mat & img_input, cv::Mat & img_output)
{


	// POST PROCESSING
	cv::bitwise_xor(result.currFGMask, result.lastRawFGMask, result.currRawFGBlinkMask);
	cv::bitwise_or(result.currRawFGBlinkMask, result.lastRawFGBlinkMask, result.model.blinksFrame);
	result.currRawFGBlinkMask.copyTo(result.lastRawFGBlinkMask);
	result.currFGMask.copyTo(result.lastRawFGMask);
	cv::morphologyEx(result.currFGMask, result.currFGMask, cv::MORPH_OPEN, cv::Mat());
	cv::Mat borderLineReconstructResult = BorderLineReconst();
	cv::Mat compensationResult = CompensationMotionHist();
	cv::bitwise_or(result.currFGMask, borderLineReconstructResult, result.currFGMask);
	cv::morphologyEx(result.currFGMask, result., cv::MORPH_CLOSE, cv::Mat());
	result.FGMaskPreFlood.copyTo(result.FGMaskFloodedHoles);
	cv::floodFill(result.FGMaskFloodedHoles, cv::Point(0, 0), UCHAR_MAX);
	cv::bitwise_not(result.FGMaskFloodedHoles, result.FGMaskFloodedHoles);
	cv::erode(result.FGMaskPreFlood, result.FGMaskPreFlood, cv::Mat(), cv::Point(-1, -1), 3);
	cv::bitwise_or(result.currFGMask, result.FGMaskFloodedHoles, result.currFGMask);
	cv::bitwise_or(result.currFGMask, result.FGMaskPreFlood, result.currFGMask);
	cv::bitwise_and(result.currFGMask, compensationResult, result.currFGMask);

	borderLineReconstructResult = BorderLineReconst();
	cv::bitwise_or(result.currFGMask, borderLineReconstructResult, result.currFGMask);
	cv::floodFill(result.currFGMask, cv::Point(0, 0), UCHAR_MAX);
	cv::bitwise_not(result.currFGMask, result.currFGMask);

	cv::medianBlur(result.currFGMask, result.lastFGMask, methodParam.postProcess.medianFilterSize);
	cv::dilate(result.lastFGMask, result.lastFGMaskDilated, cv::Mat(), cv::Point(-1, -1), 3);
	cv::bitwise_and(result.model.blinksFrame, result.lastFGMaskDilatedInverted, result.model.blinksFrame);
	cv::bitwise_not(result.lastFGMaskDilated, result.lastFGMaskDilatedInverted);
	cv::bitwise_and(result.model.blinksFrame, result.lastFGMaskDilatedInverted, result.model.blinksFrame);
	result.lastFGMask.copyTo(result.currFGMask);

	result.t_1FGMask.copyTo(result.t_2FGMask);
	result.lastFGMask.copyTo(result.t_1FGMask);
	// Frame Index
	frameIndex++;
}

// Compensation with Motion Hist
cv::Mat BackgroundSubtractorLCDP::CompensationMotionHist() {
	cv::Mat compensationResult;
	compensationResult.create(result.currFGMask.size(), CV_8UC1);
	compensationResult = cv::Scalar_<uchar>(0);
	cv::Mat totalFGMask;
	totalFGMask.create(result.currFGMask.size(), CV_32FC1);
	totalFGMask = cv::Scalar(0.0f);

	return compensationResult;
}

// Border line reconstruct
cv::Mat BackgroundSubtractorLCDP::BorderLineReconst()
{
	cv::Mat reconstructResult;
	reconstructResult.create(methodParam.frame.initFrameSize, CV_8UC1);
	reconstructResult = cv::Scalar_<uchar>(0);
	return reconstructResult;
}