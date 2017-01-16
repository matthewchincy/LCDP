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
	methodParam.preProcess.gaussianSize = cv::Size(5,5);

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
			pxInfoLUT[nPxIter].imgCoord_Y = (int)nPxIter / methodParam.frame.initFrameSize.width;
			pxInfoLUT[nPxIter].imgCoord_X = (int)nPxIter%methodParam.frame.initFrameSize.width;
			pxInfoLUT[nPxIter].nModelIdx = nModelIter;
			const size_t nPxRGBIter = nPxIter * 3;
			const size_t nDescRGBIter = nPxRGBIter * 2;
			Descriptor tempWord = *lastWordListIter++;
			DescriptorGenerator(img_input,cv::Point2d(colIndex, rowIndex), tempWord);
			++nModelIter;
			nPxIter++;			
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


void BackgroundSubtractorLCDP::Process(const cv::Mat & img_input, cv::Mat & img_output)
{
	// PRE PROCESSING
	cv::GaussianBlur(img_input, img_input, methodParam.preProcess.gaussianSize, 0, 0);




	// POST PROCESSING
	cv::bitwise_xor(result.currFGMask, result.lastRawFGMask, result.currRawFGBlinkMask);
	cv::bitwise_or(result.currRawFGBlinkMask, result.lastRawFGBlinkMask, result.model.blinksFrame);
	result.currRawFGBlinkMask.copyTo(result.lastRawFGBlinkMask);
	result.currFGMask.copyTo(result.lastRawFGMask);
	cv::morphologyEx(result.currFGMask, result.currFGMask, cv::MORPH_OPEN, cv::Mat());
	cv::Mat borderLineReconstructResult = BorderLineReconst();
	cv::Mat compensationResult = CompensationMotionHist();
	cv::bitwise_or(result.currFGMask, borderLineReconstructResult, result.currFGMask);
	//cv::morphologyEx(result.currFGMask, result., cv::MORPH_CLOSE, cv::Mat());
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

std::vector<std::vector<int>> BackgroundSubtractorLCDP::LCDGenerator(cv::Mat inputFrame, cv::Point2d coor)
{
	size_t nbNo = methodParam.descriptor.LCDP.nbNo;
	int maxHeight = methodParam.frame.initFrameSize.height - 1;
	int maxWidth = methodParam.frame.initFrameSize.width - 1;
	std::vector<std::vector<int>> result(nbNo);
	cv::Point2d offset[16];
	offset[0] = cv::Point2d(-1, -1);
	offset[1] = cv::Point2d(0, -1);
	offset[2] = cv::Point2d(1, -1);
	offset[3] = cv::Point2d(-1, 0);
	offset[4] = cv::Point2d(1, 0);
	offset[5] = cv::Point2d(-1, 1);
	offset[6] = cv::Point2d(0, 1);
	offset[7] = cv::Point2d(1, 1);
	offset[8] = cv::Point2d(-2, -2);
	offset[9] = cv::Point2d(0, -2);
	offset[10] = cv::Point2d(2, -2);
	offset[11] = cv::Point2d(-2, 0);
	offset[12] = cv::Point2d(2, 0);
	offset[13] = cv::Point2d(-2, 2);
	offset[14] = cv::Point2d(0, 2);
	offset[15] = cv::Point2d(2, 2);
	cv::Vec3b currPixel = inputFrame.at<cv::Vec3b>(coor.y, coor.x);
	for (int pixelIndex = 0;pixelIndex < nbNo;pixelIndex++) {
		cv::Point2d nbPixelCoor(std::min(maxWidth,std::max(0,int(coor.x + offset[pixelIndex].x))), std::min(maxHeight, std::max(0, int(coor.y + offset[pixelIndex].y))));
		cv::Vec3b nbPixel = inputFrame.at<cv::Vec3b>(nbPixelCoor.y, nbPixelCoor.x);
		std::vector<int> tempRes;
		// R-G
		tempRes.push_back(nbPixel[2] - currPixel[1]);
		// R-B
		tempRes.push_back(nbPixel[2] - currPixel[0]);
		// G-R
		tempRes.push_back(nbPixel[1] - currPixel[2]);
		// G-B
		tempRes.push_back(nbPixel[1] - currPixel[0]);
		// B-R
		tempRes.push_back(nbPixel[0] - currPixel[2]);
		// B-G
		tempRes.push_back(nbPixel[0] - currPixel[1]);
		// R-R
		tempRes.push_back(nbPixel[2] - currPixel[2]);
		// G-G
		tempRes.push_back(nbPixel[1] - currPixel[1]);
		// B-B
		tempRes.push_back(nbPixel[0] - currPixel[0]);
		result.at(pixelIndex) = tempRes;
	}

	return result;
}



void BackgroundSubtractorLCDP::DescriptorGenerator(cv::Mat inputFrame, cv::Point2d coor, Descriptor & tempWord)
{
	tempWord.frameCount = 1;
	tempWord.rgb = inputFrame.at<cv::Vec3b>(coor.y,coor.x);
	tempWord.p = frameIndex;
	tempWord.q = frameIndex;
	tempWord.LCDP = LCDGenerator(inputFrame,coor);
}

// Border line reconstruct
cv::Mat BackgroundSubtractorLCDP::BorderLineReconst()
{
	cv::Mat reconstructResult;
	reconstructResult.create(methodParam.frame.initFrameSize, CV_8UC1);
	reconstructResult = cv::Scalar_<uchar>(0);
	size_t height = methodParam.frame.initFrameSize.height-1;
	size_t width = methodParam.frame.initFrameSize.width-1;
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
	}
	return reconstructResult;
}