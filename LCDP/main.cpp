#include <iostream>
#include <direct.h>
#include <opencv2\opencv.hpp>
#include "BackgroundSubtractorLCDP.h"
#include "Functions.h"
#include <time.h>
#include <fstream>
#include <iomanip>
#include <conio.h>
#include <windows.h>
#include <vector>
#include <bitset>

int main() {
	// Program version
	programVersion = "PROPOSED METHOD FINAL";
	/// Frame Parameters
	// Frames per second (FPS) of the input video
	double FPS;
	// Total number of frame of the input video
	double FRAME_COUNT;
	// Frame size of the input video
	cv::Size FRAME_SIZE;
	// Video file name
	std::string filename;
	// List of video file name
	std::vector<std::string> filenames = {
		"badminton","boulevard","bungalows",
		"canoe","copyMachine","cubicle",
		"fall","fountain02","PETS2006",
		"sidewalk","sofa","wetSnow"
	};
	// Test dataset name
	std::vector<int> datasetInput;
	int datasetIndex = 0;
	std::cout << "Dataset list:" << std::endl;
	for (auto & fName: filenames) {
		std::cout << datasetIndex++ << ": " << fName << std::endl;
	}
	// Test database name
	datasetInput = readVectorIntInput("Please input dataset id (-1:Done;999:All)");
	if (datasetInput.at(0) == 999) {
		datasetIndex = 0;
		std::vector<int>().swap(datasetInput);
		for (auto & fName : filenames) {
			datasetInput.push_back(datasetIndex++);
		}
	}
	// Show input frame switch	
	showInputSwitch = readBoolInput("Show input frame(1/0)");
	// Show output frame switch
	showOutputSwitch = readBoolInput("Show output frame(1/0)");
	// Save result switch
	saveResultSwitch = readBoolInput("Save result(1/0)");
	// Evaluate result switch
	evaluateResultSwitch = readBoolInput("Evaluate result(1/0)");

	// Read video input from user
	cv::VideoCapture videoCapture;

	// Input frame
	cv::Mat inputFrame;
	// Foreground Mask
	cv::Mat fgMask;

	// Region of interest frame
	cv::Mat ROI_FRAME;

	if (showInputSwitch) {
		// Display input video windows
		cv::namedWindow("Input Video");
	}
	if (showOutputSwitch) {
		// Display results windows
		cv::namedWindow("Results");
	}
	// Program start time
	std::string startTime;

	/****Define Threshold****/
	/*=====PRE PROCESS Parameters=====*/
	bool PreSwitch = true;

	// Total number of words per pixel
	size_t Words_No = 35;
	/*=====CLASSIFIER Parameters=====*/
	//double ratio[2] = { 0.15,0.2 };
	double descColourDiffRatio = 0.15;
	// RGB detection switch
	bool clsRGBDiffSwitch = true;
	// RGB differences threshold
	double clsRGBThreshold = 10;
	//// Up RGB differences threshold
	//double clsUpRGBThreshold = 30;
	// LCDP detection switch
	bool clsLCDPDiffSwitch = true;
	// LCDP differences threshold (0-1)
	//double LCDPThreshold = readDoubleInput("LCDP Threshold (0-1)");
	double clsLCDPThreshold = 0.25;
	// Up LCDP differences threshold
	double clsUpLCDPThreshold = 0.7;
	// Maximum number of LCDP differences threshold
	double clsLCDPMaxThreshold = 0.7;
	// neighborhood matching switch
	bool clsNbMatchSwitch = true;
	// Matching threshold
	int clsMatchingThreshold = 2;

	/*=====UPDATE Parameters=====*/
	// Random replace model switch
	bool upRandomReplaceSwitch = false;
	// Random update neighborhood model switch
	bool upRandomUpdateNbSwitch = false;
	// Feedback loop switch
	bool upFeedbackSwitch = true;

	// Feedback V(x) Increment
	//float DynamicRateIncrease[3] = { 0.5f,0.8f,1.0f };
	//float upDynamicRateIncrease = 0.01f;
	//float upDynamicRateIncrease = 0.005f;
	float upDynamicRateIncrease = 1.0f;

	// Feedback V(x) Decrement
	//float DynamicRateDecrease[3] = { 0.4f,0.3f,0.1f };
	//float upDynamicRateDecrease = 0.005f;
	//float upDynamicRateDecrease = 0.05f;
	float upDynamicRateDecrease = 0.1f;
	// Minimum of Feedback V(x) value
	float upMinDynamicRate = 0.0f;
	// Feedback T(x) Increment
	float upUpdateRateIncrease = 0.5f;
	// Feedback T(x) Decrement
	float upUpdateRateDecrease = 0.5f;
	// Feedback T(x) Lowest
	float upUpdateRateLowest = 2.0f;
	// Feedback T(x) Highest
	float upUpdateRateHighest = 255.0f;

	/*=====RGB Dark Pixel Parameter=====*/
	// Minimum Intensity Ratio
	float darkMinIntensityRatio = 0.25f;
	// Maximum Intensity Ratio
	float darkMaxIntensityRatio = 0.8f;
	// R-channel different ratio
	//float darkRDiffRatio = 0.15;
	float darkRDiffRatioMin = 0.04097f;
	//float darkRDiffRatioMin = 0.0f;
	float darkRDiffRatioMax = 0.08477f;
	// G-channel different ratio
	//float darkGDiffRatio = 0.1;
	float darkGDiffRatioMin = -0.0002f;
	//float darkGDiffRatioMin = 0.0f;
	float darkGDiffRatioMax = 0.02774f;


	/*=====POST PROCESS Parameters=====*/
	bool PostSwitch = true;

	std::string versionFolderName;
	std::string saveFolderName;
	std::string resultFolderName;
	std::string imageSaveFolder;
	if (!clsLCDPDiffSwitch) {
		programVersion = programVersion + " NO LCDP";
	}
	std::cout << "Program Version: " << programVersion << std::endl;

	bool success;
	const char *s2;
	int datasetNo;
	char s[25];
	std::ofstream myfile;
	//for (size_t datasetIndex = 7; datasetIndex < 8; datasetIndex++) {
	//for (std::vector<int>::iterator datasetIndex = datasetInput.begin(); datasetIndex != datasetInput.end(); ++datasetIndex) {
	for (auto & datasetIndex : datasetInput) {
		success = false;		

		// Video file name
		filename = filenames.at(datasetIndex);		
		// Read video input from user
		//cv::VideoCapture videoCapture = readVideoInput("Video folder", &filename, &FPS, &FRAME_COUNT, &FRAME_SIZE);
		videoCapture = readVideoInput2(&filename, &FPS, &FRAME_COUNT, &FRAME_SIZE, &success);
		if (success) {
			std::cout << "Now load dataset: " << filename << std::endl;
			versionFolderName = filename + "/" + programVersion;
			s2 = versionFolderName.c_str();
			_mkdir(s2);
			ROI_FRAME.create(FRAME_SIZE, CV_8UC1);
			ROI_FRAME = cv::Scalar_<uchar>(255);

			// Read first frame from video
			videoCapture.set(cv::CAP_PROP_POS_FRAMES, 0);
			videoCapture >> inputFrame;

			// Current date/time based on current system
			tempStartTime = time(0);
			// Program start time
			startTime = currentDateTimeStamp(&tempStartTime);

			// Current process result folder name
			saveFolderName = versionFolderName + "/" + filename + "-" + programVersion + "-" + startTime;

			// Declare background subtractor constructor
			BackgroundSubtractorLCDP backgroundSubtractorLCDP(Words_No, PreSwitch,
				descColourDiffRatio, clsRGBDiffSwitch, clsRGBThreshold, clsLCDPDiffSwitch,
				clsLCDPThreshold, clsUpLCDPThreshold, clsLCDPMaxThreshold, clsMatchingThreshold,
				clsNbMatchSwitch, ROI_FRAME, FRAME_SIZE, FRAME_COUNT, upRandomReplaceSwitch, upRandomUpdateNbSwitch,
				upFeedbackSwitch, upDynamicRateIncrease, upDynamicRateDecrease, upMinDynamicRate, upUpdateRateIncrease,
				upUpdateRateDecrease, upUpdateRateLowest, upUpdateRateHighest,
				darkMinIntensityRatio, darkMaxIntensityRatio, darkRDiffRatioMin, darkRDiffRatioMax, darkGDiffRatioMin, darkGDiffRatioMax,
				PostSwitch);

			// Initialize background subtractor
			backgroundSubtractorLCDP.Initialize(inputFrame, ROI_FRAME);

			if (saveResultSwitch) {
				s2 = saveFolderName.c_str();
				_mkdir(s2);
				SaveParameter(versionFolderName, saveFolderName);
				backgroundSubtractorLCDP.folderName = saveFolderName;
				backgroundSubtractorLCDP.SaveParameter(versionFolderName, saveFolderName);
				resultFolderName = saveFolderName + "/results";
				s2 = resultFolderName.c_str();
				_mkdir(s2);
			}
			// Program start time
			time_t firstStartTime = time(0);
			time_t firstEndTime = time(0);
			firstTotalDiffSeconds = 0.0;
			for (int currFrameIndex = 1; currFrameIndex <= FRAME_COUNT; currFrameIndex++) {
				firstStartTime = time(0);
				// Process current frame
				backgroundSubtractorLCDP.Process(inputFrame, fgMask);
				firstEndTime = time(0);
				firstTotalDiffSeconds += difftime(firstEndTime, firstStartTime);
				if (showInputSwitch) {
					cv::imshow("Input Video", inputFrame);
				}
				if (showOutputSwitch) {
					cv::imshow("Results", fgMask);
				}
				if (saveResultSwitch) {
					sprintf(s, "/bin%06d.png", (currFrameIndex));
					imageSaveFolder = resultFolderName + s;
					cv::imwrite(imageSaveFolder, fgMask);
				}
				// If 'esc' key is pressed, break loop
				if (cv::waitKey(1) == 27)
				{
					std::cout << "Program ended by users." << std::endl;
					break;
				}
				bool inputCheck = videoCapture.read(inputFrame);
				if (!inputCheck && (currFrameIndex < FRAME_COUNT)) {
					std::cout << "Video having problem. Cannot read the frame from video file." << std::endl;
					return -1;
				}
			}

			tempFinishTime = time(0);
			GenerateProcessTime(FRAME_COUNT, saveFolderName);

			std::cout << "Background subtraction completed" << std::endl;

			if (evaluateResultSwitch) {
				if (saveResultSwitch) {
					std::cout << "Now starting evaluate the processed result..." << std::endl;
					EvaluateResult(filename, saveFolderName, versionFolderName);
				}
				else {
					std::cout << "No saved results for evaluation" << std::endl;
				}
			}
		}
		else {
			std::cout << "Cannot load dataset: " << filename << ". Skip it to next dataset!" << std::endl;
		}
	}
	std::cout << "Program Completed!" << std::endl;
	system("pause");
	return 0;
}


