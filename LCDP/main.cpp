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

int main() {
	// Program version
	programVersion = "RL V1.5 ALL Test 3.4";
	
	std::cout << "Program Version: " << programVersion << std::endl;

	/// Frame Parameters
	// Frames per second (FPS) of the input video
	double FPS;
	// Total number of frame of the input video
	double FRAME_COUNT;
	// Frame size of the input video
	cv::Size FRAME_SIZE;
	// Video file name
	std::string filename;
	// Test dataset name
	std::vector<int> datasetInput;
	std::cout << "Dataset list:" << std::endl;
	for (size_t datasetIndex = 0; datasetIndex < 14; datasetIndex++) {
		// Video file name
		switch (datasetIndex) {
		case 0: filename = "bungalows";
			break;
		case 1: filename = "cubicle";
			break;
		case 2: filename = "canoe";
			break;
		case 3: filename = "fountain02";
			break;
		case 4: filename = "sofa";
			break;
		case 5: filename = "highway";
			break;
		case 6: filename = "office";
			break;
		case 7: filename = "badminton";
			break;
		case 8: filename = "tunnelExit_0_35fps";
			break;
		case 9: filename = "PETS2006";
			break;
		case 10: filename = "fall";
			break;
		case 11:filename = "copyMachine";
			break;
		case 12:filename = "wetSnow";
			break;
		case 13:filename = "snowFall";
			break;
		default:
			std::cout << "Error occurs!";
			break;
		}
		std::cout << datasetIndex << ": " << filename << std::endl;
	}
	// Test database name
	datasetInput = readVectorIntInput("Please input dataset id (-1:Done;999:All)");
	if (datasetInput.at(0) == 999) {
		std::vector<int>().swap(datasetInput);
		for (size_t datasetIndex = 0; datasetIndex < 14; datasetIndex++) {
			datasetInput.push_back(datasetIndex);
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
	// Up RGB differences threshold
	double clsUpRGBThreshold = 30;
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
	float darkRDiffRatioMax = 0.08477f;
	// G-channel different ratio
	//float darkGDiffRatio = 0.1;
	float darkGDiffRatioMin = -0.0002f;
	float darkGDiffRatioMax = 0.02774f;


	/*=====POST PROCESS Parameters=====*/
	bool PostSwitch = true;

	std::string versionFolderName;
	std::string saveFolderName;
	std::string resultFolderName;
	std::string imageSaveFolder;
	const char *s2;
	int datasetNo;
	char s[25];
	std::ofstream myfile;
	//for (size_t datasetIndex = 7; datasetIndex < 8; datasetIndex++) {
	for (std::vector<int>::iterator datasetIndex = datasetInput.begin(); datasetIndex != datasetInput.end();++datasetIndex) {

		//datasetNo = *datasetIndex;
		// Video file name
		switch (*datasetIndex) {
		case 0: filename = "bungalows";
			break;
		case 1: filename = "cubicle";
			break;
		case 2: filename = "canoe";
			break;
		case 3: filename = "fountain02";
			break;
		case 4: filename = "sofa";
			break;
		case 5: filename = "highway";
			break;
		case 6: filename = "office";
			break;
		case 7: filename = "badminton";
			break;
		case 8: filename = "tunnelExit_0_35fps";
			break;
		case 9: filename = "PETS2006";
			break;
		case 10: filename = "fall";
			break;
		case 11:filename = "copyMachine";
			break;
		case 12:filename = "wetSnow";
			break;
		case 13:filename = "snowFall";
			break;
		default:
			std::cout << "Error occurs!";
			break;
		}
		// Read video input from user
		//cv::VideoCapture videoCapture = readVideoInput("Video folder", &filename, &FPS, &FRAME_COUNT, &FRAME_SIZE);
		videoCapture = readVideoInput2(&filename, &FPS, &FRAME_COUNT, &FRAME_SIZE);
		std::cout << "Now load dataset: " << filename << std::endl;

		versionFolderName = filename + "/" + programVersion;
		s2 = versionFolderName.c_str();
		_mkdir(s2);
		
		myfile.open(versionFolderName + "/parameter.csv", std::ios::app);
		myfile << "Program Version,Results Folder, Width, Height, Desc Diff. No,";
		myfile << "Desc NB,Desc Ratio, Offset, RGB Detect,LCD Detect, LCDP Threshold,Up LCDP Threshold, Max LCDP Threshold,";
		myfile << "Initial Persistence Threshold,Matching Threshold,Ratio Method, NB Match,Random Replace Switch, Random Update Switch,";
		myfile << "Feedback Switch,V Inc, V Desc,V Min, T Inc, T Desc,T Min, T Max, Words, Recall, Precision, FMeasure\n";
		myfile.close();

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
		saveFolderName = versionFolderName + "/" + programVersion + "-" + startTime;

		// Declare background subtractor constructor
		BackgroundSubtractorLCDP backgroundSubtractorLCDP(Words_No, PreSwitch,
			descColourDiffRatio, clsRGBDiffSwitch, clsRGBThreshold, clsLCDPDiffSwitch,
			clsLCDPThreshold, clsUpLCDPThreshold, clsLCDPMaxThreshold, clsMatchingThreshold,
			clsNbMatchSwitch, ROI_FRAME, FRAME_SIZE, upRandomReplaceSwitch, upRandomUpdateNbSwitch,
			upFeedbackSwitch, upDynamicRateIncrease, upDynamicRateDecrease, upMinDynamicRate, upUpdateRateIncrease,
			upUpdateRateDecrease, upUpdateRateLowest, upUpdateRateHighest,
			darkMinIntensityRatio, darkMaxIntensityRatio, darkRDiffRatioMin, darkRDiffRatioMax,darkGDiffRatioMin, darkGDiffRatioMax,
			PostSwitch);

		// Initialize background subtractor
		backgroundSubtractorLCDP.Initialize(inputFrame, ROI_FRAME);

		if (saveResultSwitch) {
			s2 = saveFolderName.c_str();
			_mkdir(s2);
			SaveParameter(versionFolderName, saveFolderName);
			backgroundSubtractorLCDP.folderName = saveFolderName;
			backgroundSubtractorLCDP.SaveParameter(versionFolderName, saveFolderName);
			resultFolderName = saveFolderName + "/v";
			s2 = resultFolderName.c_str();
			_mkdir(s2);
			resultFolderName = saveFolderName + "/d";
			s2 = resultFolderName.c_str();
			_mkdir(s2);
			resultFolderName = saveFolderName + "/r";
			s2 = resultFolderName.c_str();
			_mkdir(s2);
			resultFolderName = saveFolderName + "/u";
			s2 = resultFolderName.c_str();
			_mkdir(s2);
			resultFolderName = saveFolderName + "/results";
			s2 = resultFolderName.c_str();
			_mkdir(s2);

		}

		for (int currFrameIndex = 1; currFrameIndex <= FRAME_COUNT; currFrameIndex++) {
			// Process current frame
			backgroundSubtractorLCDP.Process(inputFrame, fgMask);
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
	std::cout << "Program Completed!" << std::endl;
	system("pause");
	return 0;
}


