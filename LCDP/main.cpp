#include <iostream>
#include <direct.h>
#include <opencv2\opencv.hpp>
#include "BackgroundSubtractorLCDP.h"
#include <time.h>
#include <fstream>
#include <iomanip>
#include <conio.h>
#include <windows.h>

/****Read input methods****/
// Read integer value input
int readIntInput(std::string question);
// Read double value input
double readDoubleInput(std::string question);
// Read boolean value input
bool readBoolInput(std::string question);
// Read video input
cv::VideoCapture readVideoInput(std::string question, std::string *filename, 
	double *FPS, double *FRAME_COUNT, cv::Size *FRAME_SIZE);
cv::VideoCapture readVideoInput2(std::string question, std::string *filename, double *FPS,
	double *FRAME_COUNT, cv::Size *FRAME_SIZE);
// Get current date/time, format is DDMMYYHHmmss
const std::string currentDateTimeStamp(time_t * now);
// Get current date/time, format is DD-MM-YY HH:mm:ss
const std::string currentDateTime(time_t * now);
// Save current's process parameter
void SaveParameter(std::string folderName);
// Evaluate results
void EvaluateResult(std::string filename, std::string folderName, std::string currFolderName);
// Calculate processing time
void GenerateProcessTime(double FRAME_COUNT, std::string currFolderName);

/****Global variable declaration****/
// Program version
const std::string programVersion = "4-0";
// Show input frame switch
bool showInputSwitch;
// Show output frame switch
bool showOutputSwitch;
// Save result switch
bool saveResultSwitch;
// Evaluate result switch
bool evaluateResultSwitch;
// Debug switch
bool debugSwitch;
// Program start time
time_t tempStartTime;
// Program finish time
time_t tempFinishTime;
int main() {
	std::cout <<"Program Version: "<< programVersion << std::endl;
	// Frames per second (FPS) of the input video
	double FPS;
	// Total number of frame of the input video
	double FRAME_COUNT;
	// Frame size of the input video
	cv::Size FRAME_SIZE;
	// Input LCDP threshold (0-1)
	double inputLCDPThreshold = readDoubleInput("LCDP Threshold (0-1)");
	for (size_t datasetIndex = 0;datasetIndex < 8;datasetIndex++) {
		// Video file name
		std::string filename;
		switch (datasetIndex) {
			case 0: filename = "bungalows";
				break;
			case 1: filename = "canoe";
				break;
			case 2:filename = "fall";
				break;
			case 3:filename = "cubicle";
				break;
			case 4: filename = "traffic";
				break;
			case 5:filename = "sofa";
				break;
			case 6: filename = "boats";
				break;
			case 7: filename = "fountain01";
				break;
			default:
				std::cout << "Error occurs!";
				break;
		}
		std::cout << "Now load dataset: " << filename << std::endl;
		//// Read video input from user
		//cv::VideoCapture videoCapture = readVideoInput("Video folder", &filename, &FPS, &FRAME_COUNT, &FRAME_SIZE);
		//// Show input frame switch
		//showInputSwitch = readBoolInput("Show input frame(1/0)");
		//// Show output frame switch
		//showOutputSwitch = readBoolInput("Show output frame(1/0)");
		//// Save result switch
		//saveResultSwitch = readBoolInput("Save result(1/0)");
		//// Evaluate result switch
		//evaluateResultSwitch = readBoolInput("Evaluate result(1/0)");
		//// Debug switch
		//debugSwitch = readBoolInput("Debug Mode(1/0)");

		// Read video input from user
		cv::VideoCapture videoCapture = readVideoInput2("Video folder", &filename, &FPS, &FRAME_COUNT, &FRAME_SIZE);
		// Show input frame switch
		showInputSwitch = false;
		// Show output frame switch
		showOutputSwitch = false;
		// Save result switch
		saveResultSwitch = true;
		// Evaluate result switch
		evaluateResultSwitch = true;
		// Debug switch
		debugSwitch = false;
		

		// Debug starting frame index
		int debugFrameIndex = 0;
		// Debug x-point
		int debugX = 0;
		// Debug y-point
		int debugY = 0;
		if (debugSwitch) {
			// Read x-point for debug
			debugX = readIntInput("X-Point");
			// Read y-point for debug
			debugY = readIntInput("Y-Point");
			// Read starting frame index for debug
			debugFrameIndex = readIntInput("Starting Frame Index for debug (Start:0)");
		}

		//// Input LCDP threshold (0-1)
		//double inputLCDPThreshold = readDoubleInput("LCDP Threshold (0-1)");
		// Input frame
		cv::Mat inputFrame;
		// Foreground Mask
		cv::Mat fgMask;

		// Region of interest frame
		cv::Mat ROI_FRAME;
		ROI_FRAME.create(FRAME_SIZE, CV_8UC1);
		ROI_FRAME = cv::Scalar_<uchar>(255);
		if (showInputSwitch) {
			// Display input video windows
			cv::namedWindow("Input Video");
		}
		if (showOutputSwitch) {
			// Display results windows
			cv::namedWindow("Results");
		}

		/****Define Threshold****/
		// Total number of words per pixel
		size_t Words_No = 35;
		/*=====CLASSIFIER Parameters=====*/
		// RGB detection switch
		bool RGBDiffSwitch = false;
		// RGB differences threshold
		double RGBThreshold = 40;
		// RGB bright pixel switch
		bool RGBBrightPxSwitch = false;
		// LCDP detection switch
		bool LCDPDiffSwitch = true;
		// LCDP differences threshold
		double LCDPThreshold = inputLCDPThreshold;
		// Maximum number of LCDP differences threshold
		double LCDPMaxThreshold = 0.7;
		// LCDP detection AND (true) OR (false) switch
		bool AndOrSwitch = false;
		// Neighbourhood matching switch
		bool NbMatchSwitch = true;
		/*=====UPDATE Parameters=====*/
		// Random replace model switch
		bool RandomReplaceSwitch = false;
		// Random update neighbourhood model switch
		bool RandomUpdateNbSwitch = false;
		// Feedback loop switch
		bool FeedbackSwitch = true;

		// Declare background subtractor construtor
		BackgroundSubtractorLCDP backgroundSubtractorLCDP(FRAME_SIZE, ROI_FRAME, Words_No, RGBDiffSwitch,
			RGBThreshold, RGBBrightPxSwitch, LCDPDiffSwitch, LCDPThreshold, LCDPMaxThreshold,
			AndOrSwitch, NbMatchSwitch, RandomReplaceSwitch, RandomUpdateNbSwitch, FeedbackSwitch);

		// Read first frame from video
		videoCapture.set(cv::CAP_PROP_POS_FRAMES, 0);
		videoCapture >> inputFrame;

		// Initialize background subtractor
		backgroundSubtractorLCDP.Initialize(inputFrame, ROI_FRAME);

		// Current date/time based on current system
		tempStartTime = time(0);
		// Program start time
		std::string startTime = currentDateTimeStamp(&tempStartTime);
		// Current process result folder name
		std::string folderName = filename + "/LCD Sample Consensus-" + programVersion + "-" + startTime;
		const std::string currFolderName = folderName;

		const char *s1;
		if (saveResultSwitch) {
			s1 = folderName.c_str();
			_mkdir(s1);
			SaveParameter(folderName);
			backgroundSubtractorLCDP.SaveParameter(folderName);
			folderName += "/results";
			s1 = folderName.c_str();
			_mkdir(s1);
		}
		char s[25];
		for (int currFrameIndex = 1;currFrameIndex <= FRAME_COUNT;currFrameIndex++) {
			if (debugFrameIndex <= currFrameIndex) {
				debugFrameIndex = FRAME_COUNT + 1;
				if (debugSwitch) {
					backgroundSubtractorLCDP.debugSwitch = true;
					backgroundSubtractorLCDP.DebugPxLocation(debugX, debugY);
				}
			}

			// Process current frame
			backgroundSubtractorLCDP.Process(inputFrame, fgMask);
			if (showInputSwitch) {
				cv::imshow("Input Video", inputFrame);
			}
			if (showOutputSwitch) {
				cv::imshow("Results", fgMask);
			}
			if (saveResultSwitch) {
				std::string saveFolder;
				sprintf(s, "/bin%06d.png", (currFrameIndex));
				saveFolder = folderName + s;
				cv::imwrite(saveFolder, fgMask);
			}
			// If 'esc' key is pressed, break loop
			if (cv::waitKey(1) == 27)
			{
				std::cout << "Program ended by users." << std::endl;
				break;
			}
			bool inputCheck = videoCapture.read(inputFrame);
			if (!inputCheck&&(currFrameIndex == FRAME_COUNT)) {
				std::cout << "Video having problem. Cannot read the frame from video file." << std::endl;
				return -1;
			}
		}

		tempFinishTime = time(0);
		GenerateProcessTime(FRAME_COUNT, currFolderName);

		std::cout << "Background subtraction completed" << std::endl;

		if (evaluateResultSwitch) {
			if (saveResultSwitch) {

				std::cout << "Now starting evaluate the processed result..." << std::endl;
				EvaluateResult(filename, folderName, currFolderName);
			}
			else {
				std::cout << "No saved results for evaluation" << std::endl;
			}
		}
	}
	std::cout << "Program Completed!" << std::endl;
	Beep(1568, 200);
	Beep(1568, 200);
	Beep(1568, 200);
	Beep(1245, 1000);
	Beep(1397, 200);
	Beep(1397, 200);
	Beep(1397, 200);
	Beep(1175, 1000);
	system("pause");
	return 0;
}
// Get current date/time, format is DDMMYYHHmmss
const std::string currentDateTimeStamp(time_t * now) {
	//time_t     now = time(0);
	struct tm  tstruct;
	char       buf[80];
	tstruct = *localtime(now);
	strftime(buf, sizeof(buf), "%d%m%y-%H%M%S", &tstruct);

	return buf;
}
// Get current date/time, format is DD-MM-YY HH:mm:ss
const std::string currentDateTime(time_t * now) {
	struct tm  tstruct;
	char       buf[80];
	tstruct = *localtime(now);
	strftime(buf, sizeof(buf), "%d-%b-%G %X", &tstruct);

	return buf;
}
// Save current process's parameter
void SaveParameter(std::string folderName) {
	std::ofstream myfile;
	myfile.open(folderName + "/parameter.txt", std::ios::app);
	myfile << "----VERSION----\n";
	myfile << programVersion;
	myfile << "\n";
	myfile << "----MAIN PROCESS PARAMETER----\n";
	myfile << "RESULT FOLDER: ";
	myfile << folderName;
	myfile << "\n";
	myfile.close();
}
// Evaluate results
void EvaluateResult(std::string filename, std::string folderName, std::string currFolderName) {
	// A video folder should contain 2 folders['input', 'groundtruth']
	//and the "temporalROI.txt" file to be valid.The choosen method will be 
	// applied to all the frames specified in \temporalROI.txt

	// Read index from temporalROI.txt
	std::ifstream infile(filename + "/temporalROI.txt");
	int idxFrom, idxTo;
	double TotalShadow = 0, TP = 0, TN = 0, FP = 0, FN = 0, SE = 0;
	infile >> idxFrom >> idxTo;
	infile.close();
	std::string groundtruthFolder = filename + "/groundtruth";
	char s[25];
	for (size_t startIndex = idxFrom;startIndex <= idxTo;startIndex++) {
		sprintf(s, "%06d.png", (startIndex));
		//cv::Mat gtImg = cv::imread( "bungalows/groundtruth/gt000001.png");
		cv::Mat gtImg = cv::imread(groundtruthFolder + "/gt" + s, CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat resultImg = cv::imread(folderName + "/bin" + s, CV_LOAD_IMAGE_GRAYSCALE);
		for (size_t pxPointer = 0;pxPointer < (resultImg.rows*resultImg.cols);pxPointer++) {

			double gtValue = gtImg.data[pxPointer];
			double resValue = resultImg.data[pxPointer];
			if (gtValue == 255) {
				// TP
				if (resValue == 255) {
					TP++;
				}
				// FN
				else {
					FN++;
				}
			}
			else if (gtValue == 50) {
				TotalShadow++;
				// TN
				if (resValue == 0) {
					TN++;
				}
				// SE FP
				else {
					SE++;
					FP++;
				}
			}
			else if (gtValue < 50) {
				// TN
				if (resValue == 0) {
					TN++;
				}
				// SE FP
				else {
					FP++;
				}
			}
		}
	}
	const double recall = TP / (TP + FN);
	const double precision = TP / (TP + FP);
	const double FMeasure = (2.0*recall*precision) / (recall + precision);

	const double specficity = TN / (TN + FP);
	const double FPR = FP / (FP + TN);
	const double FNR = FN / (TP + FN);
	const double PBC = 100.0 * (FN + FP) / (TP + FP + FN + TN);

	std::ofstream myfile;
	myfile.open(currFolderName + "/parameter.txt", std::ios::app);
	myfile << "\n<<<<<-STATISTICS  RESULTS->>>>>\n";
	myfile << "TRUE POSITIVE(TP): " << std::setprecision(0) << std::fixed << TP << std::endl;
	myfile << "FALSE POSITIVE(FP): " << std::setprecision(0) << std::fixed << FP << std::endl;
	myfile << "TRUE NEGATIVE(TN): " << std::setprecision(0) << std::fixed << TN << std::endl;
	myfile << "FALSE NEGATIVE(FN): " << std::setprecision(0) << std::fixed << FN << std::endl;
	myfile << "SHADOW ERROR(SE): " << std::setprecision(0) << std::fixed << SE << std::endl;
	myfile << "RECALL: " << std::setprecision(3) << std::fixed << recall << std::endl;
	myfile << "PRECISION: " << std::setprecision(3) << std::fixed << precision << std::endl;
	myfile << "F-MEASURE: " << std::setprecision(3) << std::fixed << FMeasure << std::endl;
	myfile << "SPECIFICITY: " << std::setprecision(3) << std::fixed << specficity << std::endl;
	myfile << "FPR: " << std::setprecision(3) << std::fixed << FPR << std::endl;
	myfile << "FNR: " << std::setprecision(3) << std::fixed << FNR << std::endl;
	myfile << "PBC: " << std::setprecision(3) << std::fixed << PBC << std::endl;
	myfile << "TOTAL SHADOW: " << std::setprecision(0) << std::fixed << TotalShadow << std::endl;
	myfile.close();
}
// Calculate processing time
void GenerateProcessTime(double FRAME_COUNT,std::string currFolderName) {
	
	double diffSeconds = difftime(tempFinishTime, tempStartTime);
	int seconds, hours, minutes;
	minutes = diffSeconds / 60;
	hours = minutes / 60;
	seconds = int(diffSeconds) % 60;
	double fpsProcess = FRAME_COUNT / diffSeconds;
	std::cout << "<<<<<-TOTAL PROGRAM TIME->>>>>\n" << "PROGRAM START TIME:" << currentDateTime(&tempStartTime) << std::endl;
	std::cout << "PROGRAM END  TIME:" << currentDateTime(&tempFinishTime) << std::endl;
	std::cout << "TOTAL SPEND TIME:" << hours << " H " << minutes << " M " << seconds << " S" << std::endl;
	std::cout << "AVERAGE FPS:" << fpsProcess << std::endl;
	if (saveResultSwitch) {
		std::ofstream myfile;
		myfile.open(currFolderName + "/parameter.txt", std::ios::app);
		myfile << "\n\n<<<<<-TOTAL PROGRAM TIME->>>>>\n";
		myfile << "PROGRAM START TIME:";
		myfile << currentDateTime(&tempStartTime);
		myfile << "\n";
		myfile << "PROGRAM END  TIME:";
		myfile << currentDateTime(&tempFinishTime);
		myfile << "\n";
		myfile << "TOTAL SPEND TIME:";
		myfile << hours << " H " << minutes << " M " << seconds << " S";
		myfile << "\n";
		myfile << "AVERAGE FPS:";
		myfile << fpsProcess;
		myfile << "\n";
		myfile.close();
	}
}

/****Read input methods****/
// Read integer value input
int readIntInput(std::string question)
{
	int input = -1;
	bool valid = false;
	do
	{
		std::cout << question << " :" << std::flush;
		std::cin >> input;
		if (std::cin.good())
		{
			valid = true;
		}
		else
		{
			std::cin.clear();
			std::cin.ignore();
			std::cout << "Invalid input; please re-enter double value only." << std::endl;
		}
	} while (!valid);

	return (input);
}
// Read double value input
double readDoubleInput(std::string question)
{
	double input = -1;
	bool valid = false;
	do
	{
		std::cout << question << " :" << std::flush;
		std::cin >> input;
		if (std::cin.good())
		{
			valid = true;
		}
		else
		{
			std::cin.clear();
			std::cin.ignore();
			std::cout << "Invalid input; please re-enter double value only." << std::endl;
		}
	} while (!valid);

	return (input);
}
// Read boolean value input
bool readBoolInput(std::string question)
{
	int input = 3;
	bool valid = false;
	bool result = false;
	do
	{
		std::cout << question << " :" << std::flush;
		std::cin >> input;
		if (std::cin.good())
		{
			switch (input) {
			case 0:	result = false;
				valid = true;
				break;
			case 1:	result = true;
				valid = true;
				break;
			default:
				std::cin.clear();
				std::cin.ignore(true, '\n');
				std::cout << "Invalid input; please re-enter boolean (0/1) only." << std::endl;
				break;
			}
		}
		else
		{
			std::cin.clear();
			std::cin.ignore(true, '\n');
			std::cout << "Invalid input; please re-enter boolean (0/1) only." << std::endl;
		}
	} while (!valid);

	return (result);
}
// Read video input
cv::VideoCapture readVideoInput(std::string question, std::string *filename,double *FPS,
	double *FRAME_COUNT,cv::Size *FRAME_SIZE )
{
	// Video capture variable
	cv::VideoCapture videoCapture;
	std::string videoName;

	int input = 3;
	bool valid = false;
	bool result = false;
	do
	{
		std::cout << question << " :" << std::flush;
		getline(std::cin, (*filename));
		bool check = false;
		for (size_t formatIndex = 0;formatIndex < 2;formatIndex++) {
			switch (formatIndex) {
			case 0:
				videoName = (*filename) + "/" + (*filename) + ".avi";
				break;
			case 1:
				videoName = (*filename) + "/" + (*filename) + ".mp4";
				break;
			}
			videoCapture.open(videoName);
			// Input frame
			cv::Mat inputFrames;
			videoCapture >> inputFrames;

			// Checking video whether successful be opened
			if (videoCapture.isOpened() && !inputFrames.empty()) {
				check = true;
				break;
			}
		}
		if (check) {
			std::cout << "Video successful loaded!" << std::endl;
			valid = true;
			// Getting frames per second (FPS) of the input video
			(*FPS) = videoCapture.get(cv::CAP_PROP_FPS);
			// Getting total number of frame of the input video
			(*FRAME_COUNT) = videoCapture.get(cv::CAP_PROP_FRAME_COUNT);
			// Getting size of the input video
			(*FRAME_SIZE) = cv::Size(videoCapture.get(cv::CAP_PROP_FRAME_WIDTH), videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT));
			videoCapture.set(cv::CAP_PROP_POS_FRAMES, 0);
			break;
		}
		else {
			std::cout << "\nVideo having problem. Cannot open the video file. Please re-enter valid filename" << std::endl;
			std::cin.sync();
		}
	} while (!valid);

	return (videoCapture);
}
// Read video input
cv::VideoCapture readVideoInput2(std::string question, std::string *filename, double *FPS,
	double *FRAME_COUNT, cv::Size *FRAME_SIZE)
{
	// Video capture variable
	cv::VideoCapture videoCapture;
	std::string videoName;

	int input = 3;
	bool valid = false;
	bool result = false;
	do
	{
		/*std::cout << question << " :" << std::flush;
		getline(std::cin, (*filename));*/
		bool check = false;
		for (size_t formatIndex = 0;formatIndex < 2;formatIndex++) {
			switch (formatIndex) {
			case 0:
				videoName = (*filename) + "/" + (*filename) + ".avi";
				break;
			case 1:
				videoName = (*filename) + "/" + (*filename) + ".mp4";
				break;
			}
			videoCapture.open(videoName);
			// Input frame
			cv::Mat inputFrames;
			videoCapture >> inputFrames;

			// Checking video whether successful be opened
			if (videoCapture.isOpened() && !inputFrames.empty()) {
				check = true;
				break;
			}
		}
		if (check) {
			std::cout << "Video successful loaded!" << std::endl;
			valid = true;
			// Getting frames per second (FPS) of the input video
			(*FPS) = videoCapture.get(cv::CAP_PROP_FPS);
			// Getting total number of frame of the input video
			(*FRAME_COUNT) = videoCapture.get(cv::CAP_PROP_FRAME_COUNT);
			// Getting size of the input video
			(*FRAME_SIZE) = cv::Size(videoCapture.get(cv::CAP_PROP_FRAME_WIDTH), videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT));
			videoCapture.set(cv::CAP_PROP_POS_FRAMES, 0);
			break;
		}
		else {
			std::cout << "\nVideo having problem. Cannot open the video file. Please re-enter valid filename" << std::endl;
			std::cin.sync();
		}
	} while (!valid);

	return (videoCapture);
}