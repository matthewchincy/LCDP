#include "Functions.h"
#include <iostream>

// Program version
std::string programVersion;
// Program start time
time_t tempStartTime;
// Program finish time
time_t tempFinishTime;
// Show input frame switch
bool showInputSwitch;
// Show output frame switch
bool showOutputSwitch;
// Save result switch
bool saveResultSwitch;
// Evaluate result switch
bool evaluateResultSwitch;

///Read input functions
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
cv::VideoCapture readVideoInput(std::string question, std::string *filename, double *FPS,
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
		std::cin.clear();
		std::cin.ignore();
		std::cout << question << " :" << std::flush;
		getline(std::cin, (*filename));
		bool check = false;
		for (size_t formatIndex = 0; formatIndex < 2; formatIndex++) {
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
cv::VideoCapture readVideoInput2(std::string *filename, double *FPS, double *FRAME_COUNT, cv::Size *FRAME_SIZE)
{
	// Video capture variable
	cv::VideoCapture videoCapture;
	std::string videoName;

	int input = 3;
	bool valid = false;
	bool result = false;
	do
	{
		bool check = false;
		for (size_t formatIndex = 0; formatIndex < 2; formatIndex++) {
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
		}
	} while (!valid);

	return (videoCapture);
}

/// Time Functions
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
void SaveParameter(std::string versionFolderName, std::string saveFolderName) {
	std::ofstream myfile;
	myfile.open(saveFolderName + "/parameter.txt", std::ios::app);
	myfile << "----VERSION----\n";
	myfile << programVersion;
	myfile << "\nRESULT FOLDER: ";
	myfile << saveFolderName;
	myfile << "\n";
	myfile.close();
	myfile.open(versionFolderName + "/parameter.csv", std::ios::app);
	myfile << programVersion << "," << saveFolderName;
	myfile.close();
}
// Evaluate results
void EvaluateResult(std::string filename, std::string saveFolderName, std::string versionFolderName) {
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
	for (size_t startIndex = idxFrom; startIndex <= idxTo; startIndex++) {
		sprintf(s, "%06d.png", (startIndex));
		//cv::Mat gtImg = cv::imread( "bungalows/groundtruth/gt000001.png");
		cv::Mat gtImg = cv::imread(groundtruthFolder + "/gt" + s, CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat resultImg = cv::imread(saveFolderName + "/results/bin" + s, CV_LOAD_IMAGE_GRAYSCALE);
		for (size_t pxPointer = 0; pxPointer < (resultImg.rows*resultImg.cols); pxPointer++) {

			double gtValue = gtImg.data[pxPointer];
			double resValue = resultImg.data[pxPointer];
			if (gtValue == 255) {
				// TP
				if (resValue == 255) {
					TP++;
				}
				// FN
				else if (resValue == 0) {
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
				else if (resValue == 255) {
					SE++;
					FP++;
				}
			}
			else if (gtValue < 50) {
				// TN
				if (resValue == 0) {
					TN++;
				}
				// FP
				else if (resValue == 255) {
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
	myfile.open(saveFolderName + "/parameter.txt", std::ios::app);
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
	std::cout << "\n<<<<<-STATISTICS  RESULTS->>>>>\n";
	std::cout << "RECALL: " << std::setprecision(3) << std::fixed << recall << std::endl;
	std::cout << "PRECISION: " << std::setprecision(3) << std::fixed << precision << std::endl;
	std::cout << "F-MEASURE: " << std::setprecision(3) << std::fixed << FMeasure << std::endl;

	myfile.open(versionFolderName + "/parameter.csv", std::ios::app);
	myfile << "," << std::setprecision(3) << std::fixed << recall << "," << std::setprecision(3) << std::fixed << precision << "," << std::setprecision(3) << std::fixed << FMeasure << "\n";
	myfile.close();
}
// Calculate processing time
void GenerateProcessTime(double FRAME_COUNT, std::string saveFolderName) {

	double diffSeconds = difftime(tempFinishTime, tempStartTime);
	int seconds, hours, minutes;
	minutes = diffSeconds / 60;
	hours = minutes / 60;
	minutes = minutes - (60 * hours);
	seconds = int(diffSeconds) % 60;
	double fpsProcess = FRAME_COUNT / diffSeconds;
	std::cout << "<<<<<-TOTAL PROGRAM TIME->>>>>\n" << "PROGRAM START TIME:" << currentDateTime(&tempStartTime) << std::endl;
	std::cout << "PROGRAM END  TIME:" << currentDateTime(&tempFinishTime) << std::endl;
	std::cout << "TOTAL SPEND TIME:" << hours << " H " << minutes << " M " << seconds << " S" << std::endl;
	std::cout << "AVERAGE FPS:" << fpsProcess << std::endl;
	if (saveResultSwitch) {
		std::ofstream myfile;
		myfile.open(saveFolderName + "/parameter.txt", std::ios::app);
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
