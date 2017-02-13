#include <iostream>
#include <direct.h>
#include <opencv2\opencv.hpp>
#include "BackgroundSubtractorLCDP.h"
#include <time.h>
#include <fstream>
#include <iomanip>
#include<conio.h>
#include<windows.h>

// Get current date/time, format is DDMMYYHHmmss
const std::string currentDateTimeStamp(time_t * now);
// Get current date/time, format is DD-MM-YY HH:mm:ss
const std::string currentDateTime(time_t * now);
// Save current's process parameter
void SaveParameter(std::string folderName);
// Evaluate results
void EvaluateResult(std::string filename, std::string folderName, std::string currFolderName);

std::string showInput;
std::string showOutput;
std::string saveResult;
std::string evaluateResult;
std::string inputLCDPThreshold;
int main() {
	
	//EvaluateResult("bungalows", "bungalows/LCD Sample Consensus-300117-004202/results", "bungalows/LCD Sample Consensus-300117-004202");
	//std::string filename = "../../fall.avi";
	//std::string filename = "C:/Users/mattc/Desktop/Background Subtraction/Results/bungalows/bungalows.avi";
	std::string filename;
	std::cout << "Video folder? ";
	getline(std::cin, filename);
	const std::string videoName = filename + "/" + filename + ".avi";
	std::cout << "Show input frame(1/0)?";
	getline(std::cin, showInput);
	std::cout << "Show output frame(1/0)?";
	getline(std::cin, showOutput);
	std::cout << "LCDP Threshold(0-Default)?";
	getline(std::cin, inputLCDPThreshold);
	std::cout << "Save output frame(1/0)?";
	getline(std::cin, saveResult);
	std::cout << "Evaluate result(1/0)?";
	getline(std::cin, evaluateResult);


	cv::VideoCapture videoCapture;
	videoCapture.open(videoName);
	// Input frame
	cv::Mat inputFrame;
	// FG Mask
	cv::Mat fgMask;
	videoCapture >> inputFrame;
	videoCapture.set(cv::CAP_PROP_POS_FRAMES, 0);

	// Checking video whether successful be opened
	if (!videoCapture.isOpened() || inputFrame.empty()) {
		std::cout << "Video having problem. Cannot open the video file." << std::endl;
		return -1;
	}
	else {
		std::cout << "Video successful loaded!" << std::endl;
	}
	// Getting frames per second (FPS) of the input video
	const double FPS = videoCapture.get(cv::CAP_PROP_FPS);
	const double FRAME_COUNT = videoCapture.get(cv::CAP_PROP_FRAME_COUNT);
	const cv::Size FRAME_SIZE = cv::Size(videoCapture.get(cv::CAP_PROP_FRAME_WIDTH), videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT));
	cv::Mat ROI_FRAME;
	ROI_FRAME.create(FRAME_SIZE, CV_8UC1);
	ROI_FRAME = cv::Scalar_<uchar>(255);
	// Display windows
	cv::namedWindow("Input Video");
	size_t maxWordsNo = 25;
	// Background Subtractor Initialize
	BackgroundSubtractorLCDP backgroundSubtractorLCDP(FRAME_SIZE, ROI_FRAME, maxWordsNo);
	if (std::stoi(inputLCDPThreshold) > 0) {
		backgroundSubtractorLCDP.EditLCDPThreshold(std::stod(inputLCDPThreshold));
	}
	backgroundSubtractorLCDP.Initialize(inputFrame, inputFrame);

	// current date/time based on current system
	time_t now = time(0);
	//tm *ltm = localtime(&now);
	std::string startTime = currentDateTimeStamp(&now);
	std::string folderName = filename + "/LCD Sample Consensus-" + startTime;
	const std::string currFolderName = folderName;
	const char *s1;
	if (std::stoi(saveResult) == 1) {
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
		//system("cls");
		//std::cout << "Current process status: " << currFrameIndex << "/" << FRAME_COUNT << " (" << ((currFrameIndex / FRAME_COUNT)*100) << "%)";
		bool inputCheck = videoCapture.read(inputFrame);
		if (!inputCheck) {
			std::cout << "Video having problem. Cannot read the frame from video file." << std::endl;
		}
		backgroundSubtractorLCDP.Process(inputFrame, fgMask);
		if (std::stoi(showInput) == 1) {
			cv::imshow("Input Video", inputFrame);
		}
		if (std::stoi(showOutput) == 1) {
			cv::imshow("Results", fgMask);
		}
		if (std::stoi(saveResult) == 1) {
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
	}
	time_t finish = time(0);
	double diffSeconds = difftime(finish, now);
	int seconds, hours, minutes;
	minutes = diffSeconds / 60;
	hours = minutes / 60;
	seconds = int(diffSeconds) % 60;
	double fpsProcess = FRAME_COUNT / diffSeconds;
	std::ofstream myfile;
	myfile.open(currFolderName + "/parameter.txt", std::ios::app);
	myfile << "\n\n<<<<<-PRE PROCESSING PARAMETER->>>>>\n";
	myfile << "PROGRAM START TIME:";
	myfile << currentDateTime(&now);
	myfile << "\n";
	myfile << "PROGRAM END  TIME:";
	myfile << currentDateTime(&finish);
	myfile << "\n";
	myfile << "TOTAL SPEND TIME:";
	myfile << hours << " H " << minutes << " M " << seconds << " S";
	myfile << "\n";
	myfile << "AVERAGE FPS:";
	myfile << fpsProcess;
	myfile << "\n";
	myfile.close();

	if (std::stoi(evaluateResult) == 1) {
		std::cout << "Background subtraction completed" << std::endl;
		std::cout << "Now starting evaluate the processed result..." << std::endl;
		EvaluateResult(filename, folderName, currFolderName);
	}
	std::cout << "Completed!" << std::endl;
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
	myfile << "----MAIN PROCESS PARAMETER----\n";
	myfile << "RESULT FOLDER: ";
	myfile << folderName;
	myfile << "\n";
	myfile.close();
}

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