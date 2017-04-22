#pragma once

#ifndef __Functions_H_INCLUDED
#define __Functions_H_INCLUDED
#include <opencv2\opencv.hpp>
#include <time.h>

// Program version
extern std::string programVersion;
// Program start time
extern time_t tempStartTime;
// Program finish time
extern time_t tempFinishTime;
// Show input frame switch
extern bool showInputSwitch;
// Show output frame switch
extern bool showOutputSwitch;
// Save result switch
extern bool saveResultSwitch;
// Evaluate result switch
extern bool evaluateResultSwitch;

///Read input Functions///
// Read integer value input
int readIntInput(std::string question);
// Read double value input
double readDoubleInput(std::string question);
// Read boolean value input
bool readBoolInput(std::string question);
// Read video input
cv::VideoCapture readVideoInput(std::string question, std::string *filename,
	double *FPS, double *FRAME_COUNT, cv::Size *FRAME_SIZE);
cv::VideoCapture readVideoInput2(std::string *filename, double *FPS, double *FRAME_COUNT, cv::Size *FRAME_SIZE);

/// Time Functions
// Get current date/time, format is DDMMYYHHmmss
const std::string currentDateTimeStamp(time_t * now);
// Get current date/time, format is DD-MM-YY HH:mm:ss
const std::string currentDateTime(time_t * now);

// Save current's process parameter
void SaveParameter(std::string versionFolderName, std::string saveFolderName);
// Evaluate results
void EvaluateResult(std::string filename, std::string saveFolderName, std::string versionFolderName);
// Calculate processing time
void GenerateProcessTime(double FRAME_COUNT, std::string saveFolderName);
#endif