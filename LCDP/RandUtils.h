#pragma once

// gaussian 3x3 pattern, based on 'floor(fspecial('gaussian', 3, 1)*256)'
static const int s_nSamplesInitPatternWidth_3x3 = 3;
static const int s_nSamplesInitPatternHeight_3x3 = 3;
static const int s_nSamplesInitPatternTot_3x3 = 256;
static const int s_anSamplesInitPattern_3x3[s_nSamplesInitPatternHeight_3x3][s_nSamplesInitPatternWidth_3x3] = {
	{19,    32,    19,},
	{32,    52,    32,},
	{19,    32,    19,},
};
//! returns a random init/sampling position for the specified pixel position; also guards against out-of-bounds values via image/border size check.
static inline void getRandSamplePosition_3x3(cv::Point & sampleCoor, const cv::Point currCoor, const int border,
	const cv::Size& imgsize) {
	int r = 1 + rand() % s_nSamplesInitPatternTot_3x3;
	for (sampleCoor.x = 0; sampleCoor.x < s_nSamplesInitPatternWidth_3x3; ++sampleCoor.x) {
		for (sampleCoor.y = 0; sampleCoor.y < s_nSamplesInitPatternHeight_3x3; ++sampleCoor.y) {
			r -= s_anSamplesInitPattern_3x3[sampleCoor.y][sampleCoor.x];
			if (r <= 0)
				goto stop;
		}
	}
stop:
	sampleCoor.x += currCoor.x - s_nSamplesInitPatternWidth_3x3 / 2;
	sampleCoor.y += currCoor.y - s_nSamplesInitPatternHeight_3x3 / 2;
	if (sampleCoor.x < border)
		sampleCoor.x = border;
	else if (sampleCoor.x >= imgsize.width - border)
		sampleCoor.x = imgsize.width - border - 1;
	if (sampleCoor.y < border)
		sampleCoor.y = border;
	else if (sampleCoor.y >= imgsize.height - border)
		sampleCoor.y = imgsize.height - border - 1;
}
// gaussian 7x7 pattern, based on 'floor(fspecial('gaussian',7,2)*512)'
static const int s_nSamplesInitPatternWidth_7x7 = 7;
static const int s_nSamplesInitPatternHeight_7x7 = 7;
static const int s_nSamplesInitPatternTot_7x7 = 512;
static const int s_anSamplesInitPattern_7x7[s_nSamplesInitPatternHeight_7x7][s_nSamplesInitPatternWidth_7x7] = {
	{2,     4,     6,     7,     6,     4,     2,},
	{4,     8,    12,    14,    12,     8,     4,},
	{6,    12,    21,    25,    21,    12,     6,},
	{7,    14,    25,    28,    25,    14,     7,},
	{6,    12,    21,    25,    21,    12,     6,},
	{4,     8,    12,    14,    12,     8,     4,},
	{2,     4,     6,     7,     6,     4,     2,},
};

//! returns a random init/sampling position for the specified pixel position; also guards against out-of-bounds values via image/border size check.
static inline void getRandSamplePosition_7x7(cv::Point & sampleCoor, const cv::Point currCoor, const int border,
	const cv::Size& imgsize) {
	int r = 1 + rand() % s_nSamplesInitPatternTot_7x7;
	for (sampleCoor.x = 0; sampleCoor.x < s_nSamplesInitPatternWidth_7x7; ++sampleCoor.x) {
		for (sampleCoor.y = 0; sampleCoor.y < s_nSamplesInitPatternHeight_7x7; ++sampleCoor.y) {
			r -= s_anSamplesInitPattern_7x7[sampleCoor.y][sampleCoor.x];
			if (r <= 0)
				goto stop;
		}
	}
stop:
	sampleCoor.x = sampleCoor.x + currCoor.x - (s_nSamplesInitPatternWidth_7x7 / 2);
	sampleCoor.y = sampleCoor.y + currCoor.y - (s_nSamplesInitPatternHeight_7x7 / 2);
	if (sampleCoor.x < border)
		sampleCoor.x = border;
	else if (sampleCoor.x >= imgsize.width - border)
		sampleCoor.x = imgsize.width - border - 1;
	if (sampleCoor.y < border)
		sampleCoor.y = border;
	else if (sampleCoor.y >= imgsize.height - border)
		sampleCoor.y = imgsize.height - border - 1;
}

// simple 8-connected (3x3) neighbors pattern
static const int s_anNeighborPatternSize_3x3 = 8;
static const int s_anNeighborPattern_3x3[8][2] = {
	{-1, 1},  { 0, 1},  { 1, 1},
	{-1, 0},            { 1, 0},
	{-1,-1},  { 0,-1},  { 1,-1},
};

//! returns a random neighbor position for the specified pixel position; also guards against out-of-bounds values via image/border size check.
static inline void getRandNeighborPosition_3x3(int& x_neighbor, int& y_neighbor, const int x_orig, const int y_orig, const int border,
	const cv::Size& imgsize) {
	int r = rand() % s_anNeighborPatternSize_3x3;
	x_neighbor = x_orig + s_anNeighborPattern_3x3[r][0];
	y_neighbor = y_orig + s_anNeighborPattern_3x3[r][1];
	if (x_neighbor < border)
		x_neighbor = border;
	else if (x_neighbor >= imgsize.width - border)
		x_neighbor = imgsize.width - border - 1;
	if (y_neighbor < border)
		y_neighbor = border;
	else if (y_neighbor >= imgsize.height - border)
		y_neighbor = imgsize.height - border - 1;
}

// 5x5 neighbors pattern
static const int s_nSamplesInitPatternWidth_5x5 = 5;
static const int s_nSamplesInitPatternHeight_5x5 = 5;
static const int s_anNeighborPatternSize_5x5 = 24;
static const int s_anNeighborPattern_5x5[24][2] = {
	{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0},
	{0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 1},
	{0, 2}, {1, 2},         {3, 2}, {4, 2},
	{0, 3}, {1, 3}, {2, 3}, {3, 3}, {4, 3},
	{0, 4}, {1, 4}, {2, 4}, {3, 4}, {4, 4},
};

//! returns a random neighbor position for the specified pixel position; also guards against out-of-bounds values via image/border size check.
static inline void 	getRandSamplePosition_5x5(cv::Point & sampleCoor, const cv::Point currCoor, const int border, const cv::Size& imgsize) {
	int r = rand() % s_anNeighborPatternSize_5x5;
	sampleCoor.x = s_anNeighborPattern_5x5[r][0];
	sampleCoor.y = s_anNeighborPattern_5x5[r][1];

	sampleCoor.x += currCoor.x - s_nSamplesInitPatternWidth_5x5 / 2;
	sampleCoor.y += currCoor.y - s_nSamplesInitPatternHeight_5x5 / 2;
	if (sampleCoor.x < border)
		sampleCoor.x = border;
	else if (sampleCoor.x >= imgsize.width - border)
		sampleCoor.x = imgsize.width - border - 1;
	if (sampleCoor.y < border)
		sampleCoor.y = border;
	else if (sampleCoor.y >= imgsize.height - border)
		sampleCoor.y = imgsize.height - border - 1;
}
