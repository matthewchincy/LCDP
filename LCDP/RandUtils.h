#pragma once

/*// gaussian 3x3 pattern, based on 'floor(fspecial('gaussian', 3, 1)*256)'
static const int s_nSamplesInitPatternWidth = 3;
static const int s_nSamplesInitPatternHeight = 3;
static const int s_nSamplesInitPatternTot = 256;
static const int s_anSamplesInitPattern[s_nSamplesInitPatternHeight][s_nSamplesInitPatternWidth] = {
    {19,    32,    19,},
    {32,    52,    32,},
    {19,    32,    19,},
};*/

// gaussian 7x7 pattern, based on 'floor(fspecial('gaussian',7,2)*512)'
static const int s_nSamplesInitPatternWidth = 7;
static const int s_nSamplesInitPatternHeight = 7;
static const int s_nSamplesInitPatternTot = 512;
static const int s_anSamplesInitPattern[s_nSamplesInitPatternHeight][s_nSamplesInitPatternWidth] = {
    {2,     4,     6,     7,     6,     4,     2,},
    {4,     8,    12,    14,    12,     8,     4,},
    {6,    12,    21,    25,    21,    12,     6,},
    {7,    14,    25,    28,    25,    14,     7,},
    {6,    12,    21,    25,    21,    12,     6,},
    {4,     8,    12,    14,    12,     8,     4,},
    {2,     4,     6,     7,     6,     4,     2,},
};

//! returns a random init/sampling position for the specified pixel position; also guards against out-of-bounds values via image/border size check.
static inline void getRandSamplePosition(cv::Point & sampleCoor,const cv::Point currCoor, const int border, const cv::Size& imgsize) {
    int r = 1+rand()%s_nSamplesInitPatternTot;
    for(sampleCoor.x=0; sampleCoor.x<s_nSamplesInitPatternWidth; ++sampleCoor.x) {
		for (sampleCoor.y = 0; sampleCoor.y < s_nSamplesInitPatternHeight; ++sampleCoor.y) {
            r -= s_anSamplesInitPattern[sampleCoor.y][sampleCoor.x];
            if(r<=0)
                goto stop;
        }
    }
    stop:
	sampleCoor.x += currCoor.x-s_nSamplesInitPatternWidth/2;
	sampleCoor.y += currCoor.y-s_nSamplesInitPatternHeight/2;
    if(sampleCoor.x<border)
		sampleCoor.x = border;
    else if(sampleCoor.x >=imgsize.width-border)
		sampleCoor.x = imgsize.width-border-1;
    if(sampleCoor.y<border)
		sampleCoor.y = border;
    else if(sampleCoor.y >=imgsize.height-border)
		sampleCoor.y = imgsize.height-border-1;
}

// simple 8-connected (3x3) neighbors pattern
static const int s_anNeighborPatternSize_3x3 = 8;
static const int s_anNeighborPattern_3x3[8][2] = {
    {-1, 1},  { 0, 1},  { 1, 1},
    {-1, 0},            { 1, 0},
    {-1,-1},  { 0,-1},  { 1,-1},
};

//! returns a random neighbor position for the specified pixel position; also guards against out-of-bounds values via image/border size check.
static inline void getRandNeighborPosition_3x3(int& x_neighbor, int& y_neighbor, const int x_orig, const int y_orig, const int border, const cv::Size& imgsize) {
    int r = rand()%s_anNeighborPatternSize_3x3;
    x_neighbor = x_orig+s_anNeighborPattern_3x3[r][0];
    y_neighbor = y_orig+s_anNeighborPattern_3x3[r][1];
    if(x_neighbor<border)
        x_neighbor = border;
    else if(x_neighbor>=imgsize.width-border)
        x_neighbor = imgsize.width-border-1;
    if(y_neighbor<border)
        y_neighbor = border;
    else if(y_neighbor>=imgsize.height-border)
        y_neighbor = imgsize.height-border-1;
}

// 5x5 neighbors pattern
static const int s_anNeighborPatternSize_5x5 = 24;
static const int s_anNeighborPattern_5x5[24][2] = {
    {-2, 2},  {-1, 2},  { 0, 2},  { 1, 2},  { 2, 2},
    {-2, 1},  {-1, 1},  { 0, 1},  { 1, 1},  { 2, 1},
    {-2, 0},  {-1, 0},            { 1, 0},  { 2, 0},
    {-2,-1},  {-1,-1},  { 0,-1},  { 1,-1},  { 2,-1},
    {-2,-2},  {-1,-2},  { 0,-2},  { 1,-2},  { 2,-2},
};

//! returns a random neighbor position for the specified pixel position; also guards against out-of-bounds values via image/border size check.
static inline void getRandNeighborPosition_5x5(int& x_neighbor, int& y_neighbor, const int x_orig, const int y_orig, const int border, const cv::Size& imgsize) {
    int r = rand()%s_anNeighborPatternSize_5x5;
    x_neighbor = x_orig+s_anNeighborPattern_5x5[r][0];
    y_neighbor = y_orig+s_anNeighborPattern_5x5[r][1];
    if(x_neighbor<border)
        x_neighbor = border;
    else if(x_neighbor>=imgsize.width-border)
        x_neighbor = imgsize.width-border-1;
    if(y_neighbor<border)
        y_neighbor = border;
    else if(y_neighbor>=imgsize.height-border)
        y_neighbor = imgsize.height-border-1;
}
