#pragma once
#include <vector>
#ifndef __BackgroundSubtractorLCDP_H_INCLUDED
#define __BackgroundSubtractorLCDP_H_INCLUDED
class BackgroundSubtractorLCDP {
public:
	// Constructer
	BackgroundSubtractorLCDP();

	// Destructer
	~BackgroundSubtractorLCDP();

	struct descriptor {
		int frameCount;
		int p;
		int q;
		std::vector<int> LCDP;
	};
	
	// Parameters and Model initialize
	void Initialize();
	// Descriptor Generator
	void DescriptorGenerator();
};
#endif