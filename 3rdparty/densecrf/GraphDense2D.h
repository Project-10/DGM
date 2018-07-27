#pragma once

#include "densecrf.h"

class CGraphDense2D : public DenseCRF {
public:
	DllExport CGraphDense2D(byte nStates) : DenseCRF(nStates) {}
	DllExport virtual ~CGraphDense2D(void) {}

	DllExport virtual void setNodes(const Mat &pots);

	// Add a Gaussian pairwise potential with standard deviation sx and sy
	DllExport void addPairwiseGaussian(CvSize imgSize, float sx, float sy, float w, const SemiMetricFunction * function = NULL);

	// Add a Bilateral pairwise potential with spacial standard deviations sx, sy and color standard deviations sr,sg,sb
	DllExport void addPairwiseBilateral(const Mat &img, float sx, float sy, float sr, float sg, float sb, float w, const SemiMetricFunction *function = NULL);
};
