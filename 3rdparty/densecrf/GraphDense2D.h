#pragma once

#include "../modules/DGM/GraphDense.h"

namespace DirectGraphicalModels {

	class CGraphDense2D : public CGraphDense {
	public:
		DllExport CGraphDense2D(byte nStates) : CGraphDense(nStates) {}
		DllExport virtual ~CGraphDense2D(void) {}

		DllExport virtual void setNodes(const Mat &pots);

		// Add a Gaussian pairwise potential with standard deviation sx and sy
		DllExport void setEdgesGaussian(CvSize imgSize, float sx, float sy, float w = 1.0f, const SemiMetricFunction *pFunction = NULL);

		// Add a Bilateral pairwise potential with spacial standard deviations sx, sy and color standard deviations sr,sg,sb
		DllExport void setEdgesBilateral(const Mat &img, float sx, float sy, float sr, float sg, float sb, float w = 1.0f, const SemiMetricFunction *pFunction = NULL);
	};

}