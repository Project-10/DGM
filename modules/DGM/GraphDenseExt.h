// Extended (dense) Graph class interface;
// Written by Sergey Kosov in 2018 for Project X
#pragma once

#include "types.h"

class SemiMetricFunction;

namespace DirectGraphicalModels 
{
	class CGraphDense;

	class CGraphDenseExt {
	public:
		DllExport CGraphDenseExt(CGraphDense & graph) : m_graph(graph) {}
		DllExport ~CGraphDenseExt(void) {}

		DllExport void setNodes(const Mat &pots);
		
		// Add a Gaussian pairwise potential with standard deviation sx and sy
		DllExport void setEdgesGaussian(CvSize imgSize, float sx, float sy, float w = 1.0f, const SemiMetricFunction *pFunction = NULL);

		// Add a Bilateral pairwise potential with spacial standard deviations sx, sy and color standard deviations sr,sg,sb
		DllExport void setEdgesBilateral(const Mat &img, float sx, float sy, float sr, float sg, float sb, float w = 1.0f, const SemiMetricFunction *pFunction = NULL);


	private:
		CGraphDense & m_graph;
	};
}