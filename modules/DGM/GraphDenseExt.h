// Extended (dense) Graph class interface;
// Written by Sergey Kosov in 2018 for Project X
#pragma once

#include "types.h"

class SemiMetricFunction;

namespace DirectGraphicalModels 
{
	class CGraphDense;
	// ================================ Extended Dense Graph Class ================================
	/**
	* @brief Extended Dense graph class
	* @ingroup moduleGraph
	* @details This graph class provides additional functionality, when the graph is used for 2d image classification
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CGraphDenseExt 
	{
	public:
		/**
		* @brief Constructor
		* @param graph The graph
		* @param gType The graph type. (Ref. @ref graphType)
		*/
		DllExport CGraphDenseExt(CGraphDense &graph) : m_graph(graph) {}
		DllExport ~CGraphDenseExt(void) {}

		/**
		* @brief Adds the graph nodes with potentials \b pots
		* @details This function builds a 2d graph of size corresponding to the size of the \b pots matrix and fills its nodes with the
		* potentials from the same \b pots matrix.
		* @param pots A block of node potentials: Mat(type: CV_32FC(nStates)). It may be obtained by:
		* @code
		* CTrainNode::getNodePotentials()
		* @endcode
		*/
		DllExport void addNodes(const Mat &pots);
		/**
		* @brief Add a Gaussian pairwise potential model with standard deviations \b sx and \b sy
		* @param graphSize The size of the graph
		* @param sx
		* @param sy
		* @param w
		* @param pFunction
		*/
		DllExport void addGaussianEdgeModel(CvSize graphSize, float sx, float sy, float w = 1.0f, const SemiMetricFunction *pFunction = NULL);
		/**
		* @brief Add a Bilateral pairwise potential with spacial standard deviations sx, sy and color standard deviations sr,sg,sb
		* @param img
		* @param sx 
		* @param sy
		* @param sr
		* @param sg
		* @param sb
		* @param w
		* param pFunction
		*/
		DllExport void addBilateralEdgeModel(const Mat &img, float sx, float sy, float sr, float sg, float sb, float w = 1.0f, const SemiMetricFunction *pFunction = NULL);


	private:
		CGraphDense & m_graph;	///< The graph
	};
}