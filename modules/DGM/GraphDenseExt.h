// Extended (dense) Graph class interface;
// Written by Sergey Kosov in 2018 for Project X
#pragma once

#include "GraphExt.h"

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
	class CGraphDenseExt : public CGraphExt
	{
	public:
		/**
		* @brief Constructor
		* @param graph The graph
		*/
		DllExport CGraphDenseExt(CGraphDense& graph) : m_graph(graph) {}
        CGraphDenseExt(const CGraphDenseExt&) = delete;
        DllExport ~CGraphDenseExt() = default;
        const CGraphDenseExt& operator= (const CGraphDenseExt&) = delete;

        DllExport void addNodes(Size graphSize);
        /**
		* @brief Fills or adds the graph nodes with potentials \b pots
		* @details This function builds a 2d graph of size corresponding to the size of the \b pots matrix and fills its nodes with the
		* potentials from the same \b pots matrix.
		* @param pots A block of node potentials: Mat(type: CV_32FC(nStates)). It may be obtained by:
		* @code
		* CTrainNode::getNodePotentials()
		* @endcode
		*/
		DllExport void setNodes(const Mat &pots);
		/**
		* @brief Add a Gaussian pairwise potential model with standard deviations \b sx and \b sy
		* @param s
		* @param w
		* @param pFunction
		*/
        DllExport void addGaussianEdgeModel(Vec2f s, float w = 1.0f, const std::function<void(const Mat &src, Mat &dst)> &SemiMetricFunction = {});
		/**
		* @brief Add a Bilateral pairwise potential with spacial standard deviations sx, sy and color standard deviations sr,sg,sb
		* @param img
		* @param s 
		* @param srgb
		* @param w
		* param pFunction
		*/
        DllExport void addBilateralEdgeModel(const Mat &img, Vec2f s, Vec3f srgb, float w = 1.0f, const std::function<void(const Mat &src, Mat &dst)> &SemiMetricFunction = {});
        DllExport Size getSize(void) const { return m_size; }


	private:
        CGraphDense& m_graph;	///< The graph
        Size         m_size;    ///< Size of the graph
	};
}
