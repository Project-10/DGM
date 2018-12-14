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
		DllExport CGraphDenseExt(CGraphDense &graph) : m_graph(graph) {}
		DllExport ~CGraphDenseExt(void) {}

        DllExport virtual void addNodes(Size graphSize);
		DllExport virtual void setNodes(const Mat &pots);
		DllExport virtual void addDefaultEdgesModel(float val, float weight = 1.0f)
		{
			addGaussianEdgeModel(Vec2f::all(val * 0.03f), weight);
		}
		DllExport virtual void addDefaultEdgesModel(const Mat &featureVectors, float val, float weight = 1.0f)
		{
			addBilateralEdgeModel(featureVectors, Vec2f::all(val), 12.75f, weight);
		}
        DllExport virtual void addDefaultEdgesModel(const vec_mat_t &featureVectors, float val, float weight = 1.0f)
        {
            addBilateralEdgeModel(featureVectors, Vec2f::all(val), 12.75f, weight);
        }

		/**
		* @brief Add a Gaussian potential model with standard deviation \b sigma
		* @param sigma The spatial standard deviation of the 2D-gaussian filter 
		* @param weight
		* @param pFunction
		*/
        DllExport void addGaussianEdgeModel(Vec2f sigma, float weight = 1.0f, const std::function<void(const Mat &src, Mat &dst)> &SemiMetricFunction = {});
		/**
		* @brief Add a Bilateral pairwise potential with spacial standard deviations sx, sy and color standard deviations sr,sg,sb
		* @param img
		* @param s 
		* @param srgb
		* @param weight
		* param pFunction
		*/
        DllExport void addBilateralEdgeModel(const Mat &featureVectors, Vec2f s, float srgb, float weight = 1.0f, const std::function<void(const Mat &src, Mat &dst)> &SemiMetricFunction = {});
        /**
        * @brief Add a Bilateral pairwise potential with spacial standard deviations sx, sy and color standard deviations sr,sg,sb
        * @param img
        * @param s
        * @param srgb
        * @param weight
        * param pFunction
        */
        DllExport void addBilateralEdgeModel(const vec_mat_t &featureVectors, Vec2f s, float srgb, float weight = 1.0f, const std::function<void(const Mat &src, Mat &dst)> &SemiMetricFunction = {});
        DllExport Size getSize(void) const { return m_size; }


	private:
        CGraphDense& m_graph;	///< The graph
        Size         m_size;    ///< Size of the graph
	};
}
