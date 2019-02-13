// Extended (dense) Graph class interface;
// Written by Sergey Kosov in 2018 for Project X
#pragma once

#include "GraphExt.h"

namespace DirectGraphicalModels 
{
	class CGraphDense;
	// ================================ Extended Dense Graph Class ================================
	/**
	* @brief Extended Dense graph class for 2D image classifaction
	* @ingroup moduleGraphExt
	* @details This graph class provides simplified interface and additional functionality, when the graph is used for 2D image classification
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
		DllExport ~CGraphDenseExt(void) = default;

        // From CGraphExt
		DllExport void buildGraph(Size graphSize) override;
		DllExport void setGraph(const Mat &pots)  override;
        /**
		* @brief Adds default data-independet edge model
		* @details This function adds a Gaussian edge model to the dense CRF model
		* @param val Value, specifying the smoothness strength - in this case the spatial standard deviation of the 2D-Gaussian filter scaled by 33.(3)
        * @param weight The weighting parameter
		*/		
		DllExport void addDefaultEdgesModel(float val, float weight = 1.0f) override
		{
			addGaussianEdgeModel(Vec2f::all(val * 0.03), weight);
		}
		/**
		* @brief Adds default contrast-sensitive edge model
		* @details This function adds a Bilateral edge model to the dense CRF model
		* @param featureVectors Multi-channel matrix, each element of which is a multi-dimensinal point: Mat(type: CV_8UC<nFeatures>)
        * @param val Value, specifying the smoothness strength - in this case the spatial standard deviation of the 2D-bilateral filter scaled by 33.(3)
        * @param weight The weighting parameter
		*/		
		DllExport void addDefaultEdgesModel(const Mat& featureVectors, float val, float weight = 1.0f) override
		{
			addBilateralEdgeModel(featureVectors, Vec2f::all(val * 0.03), 20.0f, weight);
		}
		/**
		* @brief Adds default contrast-sensitive edge model
		* @details This function adds a Bilateral edge model to the dense CRF model
		* @param featureVectors  Vector of size \a nFeatures, each element of which is a single feature - image: Mat(type: CV_8UC1)
        * @param val Value, specifying the smoothness strength - in this case the spatial standard deviation of the 2D-bilateral filter scaled by 33.(3)
        * @param weight The weighting parameter
		*/	
		DllExport void addDefaultEdgesModel(const vec_mat_t &featureVectors, float val, float weight = 1.0f) override
        {
            addBilateralEdgeModel(featureVectors, Vec2f::all(val * 0.03), 20.0f, weight);
        }
		DllExport Size getSize(void) const override { return m_size; }

		
		/**
		* @brief Add a Gaussian potential model with standard deviation \b sigma
		* @param sigma The spatial standard deviation of the 2D-Gaussian filter 
		* @param weight The weighting parameter
		* @param semiMetricFunction Reference to a semi-metric function, which arguments \b src and \b dst are: Mat(size: 1 x nFeatures; type: CV_32FC1). 
		* For more details refere to @ref CEdgeModelPotts.
		*/
        DllExport void addGaussianEdgeModel(Vec2f sigma, float weight = 1.0f, const std::function<void(const Mat& src, Mat& dst)>& semiMetricFunction = {});
		/**
		* @brief Add a Bilateral pairwise potential with spacial standard deviations \b sigma and color standard deviations sr,sg,sb
		* @param featureVectors Multi-channel matrix, each element of which is a multi-dimensinal point: Mat(type: CV_8UC<nFeatures>)
		* @param sigma The spatial standard deviation of the 2D-bilateral filter 
		* @param sigma_opt The standard deviation for \b featureVectors
		* @param weight The weighting parameter
		* @param semiMetricFunction Reference to a semi-metric function, which arguments \b src and \b dst are: Mat(size: 1 x nFeatures; type: CV_32FC1). 
		* For more details refere to @ref CEdgeModelPotts.
		*/
        DllExport void addBilateralEdgeModel(const Mat& featureVectors, Vec2f sigma, float sigma_opt = 1.0f, float weight = 1.0f, const std::function<void(const Mat& src, Mat& dst)>& semiMetricFunction = {});
        /**
        * @brief Add a Bilateral pairwise potential with spacial standard deviations sx, sy and color standard deviations sr,sg,sb
        * @param featureVectors Multi-channel matrix, each element of which is a multi-dimensinal point: Mat(type: CV_8UC<nFeatures>)
        * @param sigma The spatial standard deviation of the 2D-bilateral filter 
        * @param sigma_opt The standard deviation for \b featureVectors
        * @param weight The weighting parameter
		* @param semiMetricFunction Reference to a semi-metric function, which arguments \b src and \b dst are: Mat(size: 1 x nFeatures; type: CV_32FC1). 
		* For more details refere to @ref CEdgeModelPotts.
        */
        DllExport void addBilateralEdgeModel(const vec_mat_t& featureVectors, Vec2f sigma, float sigma_opt = 1.0f, float weight = 1.0f, const std::function<void(const Mat& src, Mat& dst)>& semiMetricFunction = {});


	private:
        CGraphDense& m_graph;	///< The graph
        Size         m_size;    ///< Size of the 2D graph
	};
}
