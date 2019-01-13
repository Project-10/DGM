// Extended (pairwise) Graph class interface;
// Written by Sergey Kosov in 2016 for Project X
#pragma once

#include "GraphLayeredExt.h"

namespace DirectGraphicalModels 
{
	class IGraphPairwise;
	
	// ================================ Extended Pairwise Graph Class ================================
	/**
	* @brief Extended Pairwise graph class
	* @ingroup moduleGraph
	* @details This graph class provides additional functionality, when the graph is used for 2d image classification
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CGraphPairwiseExt : public CGraphLayeredExt
	{
	public:
		/**
		* @brief Constructor
		* @param graph The graph
		* @param gType The graph type. (Ref. @ref graphType)
		*/
		DllExport CGraphPairwiseExt(IGraphPairwise& graph, byte gType = GRAPH_EDGES_GRID) : CGraphLayeredExt(graph, 1, gType) {}
		DllExport virtual ~CGraphPairwiseExt(void) = default;


		DllExport virtual void setGraph(const Mat& pots) override
		{
			CGraphLayeredExt::setGraph(pots, Mat());
		}

		DllExport virtual void addDefaultEdgesModel(float val, float weight = 1.0f) override;
		DllExport virtual void addDefaultEdgesModel(const Mat& featureVectors, float val, float weight = 1.0f) override;
		DllExport virtual void addDefaultEdgesModel(const vec_mat_t& featureVectors, float val, float weight = 1.0f) override;

		/**
		* @brief Fills the graph edges with potentials
		* @details This function uses \b edgeTrainer class in oerder to achieve edge potentials from feature vectors, stored in \b featureVectors
		* and fills with them the graph edges
		* > This function supports PPL
		* @param edgeTrainer A pointer to the edge trainer
		* @param featureVectors Multi-channel matrix, each element of which is a multi-dimensinal point: Mat(type: CV_8UC<nFeatures>)
		* @param vParams Array of control parameters. Please refer to the concrete model implementation of the CTrainEdge::calculateEdgePotentials() function for more details
		* @param weight The weighting parameter
		*/
		DllExport void fillEdges(const CTrainEdge& edgeTrainer, const Mat& featureVectors, const vec_float_t& vParams, float weight = 1.0f)
		{
			CGraphLayeredExt::fillEdges(edgeTrainer, NULL, featureVectors, vParams, weight);
		}
		/**
		* @brief Fills the graph edges with potentials
		* @details This function uses \b edgeTrainer class in oerder to achieve edge potentials from feature vectors, stored in \b featureVectors
		* and fills with them the graph edges
		* > This function supports PPL
		* @param edgeTrainer A pointer to the edge trainer
		* @param featureVectors  Vector of size \a nFeatures, each element of which is a single feature - image: Mat(type: CV_8UC1)
		* @param vParams Array of control parameters. Please refer to the concrete model implementation of the CTrainEdge::calculateEdgePotentials() function for more details
		* @param weight The weighting parameter
		*/
		DllExport void fillEdges(const CTrainEdge& edgeTrainer, const vec_mat_t& featureVectors, const vec_float_t& vParams, float weight = 1.0f)
		{
			CGraphLayeredExt::fillEdges(edgeTrainer, NULL, featureVectors, vParams, weight);
		}

	};
}
