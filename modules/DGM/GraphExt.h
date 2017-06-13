// Extended (pairwise) Graph class interface;
// Written by Sergey Kosov in 2016 for Project X
#pragma once

#include "GraphLayered.h"

namespace DirectGraphicalModels 
{
	class CTrainNode;
	class CTrainEdge;
	
	// ================================ Extended Graph Class ================================
	/**
	* @brief Extended Pairwise graph class
	* @ingroup moduleGraph
	* @details This graph class provides additional functionality, when the graph is used for 2d image classification
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CGraphExt : public CGraphLayered
	{
	public:
		/**
		* @brief Constructor
		* @param nStates The number of States (classes)
		* @param gType The graph type. (Ref. @ref graphType)
		*/
		DllExport CGraphExt(byte nStates, byte gType = GRAPH_EDGES_GRID) : CGraphLayered(nStates, 1, gType) {}
		DllExport virtual ~CGraphExt(void) {}

		/**
		* @brief Fills the graph nodes with potentials
		* > This function supports PPL
		* @param pots A block of potentials: Mat(type: CV_32FC(nStates))
		*/
		DllExport void setNodes(const Mat &pots)
		{
			CGraphLayered::setNodes(pots, Mat());
		}
		/**
		* @brief Fills the graph edges with potentials
		* @details This function uses \b edgeTrainer class in oerder to achieve edge potentials from feature vectors, stored in \b featureVectors
		* and fills with them the graph edges
		* > This function supports PPL
		* @param edgeTrainer A pointer to the edge trainer
		* @param featureVectors Multi-channel matrix, each element of which is a multi-dimensinal point: Mat(type: CV_8UC<nFeatures>)
		* @param params Array of control parameters. Please refer to the concrete model implementation of the CTrainEdge::calculateEdgePotentials() function for more details
		* @param params_len The length of the \b params parameter
		* @param weight The weighting parameter
		*/
		DllExport void fillEdges(const CTrainEdge *edgeTrainer, const Mat &featureVectors, float *params, size_t params_len, float weight = 1.0f)
		{
			CGraphLayered::fillEdges(edgeTrainer, NULL, featureVectors, params, params_len, weight);
		}
		/**
		* @brief Fills the graph edges with potentials
		* @details This function uses \b edgeTrainer class in oerder to achieve edge potentials from feature vectors, stored in \b featureVectors
		* and fills with them the graph edges
		* > This function supports PPL
		* @param edgeTrainer A pointer to the edge trainer
		* @param featureVectors  Vector of size \a nFeatures, each element of which is a single feature - image: Mat(type: CV_8UC1)
		* @param params Array of control parameters. Please refer to the concrete model implementation of the CTrainEdge::calculateEdgePotentials() function for more details
		* @param params_len The length of the \b params parameter
		* @param weight The weighting parameter
		*/
		DllExport void fillEdges(const CTrainEdge *edgeTrainer, const vec_mat_t &featureVectors, float *params, size_t params_len, float weight = 1.0f)
		{
			CGraphLayered::fillEdges(edgeTrainer, NULL, featureVectors, params, params_len, weight);
		}
	};
}
