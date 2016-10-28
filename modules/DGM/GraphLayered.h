// Extended (pairwise) Layered Graph class interface;
// Written by Sergey Kosov in 2016 for Project X
#pragma once

#include "Graph.h"

namespace DirectGraphicalModels
{
	class CTrainNode;
	class CTrainEdge;
	class CTrainLink;

	/**
	* @brief Graph types
	* @details Define the grid of edges, connecting nodes of the graph
	*/
	enum graphType {
		GRAPH_EDGES_NONE = 0,		///< No edges
		GRAPH_EDGES_GRID = 1,		///< Vertical and horizontal edges
		GRAPH_EDGES_DIAG = 2,		///< Diagonal edges
		GRAPH_EDGES_LINK = 4		///< Links (inter-layer edges)
	};

	// ================================ Extended Graph Class ================================
	/**
	* @brief Extended Pairwise Layered graph class
	* @ingroup moduleGraph
	* @details This graph class provides additional functionality, when the multi-layer graph is used for 2d image classification
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CGraphLayered : public CGraph
	{
	public:
		/**
		* @brief Constructor
		* @param nStates The number of States (classes)
		* @param nLayers The number of layers
		* @param gType The graph type. (Ref. @ref graphType)
		*/
		DllExport CGraphLayered(byte nStates, word nLayers, byte gType = GRAPH_EDGES_GRID) : CGraph(nStates), m_nLayers(nLayers), m_gType(gType), m_size(cvSize(0, 0)) {}
		DllExport virtual ~CGraphLayered(void) {}

		/**
		* @brief Builds a graph, which fits the image resolution
		* @details The graph is built under the assumption that each graph node is connected with arcs to its direct four neighbours.
		* @param graphSize The size of the graph
		*/
		DllExport virtual void build(CvSize graphSize);
		/**
		* @brief Fills the graph nodes with potentials
		* @details This function uses \b nodeTrainerBase and \b nodeTrainerOccl classes in order to achieve none potentials from feature vectors, stored 
		* in \b featureVectors and fills with them the graph nodes
		* > This function supports PPL
		* @param nodeTrainerBase A pointer to the node trainer
		* @param nodeTrainerOccl A pointer to the node trainer
		* @param featureVectors Multi-channel matrix, each element of which is a multi-dimensinal point: Mat(type: CV_8UC<nFeatures>)
		* @param weightBase The weighting parameter for base layer
		* @param weightOccl The weighting parameter for occlusion layer
		*/
		DllExport virtual void fillNodes(const CTrainNode *nodeTrainerBase, const CTrainNode *nodeTrainerOccl, const Mat &featureVectors, float weightBase = 1.0f, float weightOccl = 1.0f);
		/**
		* @brief Fills the graph nodes with potentials
		* @details This function uses \b nodeTrainerBase and \b nodeTrainerOccl classes in order to achieve none potentials from feature vectors, stored
		* in \b featureVectors and fills with them the graph nodes
		* > This function supports PPL
		* @param nodeTrainerBase A pointer to the node trainer
		* @param nodeTrainerOccl A pointer to the node trainer
		* @param featureVectors Vector of size \a nFeatures, each element of which is a single feature - image: Mat(type: CV_8UC1)
		* @param weightBase The weighting parameter for base layer
		* @param weightOccl The weighting parameter for occlusion layer
		*/
		DllExport virtual void fillNodes(const CTrainNode *nodeTrainerBase, const CTrainNode *nodeTrainerOccl, const vec_mat_t &featureVectors, float weightBase = 1.0f, float weightOccl = 1.0f);
		/**
		* @brief Fills the graph edges with potentials
		* @details This function uses \b edgeTrainer class in oerder to achieve edge potentials from feature vectors, stored in \b featureVectors
		* and fills with them the graph edges
		* > This function supports PPL
		* @param edgeTrainer A pointer to the edge trainer
		* @param linkTrainer A pointer to tht link (inter-layer edge) trainer
		* @param featureVectors Multi-channel matrix, each element of which is a multi-dimensinal point: Mat(type: CV_8UC<nFeatures>)
		* @param params Array of control parameters. Please refer to the concrete model implementation of the CTrainEdge::calculateEdgePotentials() function for more details
		* @param params_len The length of the \b params parameter
		* @param edgeWeight The weighting parameter for (within-layer) edges 
		* @param linkWeight The weighting parameter for (inter-layer) edges, \a i.e. links
		*/
		DllExport virtual void fillEdges(const CTrainEdge *edgeTrainer, const CTrainLink *linkTrainer, const Mat &featureVectors, float *params, size_t params_len, float edgeWeight = 1.0f, float linkWeight = 1.0f);
		/**
		* @brief Fills the graph edges with potentials
		* @details This function uses \b edgeTrainer class in oerder to achieve edge potentials from feature vectors, stored in \b featureVectors
		* and fills with them the graph edges
		* > This function supports PPL
		* @param edgeTrainer A pointer to the edge trainer
		* @param linkTrainer A pointer to tht link (inter-layer edge) trainer
		* @param featureVectors Vector of size \a nFeatures, each element of which is a single feature - image: Mat(type: CV_8UC1)
		* @param params Array of control parameters. Please refer to the concrete model implementation of the CTrainEdge::calculateEdgePotentials() function for more details
		* @param params_len The length of the \b params parameter
		* @param edgeWeight The weighting parameter for (within-layer) edges
		* @param linkWeight The weighting parameter for (inter-layer) edges, \a i.e. links
		*/
		DllExport virtual void fillEdges(const CTrainEdge *edgeTrainer, const CTrainLink *linkTrainer, const vec_mat_t &featureVectors, float *params, size_t params_len, float edgeWeight = 1.0f, float linkWeight = 1.0f);
		/**
		* @brief Assign the edges, which cross the given line to the grop \b group.
		* @details The line is given by the equation: <b>A</b>x + <b>B</b>y + <b>C</b> = 0. \b A and \b B are not both equal to zero.
		* @param A Constant line parameter
		* @param B Constant line parameter
		* @param C Constant line parameter
		* @param group New group ID
		*/
		DllExport virtual void defineEdgeGroup(float A, float B, float C, byte group);
		/**
		* @brief
		* @param group
		* @param pot
		*/
		DllExport virtual void setGroupPot(byte group, const Mat &pot);
		/**
		* @brief Marginalizes a set of nodes
		* @details This function separates the marginalized graph nodes by removing all the edges connecting them with the remaining nodes.
		* New edges are added if they correspond to the inducing pathes. The potentials of new esges are calculated as the sum of edge potentials from the 
		* corresponding inducing path.
		* > This functions operates with inducing pathes with maximal length of 3 nodes.
		* @param nodes Set of nodes to be marginalized out from the graph
		*/
		DllExport virtual void marginalize(const vec_size_t &nodes);



	protected:
		word	m_nLayers;			///< Number of layers
		byte	m_gType;			///< Graph type (Ref. @ref graphType)
		CvSize	m_size;				///< Size of the graph
	};
}
