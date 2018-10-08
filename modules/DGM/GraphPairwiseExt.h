// Extended (pairwise) Graph class interface;
// Written by Sergey Kosov in 2016 for Project X
#pragma once

#include "types.h"
#include "GraphLayered.h"

namespace DirectGraphicalModels 
{
	// ================================ Extended Pairwise Graph Class ================================
	/**
	* @brief Extended Pairwise graph class
	* @ingroup moduleGraph
	* @details This graph class provides additional functionality, when the graph is used for 2d image classification
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CGraphPairwiseExt
	{
	public:
		/**
		* @brief Constructor
		* @param graph The graph
		* @param gType The graph type. (Ref. @ref graphType)
		*/
		DllExport CGraphPairwiseExt(CGraphPairwise &graph, byte gType = GRAPH_EDGES_GRID) : m_pGraphML(new CGraphLayered(graph, 1, gType)) {}
		DllExport ~CGraphPairwiseExt(void) {}

        /**
         * @brief Builds a graph, which fits the image resolution
         * @details The graph is built under the assumption that each graph node is connected with arcs to its direct four neighbours.
         * @param graphSize The size of the graph
         */
        DllExport void addNodes(CvSize graphSize)
        {
            m_pGraphML->addNodes(graphSize);
        }
        /**
		* @brief Fills the graph nodes with potentials
		* @details
		* > This function supports PPL
		* @param pots A block of potentials: Mat(type: CV_32FC(nStates))
		*/
		DllExport void setNodes(const Mat &pots)
		{
			m_pGraphML->setNodes(pots, Mat());
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
			m_pGraphML->fillEdges(edgeTrainer, NULL, featureVectors, params, params_len, weight);
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
			m_pGraphML->fillEdges(edgeTrainer, NULL, featureVectors, params, params_len, weight);
		}
        /**
         * @brief Assign the edges, which cross the given line to the grop \b group.
         * @details The line is given by the equation: <b>A</b>x + <b>B</b>y + <b>C</b> = 0. \b A and \b B are not both equal to zero.
         * @param A Constant line parameter
         * @param B Constant line parameter
         * @param C Constant line parameter
         * @param group New group ID
         */
        DllExport void defineEdgeGroup(float A, float B, float C, byte group)
        {
            m_pGraphML->defineEdgeGroup(A, B, C, group);
        }
        /**
         * @brief Sets potential \b pot to all edges in the group \b group
         * @param group The edge group ID
         * @param pot %Edge potential matrix: Mat(size: nStates x nStates; type: CV_32FC1)
         */
        DllExport void setGroupPot(byte group, const Mat &pot)
        {
            m_pGraphML->setGroupPot(group, pot);
        }
        /**
         * @brief Returns the type of the graph
         * @returns The type of the graph (Ref. @ref graphType)
         */
        DllExport byte getType(void) const
        {
            return m_pGraphML->getType();
        }
        /**
         * @brief Returns the size of the graph
         * @return The size of the Graph
         */
        DllExport CvSize getSize(void) const
        {
            return m_pGraphML->getSize();
        }
        
        
    private:
        std::unique_ptr<CGraphLayered> m_pGraphML;          ///< Enclosure of the multi-layer graph
	};
}
