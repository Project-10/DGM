// Extended (pairwise) Graph class interface;
// Written by Sergey Kosov in 2016 for Project X
#pragma once

#include "GraphExt.h"
#include "GraphLayered.h"
#include "GraphPairwise.h"	
#include "TrainEdgePottsCS.h"
#include "macroses.h" // TODO: delete this

namespace DirectGraphicalModels 
{
	// ================================ Extended Pairwise Graph Class ================================
	/**
	* @brief Extended Pairwise graph class
	* @ingroup moduleGraph
	* @details This graph class provides additional functionality, when the graph is used for 2d image classification
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CGraphPairwiseExt : public CGraphExt
	{
	public:
		/**
		* @brief Constructor
		* @param graph The graph
		* @param gType The graph type. (Ref. @ref graphType)
		*/
		DllExport CGraphPairwiseExt(CGraphPairwise &graph, byte gType = GRAPH_EDGES_GRID) : m_pGraphML(new CGraphLayered(graph, 1, gType)) {}
		DllExport virtual ~CGraphPairwiseExt(void) {}

        /**
         * @brief Builds a graph, which fits the image resolution
         * @details The graph is built under the assumption that each graph node is connected with arcs to its direct four neighbours.
         * @param graphSize The size of the graph
         */
        DllExport virtual void addNodes(cv::Size graphSize)
        {
            m_pGraphML->addNodes(graphSize);
        }
        /**
		* @brief Fills the graph nodes with potentials
		* @details
		* > This function supports PPL
		* @param pots A block of potentials: Mat(type: CV_32FC(nStates))
		*/
		DllExport virtual void setNodes(const Mat &pots)
		{
			m_pGraphML->setNodes(pots, Mat());
		}
		/**
		* @brief Adds default data-independet edge model
		*/
		DllExport virtual void addDefaultEdgesModel(float val, float weight = 1.0f)
		{
            const byte nStates = m_pGraphML->getGraph().getNumStates();

            // Assertions
			DGM_ASSERT(m_pGraphML->getSize().width * m_pGraphML->getSize().height == m_pGraphML->getGraph().getNumNodes());

			Mat ePot = CTrainEdge::getDefaultEdgePotentials(val, nStates);
#ifdef ENABLE_PPL
            concurrency::parallel_for(0, m_pGraphML->getSize().height, [&](int y) {
#else 
            for (int y = 0; y < m_pGraphML->getSize().height; y++) {
#endif
                for (int x = 0; x < m_pGraphML->getSize().width; x++) {
                    size_t idx = y * m_pGraphML->getSize().width + x;
                    if (m_pGraphML->getType() & GRAPH_EDGES_GRID) {
                        if (x > 0)												m_pGraphML->getGraph().setArc(idx, idx - 1, ePot);
                        if (y > 0)												m_pGraphML->getGraph().setArc(idx, idx - 1 * m_pGraphML->getSize().width, ePot);
                    } // edges_grid

                    if (m_pGraphML->getType() & GRAPH_EDGES_DIAG) {
                        if ((x > 0) && (y > 0))									m_pGraphML->getGraph().setArc(idx, idx - m_pGraphML->getSize().width - 1, ePot);
                        if ((x < m_pGraphML->getSize().width - 1) && (y > 0))	m_pGraphML->getGraph().setArc(idx, idx - m_pGraphML->getSize().width + 1, ePot);
                    } // edges_diag
                } // x
#ifdef ENABLE_PPL
            }); // y
#else
            } // y
#endif
		}

		DllExport virtual void addDefaultEdgesModel(const Mat &featureVectors, float val, float weight = 1.0f)
		{
            const byte nStates = m_pGraphML->getGraph().getNumStates();
            const word nFeatures = featureVectors.channels();
            const CTrainEdge &edgeTrainer = CTrainEdgePottsCS(nStates, nFeatures);
            fillEdges(&edgeTrainer, featureVectors, { val, 0.01f }, weight);
		}

        DllExport virtual void addDefaultEdgesModel(const vec_mat_t &featureVectors, float val, float weight = 1.0f)
        {
            const byte nStates = m_pGraphML->getGraph().getNumStates();
            const word nFeatures = static_cast<word>(featureVectors.size());
            const CTrainEdge &edgeTrainer = CTrainEdgePottsCS(nStates, nFeatures);
            fillEdges(&edgeTrainer, featureVectors, { val, 0.01f }, weight);
        }
		/**
		* @brief Adds a block of new feature vectors
		* @details This function may be used only for basic graphical models, built with the CGraphExt::build() method. It extracts
		* pairs of feature vectors with corresponding ground-truth values from blocks \b featureVectors and \b gt, according to the graph structure,
		* provided via \b pGraph
		* @param edgeTrainer A pointer to the edge trainer
		* @param featureVectors Multi-channel matrix, each element of which is a multi-dimensinal point: Mat(type: CV_8UC<nFeatures>)
		* @param gt Matrix, each element of which is a ground-truth state (class)
		*/
		DllExport void addFeatureVecs(CTrainEdge *edgeTrainer, const Mat &featureVectors, const Mat &gt)
		{
			m_pGraphML->addFeatureVecs(edgeTrainer, featureVectors, gt);
		}
		/**
		* @brief Adds a block of new feature vectors
		* @details This function may be used only for basic graphical models, built with the CGraphExt::build() method. It extracts
		* pairs of feature vectors with corresponding ground-truth values from blocks \b featureVectors and \b gt, according to the graph structure,
		* provided via \b pGraph
		* @param edgeTrainer A pointer to the edge trainer
		* @param featureVectors Vector of size \a nFeatures, each element of which is a single feature - image: Mat(type: CV_8UC1)
		* @param gt Matrix, each element of which is a ground-truth state (class)
		*/
		DllExport void addFeatureVecs(CTrainEdge *edgeTrainer, const vec_mat_t &featureVectors, const Mat &gt)
		{
			m_pGraphML->addFeatureVecs(edgeTrainer, featureVectors, gt);
		}
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
		DllExport void fillEdges(const CTrainEdge *edgeTrainer, const Mat &featureVectors, const vec_float_t &vParams, float weight = 1.0f)
		{
			m_pGraphML->fillEdges(edgeTrainer, NULL, featureVectors, vParams, weight);
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
		DllExport void fillEdges(const CTrainEdge *edgeTrainer, const vec_mat_t &featureVectors, const vec_float_t &vParams, float weight = 1.0f)
		{
			m_pGraphML->fillEdges(edgeTrainer, NULL, featureVectors, vParams, weight);
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
        DllExport virtual Size getSize(void) const {
            return m_pGraphML->getSize();
        }
        
        
    private:
        std::unique_ptr<CGraphLayered> m_pGraphML;          ///< Enclosure of the multi-layer graph
	};
}
