// Extended (pairwise) Graph class interface;
// Written by Sergey Kosov in 2016 for Project X
#pragma once

#include "GraphExt.h"
#include "GraphLayered.h"

namespace DirectGraphicalModels 
{
	class CGraphPairwise;
	
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
		DllExport CGraphPairwiseExt(CGraphPairwise& graph, byte gType = GRAPH_EDGES_GRID) : m_pGraphML(new CGraphLayered(graph, 1, gType)) {}
		DllExport virtual ~CGraphPairwiseExt(void) = default;


		DllExport virtual void buildGraph(Size graphSize) override
		{
			m_pGraphML->buildGraph(graphSize);
		}
		DllExport virtual void setGraph(const Mat& pots) override
		{
			m_pGraphML->setGraph(pots, Mat());
		}
        /**
		* @brief Adds default data-independet edge model
		* @param val Value, specifying the smoothness strength 
        * @param weight The weighting parameter
		*/		
		DllExport virtual void addDefaultEdgesModel(float val, float weight = 1.0f) override;
		/**
		* @brief Adds default contrast-sensitive edge model
		* @param featureVectors Multi-channel matrix, each element of which is a multi-dimensinal point: Mat(type: CV_8UC<nFeatures>)
        * @param val Value, specifying the smoothness strength
        * @param weight The weighting parameter
		*/
		DllExport virtual void addDefaultEdgesModel(const Mat& featureVectors, float val, float weight = 1.0f) override;
		/**
        * @brief Adds default contrast-sensitive edge model
        * @param featureVectors Vector of size \a nFeatures, each element of which is a single feature - image: Mat(type: CV_8UC1)
        * @param val Value, specifying the smoothness strength
        * @param weight The weighting parameter
        */		
		DllExport virtual void addDefaultEdgesModel(const vec_mat_t& featureVectors, float val, float weight = 1.0f) override;

		DllExport virtual Size getSize(void) const override
		{
	        return m_pGraphML->getSize();
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
		DllExport void addFeatureVecs(CTrainEdge &edgeTrainer, const Mat &featureVectors, const Mat &gt)
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
		DllExport void addFeatureVecs(CTrainEdge &edgeTrainer, const vec_mat_t &featureVectors, const Mat &gt)
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
		DllExport void fillEdges(const CTrainEdge& edgeTrainer, const Mat& featureVectors, const vec_float_t& vParams, float weight = 1.0f)
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
		DllExport void fillEdges(const CTrainEdge& edgeTrainer, const vec_mat_t& featureVectors, const vec_float_t& vParams, float weight = 1.0f)
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
        DllExport void setGroupPot(const Mat &pot, byte group)
        {
            m_pGraphML->setGroupPot(pot, group);
        }
        /**
         * @brief Returns the type of the graph
         * @returns The type of the graph (Ref. @ref graphType)
         */
        DllExport byte getType(void) const
        {
            return m_pGraphML->getType();
        }
        
        
    protected:
		std::unique_ptr<CGraphLayered> m_pGraphML;          ///< Enclosure of the multi-layer graph
	};
}
