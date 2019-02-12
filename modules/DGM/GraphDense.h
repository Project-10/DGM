// Dense Graph class interface
// Written by Sergey G. Kosov in 2018 for Project X 
#pragma once

#include "Graph.h"

namespace DirectGraphicalModels 
{
    // ================================ Graph Interface Class ================================
	/**
	* @brief Fully-connected (dense) graph class
	* @ingroup moduleGraph
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CGraphDense : public CGraph 
	{
	public:
		/**
		* @brief Constructor
		* @param nStates the number of States (classes)
		*/
		DllExport CGraphDense(byte nStates) : CGraph(nStates), m_nodePotentials(EmptyMat) {}
		DllExport virtual ~CGraphDense(void) = default;

		// CGraph
		DllExport void		reset(void) override { m_nodePotentials.release(); m_vpEdgeModels.clear(); }

		DllExport size_t	addNode(const Mat &pot = EmptyMat) override;
		DllExport void		addNodes(const Mat &pots) override { m_nodePotentials.push_back(pots); }

		DllExport void		setNode(size_t node, const Mat &pot) override;
		DllExport void		setNodes(size_t start_node, const Mat &pots) override;
		
		DllExport void		getNode(size_t node, Mat &pot) const override;
		DllExport void		getNodes(size_t start_node, size_t num_nodes, Mat &pots) const override;
		
		DllExport void		getChildNodes (size_t node, vec_size_t &vNodes) const override;
		DllExport void		getParentNodes(size_t node, vec_size_t &vNodes) const override { getChildNodes(node, vNodes); }

		DllExport size_t	getNumNodes(void) const override { return static_cast<size_t>(m_nodePotentials.rows); }
		DllExport size_t	getNumEdges(void) const override { return getNumNodes() * (getNumNodes() - 1) / 2; }

		// Own
		/**
		* @brief Adds an edge model
		* @details Dense CRF may have multiple edge models. All edge models will be subsequently applied during the iterations of the inference process. 
		* @param pEdgeModel Poiter to an dense edge model (ref. @ref IEdgeModel)
		*/
		DllExport void		addEdgeModel(const ptr_edgeModel_t& pEdgeModel) { m_vpEdgeModels.push_back(pEdgeModel); }
		/**
		* @brief Returns the container with node potentials
		* @return The container with node potentials: Mat(nNodes, nStates, CV_32FC1)
		*/
		Mat					getNodePotentials(void) const { return m_nodePotentials; }
		/**
		* @brief Returns the contener with edge models
		* @details One edge model applies itself to all the edges in the graph
		* @return The container with edge models: vector of size: number of used edge models
		*/
		std::vector<ptr_edgeModel_t>& getEdgeModels(void) const { return m_vpEdgeModels; }


	private:
		Mat										m_nodePotentials;	///< The container for the node potentials:  Mat(nNodes, nStates, CV_32FC1), \a i.e. every row is a node potential vector
		mutable std::vector<ptr_edgeModel_t>    m_vpEdgeModels;		///< The set of edge models
	};
}
