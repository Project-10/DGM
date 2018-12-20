// Dense Graph class interface
// Written by Sergey G. Kosov in 2018 for Project X 
#pragma once

#include "Graph.h"

class CEdgePotential;

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
		DllExport virtual void		reset(void) override { m_nodePotentials.release(); m_vpEdgeModels.clear(); }
		
		DllExport virtual size_t	addNode(const Mat &pot = EmptyMat) override;
		DllExport virtual void		addNodes(const Mat &pots) override;

		DllExport virtual void		setNode(size_t node, const Mat &pot) override;
		DllExport virtual void		setNodes(size_t start_node, const Mat &pots) override;
		
		DllExport virtual void		getNode(size_t node, Mat &pot) const override;
		
		DllExport virtual size_t	getNumNodes(void) const override { return static_cast<size_t>(m_nodePotentials.rows); }
		DllExport virtual size_t	getNumEdges(void) const override { return getNumNodes() * (getNumNodes() - 1) / 2; }

		// Own
        /**
		 * @brief Adds an edge model
		 * @param pEdgeModel Poiter to an dense edge model
		 */
		DllExport void				addEdgeModel(CEdgePotential *pEdgeModel) { m_vpEdgeModels.emplace_back(pEdgeModel); }
        
        // For internal use
        /**
         * @brief Returns the container with node potentials
         * @return the container with node potentials: Mat(nNodes, nStates, CV_32FC1)
         */
        Mat                         getNodes(void) const { return m_nodePotentials; }
        /**
         * @brief Returns the contener with edge models
         * @details One edge model applies itself to all the edges in the graph
         * @return The container with edge models: vector of size: number of used edge models
         */
        std::vector<CEdgePotential*>& getEdgeModels(void) const { return m_vpEdgeModels; }
        
        
	private:
		/**
		* The container for the node potentials. 
		* Every row is a node potential vector. Thus the size of the matrix is width: nStates; height: nNodes
		*/
		Mat										m_nodePotentials;
		/**
		* The set of edge models
		*/
		mutable std::vector<CEdgePotential*>    m_vpEdgeModels;
	};
}
