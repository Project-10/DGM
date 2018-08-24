// Dense Graph class interface;
// Written by Sergey G. Kosov in 2018 for Project X 
#pragma once

#include "Graph.h"
#include "densecrf/edgePotentialPotts.h"

namespace DirectGraphicalModels 
{
	// ================================ Graph Interface Class ================================
	/**
	* @brief Interface class for dense graphical models
	* @ingroup moduleGraph
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CGraphDense : public CGraph {
		friend class CInferDense;
		friend class CInferDense1;
	
	public:
		/**
		* @brief Constructor
		* @param nStates the number of States (classes)
		*/
		DllExport CGraphDense(byte nStates) : CGraph(nStates), m_nodePotentials(EmptyMat)         {};
		DllExport virtual ~CGraphDense(void) {}

		// CGraph
		DllExport virtual void		reset(void) { m_nodePotentials.release(); }
		
		DllExport virtual size_t	addNode(void);
		DllExport virtual size_t	addNode(const Mat &pot);
		DllExport virtual void		addNodes(const Mat &pots);

		DllExport virtual void		setNode(size_t node, const Mat &pot);
		DllExport virtual void		setNodes(const Mat &pots, size_t start_node = 0);
		
		DllExport virtual void		getNode(size_t node, Mat &pot) const;
		
		DllExport virtual size_t	getNumNodes(void) const { return static_cast<size_t>(m_nodePotentials.rows); }
		DllExport virtual size_t	getNumEdges(void) const { return getNumNodes() * (getNumNodes() - 1) / 2; }

		/**
		* @brief Adds an edge model
		* @param pEdgeModel Poiter to an dense edge model
		*/
		DllExport void				setEdgeModel(CEdgePotential *pEdgeModel) { m_vpEdgeModels.emplace_back(pEdgeModel); }


	private:
		/**
		* The container for the node potentials. 
		* Every row is a node potential vector. Thus the size of the matrix is width: nStates; height: nNodes
		*/
		Mat												m_nodePotentials;		
		/**
		*/
		std::vector<std::unique_ptr<CEdgePotential>>	m_vpEdgeModels;

	
	};
}
