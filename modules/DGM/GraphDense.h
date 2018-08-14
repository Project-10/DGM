// Dense Graph class interface;
// Written by Sergey G. Kosov in 2018 for Project X 
#pragma once

#include "IGraph.h"

namespace DirectGraphicalModels {
	// ================================ Graph Interface Class ================================
	/**
	* @brief Interface class for dense graphical models
	* @ingroup moduleGraph
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CGraphDense : public IGraph {
		/**
		* @brief Constructor
		* @param nStates the number of States (classes)
		*/
		DllExport CGraphDense(byte nStates) : IGraph(nStates) {};
		DllExport virtual ~CGraphDense(void) {}

		// IGraph
		DllExport virtual void		reset(void) { m_vNodePotentials.clear(); }
		
		DllExport virtual size_t	addNode(void);
		DllExport virtual size_t	addNode(const Mat &pot);
//		DllExport virtual void		addNodes(const vec_float_t &pots);

		DllExport virtual void		setNode(size_t node, const Mat &pot);
//		DllExport virtual void		setNodes(const vec_float_t &pots);
		
		DllExport virtual void		getNode(size_t node, Mat &pot) const;
		
		DllExport virtual size_t	getNumNodes(void) const { return m_vNodePotentials.size() / m_nStates; }
		DllExport virtual size_t	getNumEdges(void) const { return getNumNodes() * (getNumNodes() - 1) / 2; }


	private:
		/**
		* The container for the node potentials. 
		* The potentials are stores as follows: node[0]:pot[0], node[0]:pot[1], ..., node[0]:pot[m_nStates-1], node[1]:pot[0], ...
		*/
		vec_float_t m_vNodePotentials;		
	
	};
}
