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
		DllExport virtual void		setNode(size_t node, const Mat &pot);
		DllExport virtual void		getNode(size_t node, Mat &pot) const;
		
		DllExport virtual size_t	getNumNodes(void) const { return m_vNodePotentials.size() / m_nStates; }
		DllExport virtual size_t	getNumEdges(void) const { return getNumNodes() * (getNumNodes() - 1) / 2; }


	private:
		vec_float_t m_vNodePotentials;		///< 
	};
}
