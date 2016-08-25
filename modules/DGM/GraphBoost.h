#pragma once

#include "IGraph.h"

namespace DirectGraphicalModels {
	class CGraphBoost : public IGraph
	{
	public:
		/**
		* @brief Constructor
		* @param nStates the number of States (classes)
		*/
		DllExport CGraphBoost(byte nStates);
		DllExport virtual ~CGraphBoost(void);


		DllExport virtual void		reset(void) {};
		DllExport virtual size_t	addNode(void) { return 0; };
		DllExport virtual size_t	addNode(const Mat &pot);
		DllExport virtual void		setNode(size_t node, const Mat &pot) {};
		DllExport virtual void		getNode(size_t node, Mat &pot) const {};
		DllExport virtual void		getChildNodes(size_t node, vec_size_t &vNodes) const {};
		DllExport virtual void		getParentNodes(size_t node, vec_size_t &vNodes) const {};
		DllExport virtual void		addEdge(size_t srcNode, size_t dstNode) {};
		DllExport virtual void		addEdge(size_t srcNode, size_t dstNode, const Mat &pot);
		DllExport virtual void		setEdge(size_t srcNode, size_t dstNode, const Mat &pot) {};
		DllExport virtual void		getEdge(size_t srcNode, size_t dstNode, Mat &pot) const {};
		DllExport virtual void		addArc(size_t Node1, size_t Node2) {};
		DllExport virtual void		addArc(size_t Node1, size_t Node2, const Mat &pot) {};
		DllExport virtual void		setArc(size_t Node1, size_t Node2, const Mat &pot) {};
		DllExport virtual size_t	getNumNodes(void) const { return 0; };


	};
}
