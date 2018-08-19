#include "Graph.h"

namespace DirectGraphicalModels 
{
	void	CGraph::addNodes(const Mat &pots) {
		for (int n = 0; n < pots.rows; n++)
			addNode(pots.row(n).t());
	}

	void	CGraph::setNodes(const Mat &pots, size_t start_node) {
		for (int n = 0; n < pots.rows; n++)
			setNode(start_node + n, pots.row(n).t());
	}
}