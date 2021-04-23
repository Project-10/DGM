#include "Graph.h"
#include "macroses.h"

namespace DirectGraphicalModels 
{
	void CGraph::addNodes(const Mat &pots) {
		for (int n = 0; n < pots.rows; n++)
			addNode(pots.row(n).t());
	}

	void CGraph::setNodes(size_t start_node, const Mat &pots) {
		// Assertions
		DGM_ASSERT_MSG(start_node + pots.rows <= getNumNodes(), "The given ranges exceed the number of nodes(%zu)", getNumNodes());

#ifdef ENABLE_PDP
		parallel_for_(Range(0, pots.rows), [start_node, &pots, this](const Range& range) {
#else
		const Range range(0, pots.rows);
#endif
		for (int n = range.start; n < range.end; n++)
			setNode(start_node + n, pots.row(n).t());
#ifdef ENABLE_PDP
		});
#endif
	}

	void CGraph::getNodes(size_t start_node, size_t num_nodes, Mat& pots) const {
		if (!num_nodes) num_nodes = getNumNodes() - start_node;

		// Assertions
		DGM_ASSERT_MSG(start_node + num_nodes <= getNumNodes(), "The given ranges exceed the number of nodes(%zu)", getNumNodes());

		if (pots.empty() || pots.cols != m_nStates || pots.rows != num_nodes)
			pots = Mat(static_cast<int>(num_nodes), m_nStates, CV_32FC1);
		
		transpose(pots, pots);

#ifdef ENABLE_PDP
		parallel_for_(Range(0, pots.cols), [start_node, &pots, this](const Range& range) {
#else
		const Range range(0, pots.cols);
#endif
		for (int n = range.start; n < range.end; n++)
			getNode(start_node + n, lvalue_cast(pots.col(n)));
#ifdef ENABLE_PDP
		});
#endif
		transpose(pots, pots);
	}
}
