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

#ifdef ENABLE_PPL
		int size = pots.rows;
		int rangeSize = size / (concurrency::GetProcessorCount() * 10);
		rangeSize = MAX(1, rangeSize);
		//printf("Processors: %d\n", concurrency::GetProcessorCount());
		concurrency::parallel_for(0, size, rangeSize, [start_node, size, rangeSize, &pots, this](int i) {
			for (int j = 0; (j < rangeSize) && (i + j < size); j++)
				setNode(start_node + i + j, pots.row(i + j).t());
		});
#else
		for (int n = 0; n < pots.rows; n++)
			setNode(start_node + n, pots.row(n).t());
#endif
	}

	void CGraph::getNodes(size_t start_node, size_t num_nodes, Mat& pots) const {
		if (!num_nodes) num_nodes = getNumNodes() - start_node;

		// Assertions
		DGM_ASSERT_MSG(start_node + num_nodes <= getNumNodes(), "The given ranges exceed the number of nodes(%zu)", getNumNodes());

		if (pots.empty() || pots.cols != m_nStates || pots.rows != num_nodes)
			pots = Mat(static_cast<int>(num_nodes), m_nStates, CV_32FC1);
		
		transpose(pots, pots);

#ifdef ENABLE_PPL
		int size = pots.cols;
		int rangeSize = size / (concurrency::GetProcessorCount() * 10);
		rangeSize = MAX(1, rangeSize);
		//printf("Processors: %d\n", concurrency::GetProcessorCount());
		concurrency::parallel_for(0, size, rangeSize, [start_node, size, rangeSize, &pots, this](int i) {
			Mat pot;
			for (int j = 0; (j < rangeSize) && (i + j < size); j++) 
				getNode(start_node + i + j, lvalue_cast(pots.col(i + j)));
		});
#else
		for (int n = 0; n < pots.cols; n++)
			getNode(start_node + n, lvalue_cast(pots.col(n)));
#endif
		transpose(pots, pots);
	}
}
