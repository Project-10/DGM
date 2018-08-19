#include "Graph.h"

namespace DirectGraphicalModels 
{
	void	CGraph::addNodes(const Mat &pots) {
		for (int n = 0; n < pots.rows; n++)
			addNode(pots.row(n).t());
	}

	void	CGraph::setNodes(const Mat &pots, size_t start_node) {
#ifdef ENABLE_PPL
		int size = pots.rows;
		int rangeSize = size / (concurrency::GetProcessorCount() * 10);
		rangeSize = MAX(1, rangeSize);

		printf("Processors: %d\n", concurrency::GetProcessorCount());

		concurrency::parallel_for(0, size, rangeSize, [start_node, size, rangeSize, &pots, this](int i) {
			for (int j = 0; (j < rangeSize) && (i + j < size); j++)
				setNode(start_node + i + j, pots.row(i + j).t());
		});
#else
		for (int n = 0; n < pots.rows; n++)
			setNode(start_node + n, pots.row(n).t());
#endif
	}
}
