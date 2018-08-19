#include "Graph.h"

namespace DirectGraphicalModels 
{
	void	CGraph::addNodes(const Mat &pots) {
		for (int n = 0; n < pots.rows; n++)
			addNode(pots.row(n).t());
	}

	void	CGraph::setNodes(const Mat &pots, size_t start_node) {
#ifdef ENABLE_PPL
        size_t size = pots.rows;
        size_t rangeSize = size / (GetProcessorCount() * 10);
        rangeSize = max(1, rangeSize);
        
        parallel_for(0u, size, rangeSize, [start_node, size, rangeSize] (size_t i) {
            for (size_t j = 0; (j < rangeSize) && (i + j < size); j++)
                setNode(start_node + i + j, pots.row(i + j).t());
        });
#else
        for (int n = 0; n < pots.rows; n++)
            setNode(start_node + n, pots.row(n).t());
	}
#endif
}
