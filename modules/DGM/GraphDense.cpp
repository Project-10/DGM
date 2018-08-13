#include "GraphDense.h"

namespace DirectGraphicalModels 
{
	// Add a new node to the graph
	size_t	CGraphDense::addNode(void)
	{
		float pot = 1.0f / m_nStates;
		for (byte i = 0; i < m_nStates; i++)
			m_vNodePotentials.push_back(pot);
		return getNumNodes();
	}
	
	size_t	CGraphDense::addNode(const Mat &pot)
	{
		return 0;
	}

	void	CGraphDense::setNode(size_t node, const Mat &pot)
	{

	}

	void	CGraphDense::getNode(size_t node, Mat &pot) const
	{
	
	}
}