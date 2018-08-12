#include "GraphBoost.h"

namespace DirectGraphicalModels 
{
	CGraphBoost::CGraphBoost(byte nStates) : IGraphTemp(nStates)
	{
	}

	CGraphBoost::~CGraphBoost(void)
	{
	}

	size_t CGraphBoost::addNode(const Mat &pot) 
	{ 
		return 0; 
	};

	void CGraphBoost::addEdge(size_t srcNode, size_t dstNode, const Mat &pot) 
	{
	}
}