#include "GraphKit.h"
#include "GraphDenseKit.h"
#include "GraphPairwiseKit.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
	std::shared_ptr<CGraphKit> CGraphKit::create(GraphType graphType, byte nStates) 
	{
		switch (graphType)
		{
		case DirectGraphicalModels::GraphType::pairwise:
			return std::make_shared<CGraphPairwiseKit>(nStates);
		case DirectGraphicalModels::GraphType::dense:
			return std::make_shared<CGraphDenseKit>(nStates);
		default:
			DGM_ASSERT_MSG(false, "Unknown graph type");
		}
	}
}