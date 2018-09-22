#include "TestGraph.h"
#include "DGM/random.h"

using namespace DirectGraphicalModels;

void testGraphBuilding(CGraph *pGraph, byte nStates)
{
	ASSERT_EQ(nStates, pGraph->getNumStates());

	int nNodes = random::u<int>(100, 100000);
	Mat pots1 = random::U(Size(nStates, nNodes), CV_32FC1, 0.0, 100.0);

	ASSERT_EQ(0, pGraph->addNode());
	ASSERT_EQ(1, pGraph->addNode(pots1.row(1).t()));
	pGraph->addNodes(pots1);
	ASSERT_EQ(nNodes + 2, pGraph->getNumNodes());
	pGraph->setNode(0, pots1.row(0).t());

	Mat pot0, pot1;
	pGraph->getNode(0, pot0);
	pGraph->getNode(1, pot1);

	float *pPot0 = pots1.ptr<float>(0);
	float *pPot1 = pots1.ptr<float>(1);
	for (byte s = 0; s < nStates; s++) {
		ASSERT_EQ(pot0.at<float>(s, 0), pPot0[s]);
		ASSERT_EQ(pot1.at<float>(s, 0), pPot1[s]);
	}

	Mat pots2 = random::U(Size(nStates, pGraph->getNumNodes() - 10), CV_32FC1, 0.0, 100.0);
	pGraph->setNodes(pots2, 2);
	Mat pot;
	for (size_t n = 2; n < pGraph->getNumNodes(); n++) {
		pGraph->getNode(n, pot);
		float *pPot = (n - 2 < pots2.rows) ? pots2.ptr<float>(n - 2) : pots1.ptr<float>(n - 2);
		for (byte s = 0; s < nStates; s++)
			ASSERT_EQ(pot.at<float>(s, 0), pPot[s]);
	}

	pGraph->reset();
	ASSERT_EQ(pGraph->getNumEdges(), 0);
}

TEST_F(CTestGraph, graph_dense_building)
{
	const byte nStates = static_cast<byte>(random::u(10, 255));
	CGraph *pGraph = new CGraphDense(nStates);
	testGraphBuilding(pGraph, nStates);
}

TEST_F(CTestGraph, graph_pairwise_building)
{
	const byte nStates = static_cast<byte>(random::u(10, 255));
	CGraph *pGraph = new CGraphPairwise(nStates);
	testGraphBuilding(pGraph, nStates);
}