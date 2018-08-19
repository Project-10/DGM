#include "Tests.h"
#include "DGM/parallel.h"
#include "DGM/random.h"

using namespace DirectGraphicalModels;

TEST_F(CTests, parallel_gemm) 
{
#if defined(ENABLE_PPL) && defined(ENABLE_AMP)	
	int width   = random::u<int>(10, 1000);
	int height  = random::u<int>(10, 1000);
	float alpha = random::U<float>(0.0f, 1.0f);
	float beta  = random::U<float>(0.0f, 1.0f);

	Mat A = random::U(Size(width, height),  CV_32FC1, 0.0, 100.0);
	Mat B = random::U(Size(height, width),  CV_32FC1, 0.0, 100.0);
	Mat C = random::U(Size(height, height), CV_32FC1, 0.0, 100.0);
	Mat ppl_res, amp_res;
	
	parallel::impl::ppl_gemm(A, B, alpha, C, beta, ppl_res);
	parallel::impl::amp_gemm(A, B, alpha, C, beta, amp_res);

	ASSERT_TRUE(std::equal(ppl_res.begin<float>(), ppl_res.end<float>(), amp_res.begin<float>()));
#endif
}

void testGraphBuilding(CGraph *pGraph, byte nStates)
{
	ASSERT_EQ(nStates, pGraph->getNumStates());
	
	int nNodes1 = random::u<int>(100, 100000);
	int nNodes2 = random::u<int>(50, nNodes1 - 50);
	
	Mat pots1 = random::U(Size(nStates, nNodes1), CV_32FC1, 0.0, 100.0);
	Mat pots2 = random::U(Size(nStates, nNodes2), CV_32FC1, 0.0, 100.0);

	pGraph->addNodes(pots1);
//	pGraph->setNodes(pots2, 50);

	Mat pot;
	for (int n = 0; n < nNodes1; n++) {
		pGraph->getNode(n, pot);
		for (byte s = 0; s < nStates; s++)
			if (n < 50)
				ASSERT_EQ(pot.at<float>(s, 0), pots1.at<float>(n, s));
		//		(pot.at<float>(s, 0) != pots1.at<float>(n, s)) return false;
	}

	//return true;
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