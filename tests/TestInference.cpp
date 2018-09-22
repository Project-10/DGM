#include "TestInference.h"

// Constructor
CTestInference::CTestInference(void)
{
	m_pGraph = std::make_unique<CGraphPairwise>(m_nStates);
	// Building the graph
	for (size_t i = 0; i < m_nNodes; i++)		m_pGraph->addNode();
	for (size_t i = 0; i < m_nNodes - 1; i++)	m_pGraph->addArc(i, i + 1);

	fillGraph();
	CInferExact exactInferer(*m_pGraph.get());
	exactInferer.infer();
	m_vPotExact = exactInferer.getPotentials(0);
}

void CTestInference::fillGraph(void)
{
	Mat nodePot(m_nStates, 1, CV_32FC1);			// node Potential (column-vector)
	Mat edgePot(m_nStates, m_nStates, CV_32FC1);	// edge Potential (matrix)

	// Setting the node potentials
	for (size_t i = 0; i < m_nNodes; i++) {
		if (i % 2) {							// for odd nodes
			nodePot.at<float>(0, 0) = 0.10f;	// nPot = (0.10; 0.90)^T
			nodePot.at<float>(1, 0) = 0.90f;
		}
		else {									// for even nodes
			nodePot.at<float>(0, 0) = 0.75f;	// nPot = (0.75; 0.25)^T
			nodePot.at<float>(1, 0) = 0.25f;
		}
		m_pGraph->setNode(i, nodePot);
	}

	// Defying the edge potential matrix
	edgePot.at<float>(0, 0) = 2.0f;	edgePot.at<float>(0, 1) = 1.0f;
	edgePot.at<float>(1, 0) = 1.0f;	edgePot.at<float>(1, 1) = 2.0f;

	// Setting the edges potentials
	for (size_t i = 0; i < m_nNodes - 1; i++)
		m_pGraph->setArc(i, i + 1, edgePot);
}

void CTestInference::testInferer(CInfer &inferer)
{
	fillGraph();
	inferer.infer(100);
	vec_float_t pot = inferer.getPotentials(0);
	
	ASSERT_EQ(pot.size(), m_vPotExact.size());
	for (size_t i = 0; i < pot.size(); i++) 
		ASSERT_LT(fabs(pot[i] - m_vPotExact[i]), 1e-5);
}

TEST_F(CTestInference, inference_chain)
{
	CInferChain inferer(*m_pGraph.get());
	testInferer(inferer);
}

TEST_F(CTestInference, inference_tree)
{
	CInferTree inferer(*m_pGraph.get());
	testInferer(inferer);
}

TEST_F(CTestInference, inference_LBP)
{
	CInferLBP inferer(*m_pGraph.get());
	testInferer(inferer);
}
