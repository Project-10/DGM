#include "TestInference.h"

void buildGraph(IGraphPairwise& graph, size_t nNodes)
{
	if (graph.getNumNodes()) graph.reset();
	for (size_t i = 0; i < nNodes; i++)		graph.addNode();
	for (size_t i = 0; i < nNodes - 1; i++)	graph.addArc(i, i + 1);
}

void fillGraph(IGraphPairwise& graph)
	{
		const byte	nStates = graph.getNumStates();
		const size_t nNodes = graph.getNumNodes();
		
		Mat nodePot(nStates, 1, CV_32FC1);			// node Potential (column-vector)
		Mat edgePot(nStates, nStates, CV_32FC1);	// edge Potential (matrix)

		// Setting the node potentials
		for (size_t i = 0; i < nNodes; i++) {
			if (i % 2) {							// for odd nodes
				nodePot.at<float>(0, 0) = 0.10f;	// nPot = (0.10; 0.90)^T
				nodePot.at<float>(1, 0) = 0.90f;
			}
			else {									// for even nodes
				nodePot.at<float>(0, 0) = 0.75f;	// nPot = (0.75; 0.25)^T
				nodePot.at<float>(1, 0) = 0.25f;
			}
			graph.setNode(i, nodePot);
		}

		// Defying the edge potential matrix
		edgePot.at<float>(0, 0) = 2.0f;	edgePot.at<float>(0, 1) = 1.0f;
		edgePot.at<float>(1, 0) = 1.0f;	edgePot.at<float>(1, 1) = 2.0f;

		// Setting the edges potentials
		sqrt(edgePot, edgePot);
		graph.setEdges(std::nullopt, edgePot);
	}

// Constructor
CTestInference::CTestInference(void)
{
	CGraphPairwise graph(m_nStates);
	buildGraph(graph, m_nNodes);
	fillGraph(graph);

	CInferExact exactInferer(graph);
	exactInferer.infer();
	m_vPotExact = exactInferer.getPotentials(0);
}


void CTestInference::testInferer(CInfer &inferer)
{
	inferer.infer(100);
	vec_float_t pot = inferer.getPotentials(0);
	
	ASSERT_EQ(pot.size(), m_vPotExact.size());
	for (size_t i = 0; i < pot.size(); i++) 
		ASSERT_LT(fabs(pot[i] - m_vPotExact[i]), 1e-5);
}

TEST_F(CTestInference, inference_chain)
{
	CGraphPairwise graph(m_nStates);
	buildGraph(graph, m_nNodes);
	fillGraph(graph);

	CInferChain inferer(graph);
	testInferer(inferer);
}

TEST_F(CTestInference, inference_tree)
{
	CGraphPairwise graph(m_nStates);
	buildGraph(graph, m_nNodes);
	fillGraph(graph);

	CInferTree inferer(graph);
	testInferer(inferer);
}

TEST_F(CTestInference, inference_LBP)
{
	CGraphPairwise graph(m_nStates);
	buildGraph(graph, m_nNodes);
	fillGraph(graph);
	
	CInferLBP inferer(graph);
	testInferer(inferer);
}

TEST_F(CTestInference, inference_exact_weiss)
{
	CGraphWeiss graph(m_nStates);
	buildGraph(graph, m_nNodes);
	fillGraph(graph);
	
	CInferExact inferer(graph);
	testInferer(inferer);
}
