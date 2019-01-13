#include "TestGraph.h"
#include "DGM/random.h"

using namespace DirectGraphicalModels;

// ======================================== CGraph Building ========================================
void testGraphBuilding(CGraph& graph, byte nStates)
{
	ASSERT_EQ(nStates, graph.getNumStates());

	int nNodes = random::u<int>(100, 100000);
	Mat pots1 = random::U(Size(nStates, nNodes), CV_32FC1, 0.0, 100.0);

	ASSERT_EQ(0, graph.addNode());
	ASSERT_EQ(1, graph.addNode(pots1.row(1).t()));
	graph.addNodes(pots1);
	ASSERT_EQ(nNodes + 2, graph.getNumNodes());
	graph.setNode(0, pots1.row(0).t());

	Mat pot0, pot1, pot2;
	graph.getNode(0, pot0);
	graph.getNode(1, pot1);
	graph.getNodes(0, 10, pot2);

	float *pPot0 = pots1.ptr<float>(0);
	float *pPot1 = pots1.ptr<float>(1);
	for (byte s = 0; s < nStates; s++) {
		ASSERT_EQ(pot0.at<float>(s, 0), pPot0[s]);
		ASSERT_EQ(pot1.at<float>(s, 0), pPot1[s]);
		ASSERT_EQ(pot2.at<float>(0, s), pPot0[s]);
		ASSERT_EQ(pot2.at<float>(1, s), pPot1[s]);
	}

	Mat pots2 = random::U(Size(nStates, static_cast<int>(graph.getNumNodes()) - 10), CV_32FC1, 0.0, 100.0);
	graph.setNodes(2, pots2);
	Mat pot;
	for (size_t n = 2; n < graph.getNumNodes(); n++) {
		graph.getNode(n, pot);
		float *pPot = (static_cast<int>(n) - 2 < pots2.rows) ? pots2.ptr<float>(static_cast<int>(n) - 2) : pots1.ptr<float>(static_cast<int>(n) - 2);
		for (byte s = 0; s < nStates; s++)
			ASSERT_EQ(pot.at<float>(s, 0), pPot[s]);
	}

	graph.reset();
	ASSERT_EQ(graph.getNumEdges(), 0);
}

TEST_F(CTestGraph, CG_dense_building)
{
	const byte nStates = static_cast<byte>(random::u(10, 255));
	CGraphDense graph(nStates);
	testGraphBuilding(graph, nStates);
}

TEST_F(CTestGraph, CG_pairwise_building)
{
	const byte nStates = static_cast<byte>(random::u(10, 255));
	CGraphPairwise graph(nStates);
	testGraphBuilding(graph, nStates);
}


// ======================================== IGraphPairwise Building ========================================
void testGraphPairwiseBuilding(IGraphPairwise& graph, byte nStates)
{
	// Build a graph with edges
	size_t nNodes = random::u<size_t>(100, 1000);
	size_t nEdges = 0;
	for (size_t i = 0; i < nNodes; i++) graph.addNode();
	for (size_t i = 1; i < nNodes; i++)
		if (i <= nNodes / 2) {
			graph.addEdge(i - 1, i);
			nEdges += 1;
		}
		else {
			graph.addArc(i - 1, i);
			nEdges += 2;
		}
	
	ASSERT_EQ(nNodes, graph.getNumNodes());
	ASSERT_EQ(nEdges, graph.getNumEdges());

	// Test child and parent nodes
	vec_size_t vNodes;
	for (size_t i = 1; i < nNodes - 1; i++) {
		graph.getChildNodes(i, vNodes);
		ASSERT_TRUE(std::find(vNodes.begin(), vNodes.end(), i + 1) != vNodes.end());
		graph.getParentNodes(i, vNodes);
		ASSERT_TRUE(std::find(vNodes.begin(), vNodes.end(), i - 1) != vNodes.end());
	}

	// Test existance of edges and arcs
	size_t n = random::u<size_t>(10, nNodes / 2);
	ASSERT_TRUE(graph.isEdgeExists(n - 1, n));
	ASSERT_FALSE(graph.isArcExists(n - 1, n));
	ASSERT_FALSE(graph.isEdgeArc(n - 1, n));
	graph.addEdge(n, n - 1);
	ASSERT_TRUE(graph.isArcExists(n - 1, n));
	ASSERT_TRUE(graph.isEdgeArc(n - 1, n));
	graph.removeArc(n - 1, n);
	ASSERT_FALSE(graph.isEdgeExists(n - 1, n));

	n = random::u<size_t>(nNodes / 2 + 1, nNodes - 1);
	ASSERT_TRUE(graph.isEdgeExists(n - 1, n));
	ASSERT_TRUE(graph.isArcExists(n - 1, n));
	ASSERT_TRUE(graph.isEdgeArc(n - 1, n));
	graph.removeEdge(n, n - 1);
	ASSERT_TRUE(graph.isEdgeExists(n - 1, n));
	ASSERT_FALSE(graph.isArcExists(n - 1, n));
	ASSERT_FALSE(graph.isEdgeArc(n - 1, n));

	// Test edge groups
	ASSERT_EQ(0, graph.getEdgeGroup(n - 1, n));
	graph.setEdgeGroup(n - 1, n, 1);
	ASSERT_EQ(1, graph.getEdgeGroup(n - 1, n));
	graph.setArcGroup(n + 1, n, 2);
	ASSERT_EQ(2, graph.getEdgeGroup(n + 1, n));

	graph.setEdges(std::nullopt, Mat::zeros(nStates, nStates, CV_32FC1));
	graph.setEdges(1, Mat::ones(nStates, nStates, CV_32FC1) * 1);
	graph.setEdges(2, Mat::ones(nStates, nStates, CV_32FC1) * 2);

	Mat pot;
	graph.getEdge(n - 1, n, pot);
	ASSERT_EQ(nStates, pot.cols);
	ASSERT_EQ(nStates, pot.rows);
	ASSERT_EQ(1, pot.at<float>(0, 0));
	graph.getEdge(n + 1, n, pot);
	ASSERT_EQ(nStates, pot.cols);
	ASSERT_EQ(nStates, pot.rows);
	ASSERT_EQ(2, pot.at<float>(0, 0));

	Mat pot_in = random::U(Size(nStates, nStates), CV_32FC1, 0.0, 100.0);
	graph.setArc(n, n + 1, pot_in);
	Mat pot_out;
	graph.getEdge(n, n + 1, pot_out);
	ASSERT_EQ(pot_in.cols, pot_out.cols);
	ASSERT_EQ(pot_in.rows, pot_out.rows);
	for (int y = 0; y < nStates; y++) {
		float* pIn = pot_in.ptr<float>(y);
		float* pOut = pot_out.ptr<float>(y);
		for (int x = 0; x < nStates; x++) 
			ASSERT_EQ(sqrtf(pIn[x]), pOut[x]);
	}

	// graph.marginalize(const vec_size_t &nodes);
	// graph.setEdge(size_t srcNode, size_t dstNode, const Mat &pot);
}

TEST_F(CTestGraph, IGP_pairwise_building)
{
	const byte nStates = static_cast<byte>(random::u(10, 255));
	CGraphPairwise graph(nStates);
	testGraphPairwiseBuilding(graph, nStates);
}

TEST_F(CTestGraph, IGP_weiss_building)
{
	const byte nStates = static_cast<byte>(random::u(10, 255));
	CGraphWeiss graph(nStates);
	testGraphPairwiseBuilding(graph, nStates);
}
 

// ======================================== Graph Extensions ========================================
void testGraphExtension(CGraphExt& graphExt, CGraph& graph)
{
	const byte nStates = graph.getNumStates();
	
	Size graphSize = Size(random::u<int>(100, 1000), random::u<int>(100, 1000));
	graphExt.buildGraph(graphSize);
	ASSERT_EQ(graphSize, graphExt.getSize());
	ASSERT_EQ(graphSize.width * graphSize.height, graph.getNumNodes());

	graphSize = Size(random::u<int>(100, 1000), random::u<int>(100, 1000));
	Mat pots = random::U(graphSize, CV_32FC(nStates));
	graphExt.setGraph(pots);
	ASSERT_EQ(graphSize, graphExt.getSize());
	
	Mat test_pots;
	graph.getNodes(0, 0, test_pots);
	test_pots = test_pots.clone().reshape(graph.getNumStates(), graphSize.height);
	for (int y = 0; y < pots.rows; y++) {
		float *pPots		= pots.ptr<float>(y);
		float *pTestPots	= test_pots.ptr<float>(y);
		for (int x = 0; x < pots.cols; x++) 
			for (int c = 0; c < pots.channels(); c++)
				ASSERT_EQ(pPots[x * nStates + c], pTestPots[x * nStates + c]);
	}
	
//	void addDefaultEdgesModel(float val, float weight = 1.0f);
//	void addDefaultEdgesModel(const Mat &featureVectors, float val, float weight);
//	void addDefaultEdgesModel(const vec_mat_t &featureVectors, float val, float weight);
}

TEST_F(CTestGraph, CG_dense_extension)
{
	const byte nStates = static_cast<byte>(random::u(10, 255));
	CGraphDense	graph(nStates);
	CGraphDenseExt graphExt(graph);
	testGraphExtension(graphExt, graph);
}

TEST_F(CTestGraph, CG_pairwise_extension)
{
	const byte nStates = static_cast<byte>(random::u(10, 255));
	CGraphPairwise graph(nStates);
	CGraphPairwiseExt graphExt(graph);
	testGraphExtension(graphExt, graph);
}