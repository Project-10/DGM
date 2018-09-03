#include "chain.h"

const byte	 nStates = 7;		// {grad scool, industry, video games, industry (with PhD), academia, video games (with PhD), deceased}
const size_t nNodes = 60;		// sixty years

Mat CChain::getNodePot(void)
{
	Mat res(nStates, 1, CV_32FC1);			// node Potential (column-vector)
	res.at<float>(0, 0) = 0.3f;				// grad scool 
	res.at<float>(1, 0) = 0.6f;				// industry
	res.at<float>(2, 0) = 0.1f;				// video games
	res.at<float>(3, 0) = 0.0f;				// industry (with PhD)
	res.at<float>(4, 0) = 0.0f;				// academia
	res.at<float>(5, 0) = 0.0f;				// video games (with PhD)
	res.at<float>(6, 0) = 0.0f;				// deceased
	return res;
}

Mat CChain::getEdgePot(void)
{
	Mat res(nStates, nStates, CV_32FC1);	// edge Potential
	res.at<float>(0, 0) = 0.08f;	res.at<float>(0, 1) = 0.9f;		res.at<float>(0, 2) = 0.01f;	res.at<float>(0, 3) = 0.0f;		res.at<float>(0, 4) = 0.0f;		res.at<float>(0, 5) = 0.0f;		res.at<float>(0, 6) = 0.01f;
	res.at<float>(1, 0) = 0.03f;	res.at<float>(1, 1) = 0.95f;	res.at<float>(1, 2) = 0.01f;	res.at<float>(1, 3) = 0.0f;		res.at<float>(1, 4) = 0.0f;		res.at<float>(1, 5) = 0.0f;		res.at<float>(1, 6) = 0.01f;
	res.at<float>(2, 0) = 0.06f;	res.at<float>(2, 1) = 0.06f;	res.at<float>(2, 2) = 0.75f;	res.at<float>(2, 3) = 0.05f;	res.at<float>(2, 4) = 0.05f;	res.at<float>(2, 5) = 0.02f;	res.at<float>(2, 6) = 0.01f;
	res.at<float>(3, 0) = 0.0f;		res.at<float>(3, 1) = 0.0f;		res.at<float>(3, 2) = 0.0f;		res.at<float>(3, 3) = 0.3f;		res.at<float>(3, 4) = 0.6f;		res.at<float>(3, 5) = 0.09f;	res.at<float>(3, 6) = 0.01f;
	res.at<float>(4, 0) = 0.0f;		res.at<float>(4, 1) = 0.0f;		res.at<float>(4, 2) = 0.0f;		res.at<float>(4, 3) = 0.02f;	res.at<float>(4, 4) = 0.95f;	res.at<float>(4, 5) = 0.02f;	res.at<float>(4, 6) = 0.01f;
	res.at<float>(5, 0) = 0.0f;		res.at<float>(5, 1) = 0.0f;		res.at<float>(5, 2) = 0.0f;		res.at<float>(5, 3) = 0.01f;	res.at<float>(5, 4) = 0.01f;	res.at<float>(5, 5) = 0.97f;	res.at<float>(5, 6) = 0.01f;
	res.at<float>(6, 0) = 0.0f;		res.at<float>(6, 1) = 0.0f;		res.at<float>(6, 2) = 0.0f;		res.at<float>(6, 3) = 0.0f;		res.at<float>(6, 4) = 0.0f;		res.at<float>(6, 5) = 0.0f;		res.at<float>(6, 6) = 1.0f;
	return res;
}

void CChain::Main(void)
{
	size_t i;

	CGraphPairwise	graph(nStates);
	CInferChain		inferer(graph);

	Mat nodePot = getNodePot();
	graph.addNode(nodePot);				// add the first node
	nodePot.setTo(1.0f / nStates);			// uniform distribution
	for (i = 1; i < nNodes; i++)
		graph.addNode(nodePot);			// add nodes

	Mat edgePot = getEdgePot();
	for (i = 0; i < nNodes - 1; i++)
		graph.addArc(i, i + 1, edgePot);	// add arcs

	// Inference
	inferer.infer();

	// Print Out Results
	printf("Node\t"); for (byte s = 0; s < nStates; s++) printf("State %d\t", s); printf("\n");
	printf("---------------------------------------------------------------\n");
	for (i = 0; i < nNodes; i++) {
		printf("%zd \t", i);
		graph.getNode(i, nodePot);
		printf("%.4f", nodePot.at<float>(0, 0));  for (byte s = 1; s < nStates; s++) printf("\t%.4f", nodePot.at<float>(s, 0)); printf("\n");
	}
}