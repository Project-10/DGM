#include "exact.h"

const byte   nStates = 2;						// {false, true}
const size_t nNodes  = 4;						// four students

void CExact::fillGraph(CGraphPairwise &graph)
{
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
	for (size_t i = 0; i < nNodes - 1; i++)
		graph.setArc(i, i + 1, edgePot);
}

void CExact::printMarginals(const CGraphPairwise &graph, const std::string &str)
{
	Mat		 nodePot;
	printf("%s:\t", str.c_str());
	for (int i = 0; i < nNodes; i++) {
		graph.getNode(i, nodePot);
		printf((i < nNodes - 1) ? " %.2f\t%.2f\t|" : " %.2f\t%.2f\n", nodePot.at<float>(0, 0), nodePot.at<float>(1, 0));
	}
}

void CExact::Main(void)
{
	size_t			i;
	CGraphPairwise	graph(nStates);
	CDecodeExact	decoderExcact(graph);
	CInferExact		infererExact(graph);
	CInferChain		infererChain(graph);
	CInferTree		infererTree(graph);
	CInferLBP		infererLBP(graph);
	CInferViterbi	infererViterbi(graph);

	// Building the graph
	for (i = 0; i < nNodes; i++)		graph.addNode();
	for (i = 0; i < nNodes - 1; i++)	graph.addArc(i, i + 1);

	printf("\t\t\t\tINFERENCE\n\t");
	for (i = 0; i < nNodes - 1; i++) printf("   Node %zd\t|", i); printf("   Node %zd\n\t", i);
	for (i = 0; i < nNodes - 1; i++) printf(" False\tTrue\t|"); printf(" False\tTrue\n");
	printf("--------------------------------------------------------------------\n");

	fillGraph(graph);												// Without training, filling the graph with pre-defined potentials
	printMarginals(graph, "Init");									// Initial potentials

	vec_byte_t decoding_decoderExcact = decoderExcact.decode();	    // Exact decoding from decode

	infererExact.infer();											// Exact inference
	vec_byte_t decoding_infererExcact = infererExact.decode();		// Exact decoding from inferer
	printMarginals(graph, "Exact");

	fillGraph(graph);
	infererChain.infer();											// Chain inference
	vec_byte_t decoding_infererChain = infererChain.decode();		// Chain decoding
	printMarginals(graph, "Chain");

	fillGraph(graph);
	infererTree.infer();											// Tree inference
	vec_byte_t decoding_infererTree = infererTree.decode();		    // Tree decoding
	printMarginals(graph, "Tree");

	fillGraph(graph);
	infererLBP.infer(10);											// Loopy Belief Probagation inference
	vec_byte_t decoding_infererLBP = infererLBP.decode();			// Loopy Belief Probagation decoding
	printMarginals(graph, "LBP");

	fillGraph(graph);
	infererViterbi.infer(10);										// Viterbi inference
	vec_byte_t decoding_infererViterbi = infererViterbi.decode();	// Viterbi decoding
	printMarginals(graph, "Vtrbi");

	printf("\n\t\t\t\tDECODING\n\t\t");
	for (i = 0; i < nNodes - 1; i++) printf("Node%zd\t", i); printf("Node%zd\n", i);
	printf("---------------------------------------------\n");
	printf("Exact decoder:\t");	for (i = 0; i < nNodes; i++) printf(decoding_decoderExcact[i]  ? "true\t" : "false\t");	printf("\n");
	printf("Exact inferer:\t");	for (i = 0; i < nNodes; i++) printf(decoding_infererExcact[i]  ? "true\t" : "false\t");	printf("\n");
	printf("LBP inferer:\t");	for (i = 0; i < nNodes; i++) printf(decoding_infererLBP[i]     ? "true\t" : "false\t");	printf("\n");
	printf("Vtrbi inferer:\t");	for (i = 0; i < nNodes; i++) printf(decoding_infererViterbi[i] ? "true\t" : "false\t"); printf("\n");
}
