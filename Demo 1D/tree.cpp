#include "tree.h"

const byte   nStates  = 4;					// {very safe, safe, unsafe, very unsave}
const size_t nNodes	  = 100;				// number of nodes 
const size_t nSources = 1;					// number of sources

Mat CTree::getNodePot(void)
{
	Mat res(nStates, 1, CV_32FC1);			// node Potential (column-vector)
	res.at<float>(0, 0) = 0.9f;				// very safe
	res.at<float>(1, 0) = 0.09f;			// safe
	res.at<float>(2, 0) = 0.009f;			// unsafe
	res.at<float>(3, 0) = 0.001f;			// very unsave
	return res;
}

Mat CTree::getEdgePot(void)
{
	Mat res(nStates, nStates, CV_32FC1);	// edge Potential
	res.at<float>(0, 0) = 0.9890f;	res.at<float>(0, 1) = 0.0099f;	res.at<float>(0, 2) = 0.0010f;	res.at<float>(0, 3) = 0.0001f;		
	res.at<float>(1, 0) = 0.1309f;	res.at<float>(1, 1) = 0.8618f;	res.at<float>(1, 2) = 0.0066f;	res.at<float>(1, 3) = 0.0007f;		
	res.at<float>(2, 0) = 0.0420f;	res.at<float>(2, 1) = 0.0841f;	res.at<float>(2, 2) = 0.8682f;	res.at<float>(2, 3) = 0.0057f;	
	res.at<float>(3, 0) = 0.0667f;	res.at<float>(3, 1) = 0.0333f;	res.at<float>(3, 2) = 0.1667f;	res.at<float>(3, 3) = 0.7333f;		
	return res;
}

CGraph * CTree::buildTree()
{
	CGraph *graph = new CGraph(nStates);
	
	// Build a complete graph
	// Add nodes with default potentials
	Mat nPot(nStates, 1, CV_32FC1, 1.0f / nStates);
	for (size_t n = 0; n < nNodes; n++)	graph->addNode(nPot);

	// Create weighted edges with random weights
	std::vector<std::pair<Edge, float>> edges; 
	for (size_t n1 = 0; n1 < nNodes; n1++) 
		for (size_t n2 = n1 + 1; n2 < nNodes; n2++) 
			edges.push_back(std::make_pair(Edge(n1, n2), static_cast<float>(rand()) / RAND_MAX));

	// Sort these edges by weight
	std::sort(edges.begin(), edges.end(), [](std::pair<Edge, float> &left, std::pair<Edge, float> &right) { return left.second < right.second; });
	
	Mat edgePot;													// Default symmetric edge potentials
	addWeighted(getEdgePot(), 0.5, getEdgePot().t(), 0.5, 0.0, edgePot);

	// Build a minimum spanning tree upon the complete graph
	std::vector<bool> N(nNodes, false);							// Accounted nodes
	N[0] = true;												// Start from the first node
	while (std::find(N.begin(), N.end(), false) != N.end()) {	// while there is at least one non-accounted node
		std::vector<std::pair<Edge, float>>::iterator it;		// Find an edge with minimal weight, such that one node is accounted and the second is not
		while ((it = std::find_if(edges.begin(), edges.end(), [&](std::pair<Edge, float> edge) { return N[edge.first.node1] ^ N[edge.first.node2]; })) != edges.end()) {
			size_t n1 = it->first.node1;
			size_t n2 = it->first.node2;
			graph->addArk(n1, n2, edgePot);						// Add an arc to the tree
			N[n1] = N[n2] = true;								// Now both nodes are accounted
		}
	} 

	return graph;
}

void CTree::Main(void)
{
	srand(0);
	CGraph	*graph = buildTree();					// Returns a tree with default potentials
	CInfer	*inferer = new CInferTree(graph);


	std::vector<size_t>		 vParents, vChilds;
	std::vector<size_t>		 sources;
	std::deque<size_t>		 sourceQueue;			// Queue with indexes of the source nodes
	std::vector<std::string> labels(nNodes);
	

	// Separate all nodes into Source, Internal and Tap nodes
	for (size_t n = 0; n < nNodes; n++) {
		graph->getParentNodes(n, vParents);
		if (vParents.size() <= 1) {					// if the node is a leaf
			if (sources.size() < nSources) {
				labels[n] = "Source";				// => it is either a source
				sources.push_back(n);
			} else 
				labels[n] = "Tap";					// => or a tap
		} else 
			labels[n] = "I";						// otherwise it is an internal node
	} // n

	// Setting the node and edge potentials in the tree
	// Nodes
	Mat nodePot = getNodePot();
	std::for_each(sources.begin(), sources.end(), [&](size_t n) { graph->setNode(n, nodePot); });

	// Edges
	std::vector<bool> ifSource(nNodes, false); 
	std::for_each(sources.begin(), sources.end(), [&](size_t n) { 
		ifSource[n] = true; 
		sourceQueue.push_back(n);
	});	

	Mat edgePot = getEdgePot();
	while (!sourceQueue.empty()) {
		size_t n1 = sourceQueue.front();			// pop the front index of a source node
		sourceQueue.pop_front();
		graph->getChildNodes(n1, vChilds);
		std::for_each(vChilds.begin(), vChilds.end(), [&, n1](size_t n2) {
			if (!ifSource[n2]) {					// if the connected node is not a source
				graph->setArk(n1, n2, edgePot);		// set the potential,
				ifSource[n2] = true;				// mark it as a source
				sourceQueue.push_back(n2);			// and add it to the queue
			}
		});
	}

	inferer->infer();

	// Print Out Results
	printf("Node\tType\t"); for (byte s = 0; s < nStates; s++) printf("State %d\t", s); printf("\n");
	printf("-----------------------------------------------\n");
	for (size_t n = 0; n < nNodes; n++) {
		printf("%zd\t%s\t", n, labels[n].c_str());
		graph->getNode(n, nodePot);
		printf("%.4f", nodePot.at<float>(0, 0));  for (byte s = 1; s < nStates; s++) printf("\t%.4f", nodePot.at<float>(s, 0)); printf("\n");
	}
	

	if (true) {
		Mat img = CMarkerGraph::drawGraph(graph, [](size_t n, int size) { 
			return cvPoint(
				size / 2 + static_cast<int>(0.45 * size * cos(2 * n * Pi / nNodes)),
				size / 2 + static_cast<int>(0.45 * size * sin(2 * n * Pi / nNodes)) );
		});

		imshow("Image", img);
		cvWaitKey();
		destroyAllWindows();
	}

	delete graph;
	delete inferer;
}
