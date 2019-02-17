// Example "TREE" 1D-case with exact tree inference
// (https://www.cs.ubc.ca/~schmidtm/Software/UGM/tree.html)
/// @cond DemoCode
#pragma once

#include "Demo 1D.h"

class CTree : public CDemo1D
{
public:
	virtual void Main(void);


private:
	Mat		getNodePot(void);
	Mat		getEdgePot(void);
    void    buildTree(CGraphPairwise &graph);
};
/// @endcond

/**
@page demo1d_tree Tree
In the @ref demo1d_chain demo we considered chain-structured graph, one of the simplest types of dependency where we 
could take advantage of the graphical structure to allow efficient inference. In this demo, we consider more complex 
problem involving tree-structured graphical model. In particular, we consider the case where the arcs in the graph 
can be arbitrary, as long as there is one, and only one, path between any pair of nodes. For such graphs we can still
perform efficient inference by applying generalizations of the methods designed for chain-structured models. This 
example copies the idea from the 
<a href="https://www.cs.ubc.ca/~schmidtm/Software/UGM/tree.html" target="_blank">Water Turbidity Problem</a>

> In order to run this demo, please execute \b "Demo 1D.exe" \b tree command

Building The Tree
-----------------
We build a tree, consisting of 100 nodes with each having 4 states. Function \b buildTree() returns a random 
tree-structured graph, filled with initial node and edge potentials. The tree is build such that there is one, and 
only one, path between any pair of nodes and therefore does not have loops. Nodes with only arc is called a \a leaf. 
Function \b srand(0) assures that the graph structure remains the same during multiple execution of the demo:
@code
	using namespace DirectGraphicalModels;

	const byte   nStates  = 4;						// {very safe, safe, unsafe, very unsave}
	const size_t nNodes	  = 100;					// number of nodes
	const size_t nSources = 1;						// number of sources

	srand(0);
    CGraphPairwise graph(nStates);
    buildTree(graph);					            // Returns a tree with default potentials
@endcode

Next, we separate all the nodes into 3 types and assigne them the following labels: 
- \b Source: fist \b nSources leaf nodes,
- \b Tap: remaing leaf nodes and 
- \b I (\a internal \a structure): all other nodes. 

@code
	std::vector<size_t>		 vParents;
	std::vector<size_t>		 sources;
	std::vector<std::string> labels(nNodes);
	
	for (size_t n = 0; n < nNodes; n++) {
		graph.getParentNodes(n, vParents);
		if (vParents.size() <= 1) {					// if the node is a leaf
			if (sources.size() < nSources) {
				labels[n] = "Source";				// => it is either a source
				sources.push_back(n);
			} else
				labels[n] = "Tap";					// => or a tap
		} else
			labels[n] = "I";						// otherwise it is an internal node
	} // n
@endcode

After the tree was created, its nodes had potentials \f$(\frac{1}{4}, \frac{1}{4}, \frac{1}{4}, \frac{1}{4})^\top\f$.
Now, we substitute the potentials of the source nodes with the potential \f$(0.9, 0.09, 0.009, 0.001)^\top\f$, 
returned with function \b getNodePot():
@code
	Mat nodePot = getNodePot();

	for(size_t n: sources)  
		graph.setNode(n, nodePot); 
@endcode

The edge potential matrix is represented here as a transition matrix:
<table>
<caption>edgePot</caption>
<tr><th>from\\to<th>st.0  <th>st.1  <th>st.2  <th>st.3
<tr><th>st.0    <td>0.9890<td>0.0099<td>0.0010<td>0.0001
<tr><th>st.1    <td>0.1309<td>0.8618<td>0.0066<td>0.0007
<tr><th>st.2    <td>0.0420<td>0.0841<td>0.8682<td>0.0057
<tr><th>st.3    <td>0.0667<td>0.0333<td>0.1667<td>0.7333
</table>
\a e.g. a probability of switching from state 3 to state 1 is 0.0333. The edge potential is returned with function 
\b getEdgePot(). 

After the tree was created, its edges had symmetric potentials: 
\f$\frac{1}{2}edgePot^\top+\frac{1}{2}edgePot\f$. Now, we have to re-initialize the edge potentials with the 
original nonsymmetrical \f$edgePot\f$: set them in the direction from the source nodes to the tap nodes
DirectGraphicalModels::CGraphPairwise::setArc(size_t src, size_t dst, const Mat& edgePot):
@code
	std::vector<size_t> vChilds;
	std::vector<bool>	ifSource(nNodes, false);
	std::deque<size_t>	sourceQueue;				// queue with indexes of the source nodes

	for(size_t n: sources) {
		ifSource[n] = true;
		sourceQueue.push_back(n);
	}

	Mat edgePot = getEdgePot();
	while (!sourceQueue.empty()) {
		size_t n1 = sourceQueue.front();			// pop the front index of a source node
		sourceQueue.pop_front();
		graph.getChildNodes(n1, vChilds);
		for(size_t n2: vChilds) {
			if (!ifSource[n2]) {					// if the connected node is not a source
				graph.setArc(n1, n2, edgePot);		// set the potential,
				ifSource[n2] = true;				// mark it as a source
				sourceQueue.push_back(n2);			// and add it to the queue
			}
		}
	}
@endcode

Inference
---------
The number of all possible configurations for our tree is equal to \f$4^{100}\approx 1.61\times 10^{60}\f$, and it is not feasible to calculate joint 
probabilities for all of these configurations. Nevertheless, DGM has an efficient framework for exact inference in tree-structured graphs 
@ref DirectGraphicalModels::CInferTree, which is based on the \a sum-product algorithm:
@code
	using namespace DirectGraphicalModels;

	CInferTree inferer(graph);

	inferer.infer();
@endcode

Please note, that in case of a single source, its marginal probability is equal to the initial probability.
*/
