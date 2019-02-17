// Example "CHAIN" 1D-case with exact chain inference
// (https://www.cs.ubc.ca/~schmidtm/Software/UGM/chain.html)
/// @cond DemoCode
#pragma once

#include "Demo 1D.h"

class CChain : public CDemo1D
{
public:
	virtual void Main(void);

private:
	Mat getNodePot();
	Mat getEdgePot();
};
/// @endcond

/**
@page demo1d_chain Chain
In this demo we consider more complex problem involving the chain of nodes.  Such a problem is common in 
<a href="https://en.wikipedia.org/wiki/Markov_chain" target="_blank">Markov chains</a> and 
<a href="https://en.wikipedia.org/wiki/Hidden_Markov_model" target="_blank">hidden Markov models</a>
These models may incorporate large graphs, what prevents from using brute-force exact decoding and inference algorithms. 
Nevertheless, we can take an advantage of the conditional independence properties induced by the chain structure and use polynomial-time exact message-passing inference algorithm.
This example copies the idea from the <a href="https://www.cs.ubc.ca/~schmidtm/Software/UGM/chain.html" target="_blanck">Computer Science Graduate Careers</a>

> In order to run this demo, please execute \b "Demo 1D.exe" \b chain command

Building The Markov Chain
-------------------------
We build a chain, consisting of 60 nodes, which represents 60 years. Every node has 7 states and the first, initial node will have the potential: \f$(0.3, 0.6, 0.1, 0, 0, 0, 0)^\top\f$, whereas
all other nodes are initiated with uniform distribution: \f$(\frac{1}{7}, \frac{1}{7}, \frac{1}{7}, \frac{1}{7}, \frac{1}{7}, \frac{1}{7}, \frac{1}{7})^\top\f$. We connect all nodes into a chain 
with undirected arcs:
@code
	using namespace DirectGraphicalModels;
	
	const byte	 nStates = 7;				// {grad scool, industry, video games, industry (with PhD), academia, video games (with PhD), deceased}
	const size_t nNodes = 60;				// sixty years

	CGraphPairwise	graph(nStates);

	Mat nodePot = getNodePot();
	graph.addNode(nodePot);					// add the first node
	nodePot.setTo(1.0f  / nStates);			// uniform distribution
	for (size_t i = 1; i < nNodes; i++) 
		graph.addNode(nodePot);				// add nodes

	Mat edgePot = getEdgePot();
	for (size_t i = 0; i < nNodes - 1; i++) 
		graph.addArc(i, i + 1, edgePot);	// add arcs
@endcode

The edge potential matrix is represented here as a transition matrix:
<table>
	<tr><th>from\\to<th>st.0<th>st.1<th>st.2<th>st.3<th>st.4<th>st.5<th>st.6
	<tr><th>st.0    <td>0.08<td>0.90<td>0.01<td>0   <td>0   <td>0   <td>0.01
	<tr><th>st.1    <td>0.03<td>0.95<td>0.01<td>0   <td>0   <td>0   <td>0.01
	<tr><th>st.2    <td>0.06<td>0.06<td>0.75<td>0.05<td>0.05<td>0.02<td>0.01
	<tr><th>st.3    <td>0   <td>0   <td>0   <td>0.30<td>0.60<td>0.09<td>0.01
	<tr><th>st.4    <td>0   <td>0   <td>0   <td>0.02<td>0.95<td>0.02<td>0.01
	<tr><th>st.5    <td>0   <td>0   <td>0   <td>0.01<td>0.01<td>0.97<td>0.01
	<tr><th>st.6    <td>0   <td>0   <td>0   <td>0   <td>0   <td>0   <td>1   
</table>
\a e.g. a probability of switching from state 3 at year i to state 5 at year i + 1 is 0.09. The node and edge potentials are returned with functions \b getNodePot() and \b getEdgePot() respectively.


Inference
---------
The number of all possible configurations for our chain is equal to \f$7^{60}\approx 5.08\times 10^{50}\f$, and it is not feasible to calculate joint probabilities for all of these configurations.
Nevertheless, DGM has exact inference method for chains @ref DirectGraphicalModels::CInferChain, which is based on a message-passing algorithm with the total cost of evaluating the marginals of
\f$O(nNodes\cdot nStates^2)\f$:
@code
	using namespace DirectGraphicalModels;

	CInferChain inferer(graph);

	inferer.infer();
@endcode


Results
-------
Finally, we depict the results of inference. Please note, that the marginal probability for the first node is equal to the initial probability:
<img src="res_demo1d_chain.gif">

*/
