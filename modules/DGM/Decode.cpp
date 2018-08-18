#include "Decode.h"
#include "GraphPairwise.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
vec_byte_t CDecode::decode(const CGraphPairwise *pGraph, Mat &lossMatrix)
{
	size_t		nNodes		= pGraph->getNumNodes();			// number of nodes
	vec_byte_t	state(nNodes);
	Mat			pot;
	bool		ifLossMat	= !lossMatrix.empty();

	// Getting optimal state
	for (size_t n = 0; n < nNodes; n++) {						// all nodes
		pGraph->getNode(n, pot);
		if (ifLossMat) gemm(lossMatrix, pot, 1.0, Mat(), 0.0, pot);
		
		Point extremumLoc;
		if (ifLossMat) minMaxLoc(pot, NULL, NULL, &extremumLoc, NULL);
		else minMaxLoc(pot, NULL, NULL, NULL, &extremumLoc);
		state[n] = static_cast<byte>(extremumLoc.y);
	} // n

	return state;
}

Mat	CDecode::getDefaultLossMatrix(byte nStates)
{
	Mat res(nStates, nStates, CV_32FC1, Scalar(1.0f));
	for (byte i = 0; i < nStates; i++) res.at<float>(i,i) = 0.0f;
	return res;
}

// Sets the <state> according to the configuration number <c>
void CDecode::setState(vec_byte_t &state, qword c) const
{
	size_t nNodes = m_pGraph->getNumNodes();
	for (size_t n = 0; n < nNodes; n++) {
		state[n] = c % m_pGraph->m_nStates;
		c = (c - state[n]) / m_pGraph->m_nStates;
	}
}

// Increases the <state> by one
void CDecode::incState(vec_byte_t &state) const
{
	size_t nNodes = m_pGraph->getNumNodes();
	for (size_t n = 0; n < nNodes; n++)
		if (++state[n] >= m_pGraph->m_nStates) state[n] = 0;
		else break;
}

// Calculates potentials for all possible configurations
vec_float_t CDecode::calculatePotentials(void) const
{
	size_t		nNodes = m_pGraph->getNumNodes();
	size_t		nConfigurations = static_cast<size_t> (powl(m_pGraph->m_nStates, static_cast<long double>(nNodes)));
	vec_byte_t	state(nNodes);

	vec_float_t		  res;
	DGM_ASSERT_MSG(nConfigurations < res.max_size(), "The number of configurations %d^%zu exceeds the maximal possible size of container.", m_pGraph->m_nStates, nNodes);
	res.resize(nConfigurations, 1.0f);

	setState(state, 0);
	for (float &p: res) {
		for (ptr_node_t &node: m_pGraph->m_vNodes)	p *= node->Pot.at<float>(state[node->id], 0); 
		for (ptr_edge_t &edge : m_pGraph->m_vEdges) p *= edge->Pot.at<float>(state[edge->node1], state[edge->node2]);
		incState(state);
	} 

	return res;
}
}
