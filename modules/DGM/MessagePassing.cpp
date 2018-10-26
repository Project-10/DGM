#include "MessagePassing.h"
#include "GraphPairwise.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
void CMessagePassing::infer(unsigned int nIt)
{
	const byte   nStates = getGraph().getNumStates();

	// ====================================== Initialization ======================================			
	createMessages(); 
#ifdef ENABLE_PPL
	concurrency::parallel_for_each(getGraphPairwise().m_vEdges.begin(), getGraphPairwise().m_vEdges.end(), [nStates](ptr_edge_t &edge) {
#else
	std::for_each(getGraphPairwise().m_vEdges.begin(), getGraphPairwise().m_vEdges.end(), [nStates](ptr_edge_t &edge) {
#endif
		std::fill(edge->msg, edge->msg + nStates, 1.0f / nStates);							// msg[] = 1 / nStates;
		std::fill(edge->msg_temp, edge->msg_temp + nStates, 1.0f / nStates);					// msg_temp[] = 1 / nStates;
	});

	// =================================== Calculating messages ==================================	
	calculateMessages(nIt);

	// =================================== Calculating beliefs ===================================	
#ifdef ENABLE_PPL
	concurrency::parallel_for_each(getGraphPairwise().m_vNodes.begin(), getGraphPairwise().m_vNodes.end(), [&, nStates](ptr_node_t &node) {
#else
	std::for_each(getGraphPairwise().m_vNodes.begin(), getGraphPairwise().m_vNodes.end(), [&,nStates](ptr_node_t &node) {
#endif
		size_t nFromEdges = node->from.size();
		// Don't understand the normalization step, replaced with another version.
		//for (size_t e_f = 0; e_f < nFromEdges; e_f++) {				
		//	Edge *edge_from = m_pGraph->m_vEdges[node->from[e_f]].get();	// current incoming edge
		//	float SUM_pot = 0;

		//	float epsilon = FLT_EPSILON;
		//	for (byte s = 0; s < nStates; s++) { 		// states
		//		SUM_pot += node->Pot.at<float>(s, 0);
		//		// node.Pot.at<float>(s,0) *= edge_from->msg[s];
		//		node->Pot.at<float>(s, 0) = (epsilon + node->Pot.at<float>(s, 0)) * (epsilon + edge_from->msg[s]);		// Soft multiplication
		//	} //s
		//	
		//	// Normalization
		//	float SUM_new_pot = 0;
		//	for (byte s = 0; s < nStates; s++)			// states
		//		SUM_new_pot += node->Pot.at<float>(s, 0);
		//	for (byte s = 0; s < nStates; s++) {		// states
		//		node->Pot.at<float>(s, 0) *= SUM_pot / SUM_new_pot;
		//		//node->Pot.at<float>(s, 0) /= SUM_new_pot;
		//		DGM_ASSERT_MSG(!std::isnan(node->Pot.at<float>(s, 0)), "The lower precision boundary for the potential of the node %zu is reached.\n \
				//			SUM_pot = %f\nSUM_new_pot = %f\n", node->id, SUM_pot, SUM_new_pot);
//	}
//} // e_f


		for (size_t e_f = 0; e_f < nFromEdges; e_f++) {
			Edge *edge_from = getGraphPairwise().m_vEdges[node->from[e_f]].get();	// current incoming edge

			float epsilon = FLT_EPSILON;
			for (byte s = 0; s < nStates; s++) { 		// states
														// node.Pot.at<float>(s,0) *= edge_from->msg[s];
				node->Pot.at<float>(s, 0) = (epsilon + node->Pot.at<float>(s, 0)) * (epsilon + edge_from->msg[s]);		// Soft multiplication
			} //s
		} // e_f
		  // Normalization
		float SUM_pot = 0;
		for (byte s = 0; s < nStates; s++)			// states
			SUM_pot += node->Pot.at<float>(s, 0);
		for (byte s = 0; s < nStates; s++) {		// states
			node->Pot.at<float>(s, 0) /= SUM_pot;
			//node->Pot.at<float>(s, 0) /= SUM_new_pot;
			DGM_ASSERT_MSG(!std::isnan(node->Pot.at<float>(s, 0)), "The lower precision boundary for the potential of the node %zu is reached.\n \
					SUM_pot = %f\n", node->id, SUM_pot);
		}
	});

	deleteMessages();
}

// dst: usually edge_to->msg or edge_to->msg_temp
void CMessagePassing::calculateMessage(Edge *edge_to, float *temp, float *&dst, bool maxSum)
{
	byte		  s;													// state indexes
	Node		* node = getGraphPairwise().m_vNodes[edge_to->node1].get();		// source node
	size_t		  nFromEdges = node->from.size();						// number of incoming eges
	const byte	  nStates = getGraph().getNumStates();					// number of states

	// Compute temp = product of all incoming msgs except e_t
	for (s = 0; s < nStates; s++) temp[s] = node->Pot.at<float>(s, 0);		// temp = node.Pot

	for (size_t e_f = 0; e_f < nFromEdges; e_f++) {							// incoming edges
		Edge *edge_from = getGraphPairwise().m_vEdges[node->from[e_f]].get();		// current incoming edge
		if (edge_from->node1 != edge_to->node2)
			for (s = 0; s < nStates; s++)
				temp[s] *= edge_from->msg[s];								// temp = temp * msg
		else
			edge_from->suspend = true;
	} // e_f

	// Compute new message: new_msg = (edge_to.Pot^2)^t x temp
	float Z = MatMul(edge_to->Pot, temp, dst, maxSum);

	// Normalization and setting new values
	if (Z > FLT_EPSILON)
		for (s = 0; s < nStates; s++)
			dst[s] /= Z;
	else
		for (s = 0; s < nStates; s++)
			dst[s] = 1.0f / nStates;
}

void CMessagePassing::createMessages(void)
{
	const byte nStates = getGraph().getNumStates();
#ifdef ENABLE_PPL
	concurrency::parallel_for_each(getGraphPairwise().m_vEdges.begin(), getGraphPairwise().m_vEdges.end(), [nStates](ptr_edge_t &edge) {
#else
	std::for_each(getGraphPairwise().m_vEdges.begin(), getGraphPairwise().m_vEdges.end(), [nStates](ptr_edge_t &edge) {
#endif
		if (!edge->msg) edge->msg = new float[nStates];
		DGM_ASSERT_MSG(edge->msg, "Out of Memory");

		if (!edge->msg_temp) edge->msg_temp = new float[nStates];
		DGM_ASSERT_MSG(edge->msg_temp, "Out of Memory");
	});
}

void CMessagePassing::deleteMessages(void)
{
#ifdef ENABLE_PPL
	concurrency::parallel_for_each(getGraphPairwise().m_vEdges.begin(), getGraphPairwise().m_vEdges.end(), [](ptr_edge_t &edge) {
#else
	std::for_each(getGraphPairwise().m_vEdges.begin(), getGraphPairwise().m_vEdges.end(), [](ptr_edge_t &edge) {
#endif
		if (edge->msg) {
			delete[] edge->msg;
			edge->msg = NULL;
		}
		if (edge->msg_temp) {
			delete[] edge->msg_temp;
			edge->msg_temp = NULL;
		}
	});
}

void CMessagePassing::swapMessages(void)
{
#ifdef ENABLE_PPL
	concurrency::parallel_for_each(getGraphPairwise().m_vEdges.begin(), getGraphPairwise().m_vEdges.end(), [](ptr_edge_t &edge) { edge->msg_swap(); });
#else
	std::for_each(getGraphPairwise().m_vEdges.begin(), getGraphPairwise().m_vEdges.end(), [](ptr_edge_t &edge) { edge->msg_swap(); });
#endif	
}

// dst = (M * M)^T x v
float CMessagePassing::MatMul(const Mat &M, const float *v, float *&dst, bool maxSum)
{
	float res = 0;
	if (!dst) dst = new float[M.cols];
	for (int x = 0; x < M.cols; x++) {
		float sum = 0;
		for (int y = 0; y < M.rows; y++) {
			float m = M.at<float>(y, x);
			float prod = v[y] * m * m;
			if (maxSum) { if (prod > sum) sum = prod; }
			else sum += prod;
		} // y
		dst[x] = sum;
		res += sum;
	} // x
	return res;
}
}
