#include "MessagePassing.h"
#include "GraphPairwise.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
	void CMessagePassing::infer(unsigned int nIt)
	{
		const byte   nStates = getGraph().getNumStates();

		// ====================================== Initialization ======================================
		createMessages(1.0f / nStates);				// msg[] = 1 / nStates; msg_temp[] = 1 / nStates;

		// =================================== Calculating messages ==================================
		calculateMessages(nIt);

		// =================================== Calculating beliefs ===================================
#ifdef ENABLE_PDP
		parallel_for_(Range(0, getGraphPairwise().m_vNodes.size()), [&, nStates](const Range& range) {
#else
		const Range range(0, getGraphPairwise().m_vNodes.size());
#endif
		for (int i = range.start; i < range.end; i++) {
			auto& node = getGraphPairwise().m_vNodes[i];
			for (size_t e_f : node->from) {
				float* msg = getMessage(e_f);				// message of current incoming edge
				float epsilon = FLT_EPSILON;
				for (byte s = 0; s < nStates; s++) { 		// states
					// node->Pot.at<float>(s,0) *= msg[s];
					node->Pot.at<float>(s, 0) = (epsilon + node->Pot.at<float>(s, 0)) * (epsilon + msg[s]);		// Soft multiplication
				} //s
			} // e_f

			// Normalization
			float SUM_pot = 0;
			for (byte s = 0; s < nStates; s++)				// states
				SUM_pot += node->Pot.at<float>(s, 0);
			for (byte s = 0; s < nStates; s++) {			// states
				node->Pot.at<float>(s, 0) /= SUM_pot;
				DGM_ASSERT_MSG(!std::isnan(node->Pot.at<float>(s, 0)), "The lower precision boundary for the potential of the node %zu is reached.\n \
						SUM_pot = %f\n", node->id, SUM_pot);
			}
		}
#ifdef ENABLE_PDP
		});
#endif
		deleteMessages();
	}

	// dst: usually edge msg or edge msg_temp
	void CMessagePassing::calculateMessage(const Edge& edge_to, float* temp, float* dst, bool maxSum)
	{
		Node		* node = getGraphPairwise().m_vNodes[edge_to.node1].get();		// source node
		const byte	  nStates = getGraph().getNumStates();							// number of states

		// Compute temp = product of all incoming msgs except e_t
		for (byte s = 0; s < nStates; s++) temp[s] = node->Pot.at<float>(s, 0);		// temp = node.Pot

		for (size_t e_f : node->from) {												// incoming edges
			Edge *edge_from = getGraphPairwise().m_vEdges[e_f].get();				// current incoming edge
			if (edge_from->node1 != edge_to.node2) {
				float *msg = getMessage(e_f);										// message of current incoming edge
				for (byte s = 0; s < nStates; s++)
					temp[s] *= msg[s];												// temp = temp * msg
			}
		} // e_f

		// Compute new message: new_msg = (edge_to.Pot^2)^t x temp
		float Z = MatMul(edge_to.Pot, temp, dst, maxSum);

		// Normalization and setting new values
		if (Z > FLT_EPSILON)
			for (byte s = 0; s < nStates; s++)
				dst[s] /= Z;
		else
			for (byte s = 0; s < nStates; s++)
				dst[s] = 1.0f / nStates;
	}

	void CMessagePassing::createMessages(std::optional<float> val)
	{
		const size_t nEdges = getGraph().getNumEdges();
		const byte	nStates	= getGraph().getNumStates();
		
		m_msg = new float[nEdges * nStates];
		DGM_ASSERT_MSG(m_msg, "Out of Memory");
		m_msg_temp = new float[nEdges * nStates];
		DGM_ASSERT_MSG(m_msg_temp, "Out of Memory");

		if (val) {
			std::fill(m_msg, m_msg + nEdges * nStates, val.value());
			std::fill(m_msg_temp, m_msg_temp + nEdges * nStates, val.value());
		}
	}

	void CMessagePassing::deleteMessages(void)
	{
		if (m_msg) {
			delete[] m_msg;
			m_msg = NULL;
		}
		if (m_msg_temp) {
			delete[] m_msg_temp;
			m_msg_temp = NULL;
		}
	}

	void CMessagePassing::swapMessages(void)
	{
		float *pTemp = m_msg;
		m_msg = m_msg_temp;
		m_msg_temp = pTemp;
	}

	float* CMessagePassing::getMessage(size_t edge) 
	{ 
		return m_msg ? m_msg + edge * getGraph().getNumStates() : NULL;
	}

	float* CMessagePassing::getMessageTemp(size_t edge) 
	{ 
		return m_msg_temp ? m_msg_temp + edge * getGraph().getNumStates() : NULL;
	}

	// dst = (M * M)^T x v
	float CMessagePassing::MatMul(const Mat& M, const float* v, float* dst, bool maxSum)
	{
		float res = 0;
		DGM_ASSERT(dst);
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
