#include "DecodeExact.h"
#include "macroses.h"
#include <numeric>

namespace DirectGraphicalModels
{
	vec_byte_t CDecodeExact::decode(unsigned int, Mat &lossMatrix) const
	{
		DGM_IF_WARNING(!lossMatrix.empty(), "The Loss Matrix is not supported by the algorithm.");

		size_t			nNodes = getGraph()->getNumNodes();
		vec_byte_t		state(nNodes);

		// Calculating the potentials for every possible configuration
		vec_float_t		  P = calculatePotentials();

		// Calculating the partition function
		float Z = std::accumulate(P.cbegin(), P.cend(), 0.0f);

#ifdef DEBUG_PRINT_INFO
		// Printing out
		printf("nConfigurations = %zd\n", P.size());

		setState(state, 0);
		std::for_each(P.begin(), P.end(), [&](float & p) {
			for (size_t n = 0; n < nNodes; n++) printf("%d ", state[n]);
			printf(":-> %2.1f\t| %2.1f %% \n", p, p * 100 / Z);
			incState(state);
		});
#endif

		// Finding the most probable configuration
		qword c = std::max_element(P.cbegin(), P.cend()) - P.begin();
		setState(state, c);
		return state;
	}

	// Sets the <state> according to the configuration number <c>
	void CDecodeExact::setState(vec_byte_t &state, qword c) const
	{
		size_t nNodes = getGraph()->getNumNodes();
		for (size_t n = 0; n < nNodes; n++) {
			state[n] = c % getGraph()->getNumStates();
			c = (c - state[n]) / getGraph()->getNumStates();
		}
	}

	// Increases the <state> by one
	void CDecodeExact::incState(vec_byte_t &state) const
	{
		size_t nNodes = getGraph()->getNumNodes();
		for (size_t n = 0; n < nNodes; n++)
			if (++state[n] >= getGraph()->getNumStates()) state[n] = 0;
			else break;
	}

	// Calculates potentials for all possible configurations
	vec_float_t CDecodeExact::calculatePotentials(void) const
	{
		size_t		nNodes = getGraph()->getNumNodes();
		size_t		nConfigurations = static_cast<size_t> (powl(getGraph()->getNumStates(), static_cast<long double>(nNodes)));
		vec_byte_t	state(nNodes);

		vec_float_t		  res;
		DGM_ASSERT_MSG(nConfigurations < res.max_size(), "The number of configurations %d^%zu exceeds the maximal possible size of container.", getGraph()->getNumStates(), nNodes);
		res.resize(nConfigurations, 1.0f);

		setState(state, 0);
		for (float &p : res) {
			for (ptr_node_t &node : getGraphPairwise()->m_vNodes) p *= node->Pot.at<float>(state[node->id], 0);
			for (ptr_edge_t &edge : getGraphPairwise()->m_vEdges) p *= edge->Pot.at<float>(state[edge->node1], state[edge->node2]);
			incState(state);
		}

		return res;
	}
}
