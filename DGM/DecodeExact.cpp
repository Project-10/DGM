#include "DecodeExact.h"
#include "Graph.h"
#include "macroses.h"
#include <numeric>

namespace DirectGraphicalModels
{
vec_byte_t CDecodeExact::decode(unsigned int, Mat &lossMatrix) const
{
	DGM_IF_WARNING(!lossMatrix.empty(), "The Loss Matrix is not supported by the algorithm.");

	size_t			nNodes = m_pGraph->getNumNodes();
	vec_byte_t		state(nNodes);

	// Calculating the potentials for every possible configuration
	vec_float_t		  P = calculatePotentials();

	// Calculating the partition function
	float Z = std::accumulate(P.cbegin(), P.cend(), 0.0f);

#ifdef PRINT_DEBUG_INFO
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

}
