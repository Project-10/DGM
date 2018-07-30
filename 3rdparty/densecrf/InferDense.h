#pragma once

#include "types.h"

class CGraphDense;

class CInferDense
{
public:
	CInferDense(CGraphDense *pGraph) : m_pGraph(pGraph) {}
	virtual ~CInferDense(void) {}

	
	// Run inference and return the probabilities
	DllExport vec_float_t infer(unsigned int nIt = 1, float relax = 1.0f);

	// Run MAP inference and return the map for each pixel
	DllExport vec_byte_t decode(unsigned int nIt = 0, float relax = 1.0);

	// Step by step inference
	DllExport void startInference(void);
	DllExport void stepInference(float relax = 1.0);
	DllExport void currentMap(short *result);


#ifdef DEBUG_MODE1
	/* Debugging functions */
	// Compute the unary energy of an assignment
	DllExport void unaryEnergy(const short *ass, float *result);

	// Compute the pairwise energy of an assignment (half of each pairwise potential is added to each of it's endpoints)
	DllExport void pairwiseEnergy(const short *ass, float *result, int term = -1);
#endif

protected:
	CGraphDense * m_pGraph;		///< Pointer to the graph


private:
	// Run inference and return the pointer to the result
	vec_float_t runInference(unsigned int nIt, float relax);

	// Auxillary functions
	void expAndNormalize(vec_float_t &out, const vec_float_t &in, float scale = 1.0f, float relax = 1.0f);


private:
	vec_float_t m_vAdditionalUnary;
	vec_float_t m_vCurrent;					
	vec_float_t	m_vTmp;
	vec_float_t m_vNext;
};