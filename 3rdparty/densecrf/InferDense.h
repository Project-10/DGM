#pragma once

#include "types.h"

class CGraphDense;

class CInferDense
{
public:
	CInferDense(CGraphDense *pGraph) : m_pGraph(pGraph) {}
	virtual ~CInferDense(void) {}

    	// Run MAP inference and return the map for each pixel
	DllExport vec_byte_t decode(unsigned int nIt = 0, float relax = 1.0);

    // Run inference and return the probabilities
    vec_float_t infer(unsigned int nIt, float relax);
    
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
	Mat m_additionalUnary;
	Mat m_current;
	Mat	m_temp;
	Mat m_next;
};
