#pragma once

#include "types.h"

namespace DirectGraphicalModels {

	class CGraphDense;

	class CInferDense
	{
	public:
		CInferDense(CGraphDense *pGraph) : m_pGraph(pGraph) {}
		virtual ~CInferDense(void) {}

		// Run MAP inference and return the map for each pixel
		DllExport vec_byte_t decode(unsigned int nIt = 0, float relax = 1.0);

		// Run inference and return the probabilities
		DllExport vec_float_t infer(unsigned int nIt, float relax);


	protected:
		CGraphDense * m_pGraph;		///< Pointer to the graph


	private:
		// Step by step inference
		void startInference(void);
		void stepInference(float relax = 1.0);
		void currentMap(short *result);


	private:
		Mat m_additionalUnary;
		Mat m_current;
		Mat	m_temp;
		Mat m_next;
	};

}