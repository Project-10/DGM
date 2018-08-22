#pragma once

#include "../modules/DGM/Infer.h"
#include "../modules/DGM/GraphDense.h"

namespace DirectGraphicalModels 
{
	class CInferDense : public CInfer
	{
	public:
		DllExport CInferDense(CGraphDense *pGraph) : CInfer(pGraph) {}
		DllExport virtual ~CInferDense(void) {}

		DllExport virtual void	infer(unsigned int nIt = 1);


	protected:
		/**
		* @brief Returns the pointer to the graph
		* @return The pointer to the graph
		*/
		CGraphDense * getGraphDense(void) const { return reinterpret_cast<CGraphDense *>(getGraph()); }


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