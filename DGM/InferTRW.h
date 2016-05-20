#pragma once

#include "Infer.h"

namespace DirectGraphicalModels
{
	class CInferTRW : public CInfer
	{
	public:
		DllExport CInferTRW(CGraph *pGraph) : CInfer(pGraph) {}
		DllExport virtual ~CInferTRW(void) {}

		DllExport virtual void	  infer(unsigned int nIt = 1);
	};
}
