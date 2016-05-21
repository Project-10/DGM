#pragma once

#include "MessagePassing.h"

namespace DirectGraphicalModels
{
	class CInferTRW : public CMessagePassing
	{
	public:
		DllExport CInferTRW(CGraph *pGraph) : CMessagePassing(pGraph) {}
		DllExport virtual ~CInferTRW(void) {}

		DllExport virtual void	  infer(unsigned int nIt = 1);


	protected:
		virtual void calculateMessages(unsigned int nIt) {}
	};
}
