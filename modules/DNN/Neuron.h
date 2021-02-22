#pragma once

#include "types.h"

namespace DirectGraphicalModels { namespace dnn
{
	class CNeuron
	{
	public:
		DllExport CNeuron(void) { printf("CNeuron constructor\n"); }
		DllExport ~CNeuron(void) = default;
	private:
	};
} }