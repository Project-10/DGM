// K-Dimensional Tree class interface
// Copied from http://codereview.stackexchange.com/questions/110225/k-d-tree-implementation-in-c11
#pragma once

#include "types.h"

#include "KDNode.h"

namespace DirectGraphicalModels
{
	class CKDTree
	{
	public:
		DllExport static std::shared_ptr<CKDNode> createTree(Mat &vData);
	};
}