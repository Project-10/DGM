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
//		DllExport CKDTree(void) : m_root(nullptr) {}
//		DllExport CKDTree(Mat &data) : m_root(nullptr) { build(data); }	
//		DllExport CKDTree(const CKDTree &obj) = delete;
//		DllExport ~CKDTree(void) {}
		
//		DllExport bool operator==(const CKDTree rhs) = delete;
//		DllExport bool operator=(const CKDTree rhs)  = delete;

//		DllExport void								reset(void) { m_root.reset(); }
//		DllExport void								build(Mat &data);

		DllExport static std::shared_ptr<CKDNode>	createTree(Mat &data);
		DllExport static float						findNearestNeighbor(std::shared_ptr<CKDNode> root, Mat &key);
		DllExport static std::shared_ptr<CKDNode>	findNearestNode(std::shared_ptr<CKDNode> root, Mat &key);
	

//	private:
//		DllExport std::shared_ptr<CKDNode>			findNearestNode(Mat &key) const;


//	private:
//		std::shared_ptr<CKDNode>	m_root;
	};
}