#include "KDNode.h"
#include "mathop.h"

namespace DirectGraphicalModels
{
	// Constructor
	CKDNode::CKDNode(Mat &key, byte value)
		: m_value(value)
		, m_pLeft(nullptr)
		, m_pRight(nullptr)
	{
		key.copyTo(m_key);
	}

	// Constructor
	CKDNode::CKDNode(byte splitVal, int splitDim, std::shared_ptr<CKDNode> left, std::shared_ptr<CKDNode> right)
		: m_splitVal(splitVal)
		, m_splitDim(splitDim)
		, m_pLeft(left)
		, m_pRight(right)
	{ }

	// Destructor
	CKDNode::~CKDNode(void) 
	{ }
}