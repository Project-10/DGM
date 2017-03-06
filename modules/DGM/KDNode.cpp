#include "KDNode.h"
#include "mathop.h"

namespace DirectGraphicalModels
{
	// Constructor
	CKDNode::CKDNode(Mat &key)
		: m_pLeft(NULL)
		, m_pRight(NULL)
	{
		key.copyTo(m_key);
	}

	// Constructor
	CKDNode::CKDNode(byte median, CKDNode *left, CKDNode *right)
		: m_median(median)
		, m_pLeft(left)
		, m_pRight(right)
	{ }

	// Destructor
	CKDNode::~CKDNode(void) 
	{
		if (m_pLeft) delete m_pLeft;
		if (m_pRight) delete m_pRight;
	}
}