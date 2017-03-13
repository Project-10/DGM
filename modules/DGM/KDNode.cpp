#include "KDNode.h"
#include "mathop.h"

namespace DirectGraphicalModels
{
	// Protected Constructor
	CKDNode::CKDNode(Mat &key, byte value, pair_mat_t &boundingBox, byte splitVal, int splitDim, std::shared_ptr<CKDNode> left, std::shared_ptr<CKDNode> right)
		: m_value(value)
		, m_boundingBox(boundingBox)
		, m_splitVal(splitVal)
		, m_splitDim(splitDim)
		, m_pLeft(left)
		, m_pRight(right)
	{
		key.copyTo(m_key);
	}

	void CKDNode::findNearestNeighbor(Mat &key, pair_mat_t &searchBox, float &searchRadius, std::shared_ptr<const CKDNode> &nearestNeighbor) const
	{
		if (isLeaf()) {		// --- Leaf node ---
			float distance = mathop::Euclidian<byte, float>(key, m_key) + 0.5f;
			if (distance < searchRadius) {
				searchRadius = distance;
				nearestNeighbor = shared_from_this(); 

				searchBox.first = key - searchRadius;
				searchBox.second = key + searchRadius;
			}
		} else {			// --- Branch node ---
			if (mathop::ifOverlap<byte>(m_pLeft->getBoundingBox(), searchBox))
				m_pLeft->findNearestNeighbor(key, searchBox, searchRadius, nearestNeighbor);
			if (mathop::ifOverlap<byte>(m_pRight->getBoundingBox(), searchBox))
				m_pRight->findNearestNeighbor(key, searchBox, searchRadius, nearestNeighbor);
		}
	}

}