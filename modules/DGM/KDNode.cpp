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

	void CKDNode::findNearestNeighbors(const Mat &key, size_t maxNeighbors, pair_mat_t &searchBox, float &searchRadius, std::vector<std::shared_ptr<const CKDNode>> &nearestNeighbors) const
	{
		if (isLeaf()) {		// --- Leaf node ---
			float distance = mathop::Euclidian<byte, float>(key, m_key) + 0.5f;
			if (distance < searchRadius) {
				if (nearestNeighbors.size() < maxNeighbors) nearestNeighbors.push_back(shared_from_this());
				else nearestNeighbors.back() = shared_from_this();

				// Sort the nodes
				for (size_t i = nearestNeighbors.size() - 1; i > 0; i--) {
					if (mathop::Euclidian<byte, float>(key, nearestNeighbors[i - 1]->getKey()) > mathop::Euclidian<byte, float>(key, nearestNeighbors[i]->getKey()))
						std::swap(nearestNeighbors[i - 1], nearestNeighbors[i]);
					else
						break;
				}

				if (nearestNeighbors.size() == maxNeighbors) {
					searchRadius = (maxNeighbors == 1) ? distance : mathop::Euclidian<byte, float>(key, nearestNeighbors.back()->getKey()) + 0.5f;
					searchBox.first = key - searchRadius;
					searchBox.second = key + searchRadius;
				}
			} // if distance
		} else {			// --- Branch node ---
			if (mathop::ifOverlap<byte>(m_pLeft->getBoundingBox(), searchBox))
				m_pLeft->findNearestNeighbors(key, maxNeighbors, searchBox, searchRadius, nearestNeighbors);
			if (mathop::ifOverlap<byte>(m_pRight->getBoundingBox(), searchBox))
				m_pRight->findNearestNeighbors(key, maxNeighbors, searchBox, searchRadius, nearestNeighbors);
		}
	}
}