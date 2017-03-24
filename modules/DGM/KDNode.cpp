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

	void CKDNode::findNearestNeighbor(const Mat &key, pair_mat_t &searchBox, float &searchRadius, std::shared_ptr<const CKDNode> &nearestNeighbor) const
	{
		std::vector<std::shared_ptr<const CKDNode>> nearestNeighbors;
		findNearestNeighbors(key, 1, searchBox, searchRadius, nearestNeighbors);
		nearestNeighbor = nearestNeighbors.front();

		/*if (isLeaf()) {		// --- Leaf node ---
			float distance = mathop::Euclidian<byte, float>(key, m_key) + 0.5f;
			if (distance < searchRadius) {
				nearestNeighbor = shared_from_this();
				searchRadius = distance;
				searchBox.first = key - searchRadius;
				searchBox.second = key + searchRadius;
			}
		} else {			// --- Branch node ---
			if (mathop::ifOverlap<byte>(m_pLeft->getBoundingBox(), searchBox))
				m_pLeft->findNearestNeighbor(key, searchBox, searchRadius, nearestNeighbor);
			if (mathop::ifOverlap<byte>(m_pRight->getBoundingBox(), searchBox))
				m_pRight->findNearestNeighbor(key, searchBox, searchRadius, nearestNeighbor);
		}*/
	}

	void CKDNode::findNearestNeighbors(const Mat &key, size_t k, pair_mat_t &searchBox, float &searchRadius, std::vector<std::shared_ptr<const CKDNode>> &nearestNeighbors) const
	{
		if (isLeaf()) {		// --- Leaf node ---
			float distance = mathop::Euclidian<byte, float>(key, m_key) + 0.5f;
			if (distance < searchRadius /*&& nearSet.find(m_key) == nearSet.end()*/) {
				if (nearestNeighbors.size() < k) nearestNeighbors.push_back(shared_from_this());
				else nearestNeighbors.back() = shared_from_this();

				// sort the nodes
				for (size_t i = nearestNeighbors.size() - 1; i > 0; i--) {
					if (mathop::Euclidian<byte, float>(key, nearestNeighbors[i - 1]->getKey()) > mathop::Euclidian<byte, float>(key, nearestNeighbors[i]->getKey()))
						std::swap(nearestNeighbors[i - 1], nearestNeighbors[i]);
					else
						break;
				}

				if (nearestNeighbors.size() == k) {
					searchRadius = (k == 1) ? distance : mathop::Euclidian<byte, float>(key, nearestNeighbors.back()->getKey()) + 0.5f;
					searchBox.first = key - searchRadius;
					searchBox.second = key + searchRadius;
				}
			}
		} else {			// --- Branch node ---
			if (mathop::ifOverlap<byte>(m_pLeft->getBoundingBox(), searchBox))
				m_pLeft->findNearestNeighbors(key, k, searchBox, searchRadius, nearestNeighbors/*, nearSet*/);
			if (mathop::ifOverlap<byte>(m_pRight->getBoundingBox(), searchBox))
				m_pRight->findNearestNeighbors(key, k, searchBox, searchRadius, nearestNeighbors/*, nearSet*/);
		}
	}
}