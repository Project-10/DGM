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

	void CKDNode::save(FILE *pFile) const
	{
		byte _isLeaf = isLeaf();
		fwrite(&_isLeaf, sizeof(byte), 1, pFile);
		if (isLeaf()) {		// --- Leaf node ---
			fwrite(m_key.data, sizeof(byte), m_key.cols, pFile);
			fwrite(&m_value, sizeof(byte), 1, pFile);
		} else {			// --- Branch node ---
			fwrite(m_boundingBox.first.data,  sizeof(byte), m_boundingBox.first.cols,  pFile);
			fwrite(m_boundingBox.second.data, sizeof(byte), m_boundingBox.second.cols, pFile);
			fwrite(&m_splitVal, sizeof(byte), 1, pFile);
			fwrite(&m_splitDim, sizeof(int),  1, pFile);

			m_pLeft->save(pFile);
			m_pRight->save(pFile);
		}
	}

	void CKDNode::findNearestNeighbors(const Mat &key, size_t maxNeighbors, pair_mat_t &searchBox, float &searchRadius, std::vector<std::shared_ptr<const CKDNode>> &nearestNeighbors) const
	{
		if (isLeaf()) {		// --- Leaf node ---
			float distance = mathop::Euclidian<byte, float>(key, m_key) + 0.5f;
			if (distance < searchRadius) {
				if (nearestNeighbors.size() < maxNeighbors) nearestNeighbors.push_back(shared_from_this());
				else nearestNeighbors.back() = shared_from_this();

				// Sort the nodes
				float distTo_i = mathop::Euclidian<byte, float>(key, nearestNeighbors.back()->m_key);
				for (size_t i = nearestNeighbors.size() - 1; i > 0; i--) {
					float distTo_p = mathop::Euclidian<byte, float>(key, nearestNeighbors[i - 1]->m_key);
					if (distTo_p <= distTo_i) break;
					std::swap(nearestNeighbors[i - 1], nearestNeighbors[i]);
				}

				if (nearestNeighbors.size() == maxNeighbors) {
					searchRadius = (maxNeighbors == 1) ? distance : mathop::Euclidian<byte, float>(key, nearestNeighbors.back()->m_key) + 0.5f;
					searchBox.first = key - searchRadius;
					searchBox.second = key + searchRadius;
				}
			} // if distance
		} else {			// --- Branch node ---
			// if (mathop::ifOverlap<byte>(m_pLeft->getBoundingBox(), searchBox))
			if (mathop::ifOverlap<byte>(m_pLeft->m_boundingBox, searchBox))
				m_pLeft->findNearestNeighbors(key, maxNeighbors, searchBox, searchRadius, nearestNeighbors);
			// if (mathop::ifOverlap<byte>(m_pRight->getBoundingBox(), searchBox))
			if (mathop::ifOverlap<byte>(m_pRight->m_boundingBox, searchBox))
				m_pRight->findNearestNeighbors(key, maxNeighbors, searchBox, searchRadius, nearestNeighbors);
		}
	}
}