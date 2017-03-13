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

	namespace {
		template<typename T>
		bool ifOverlap(pair_mat_t &box1, pair_mat_t &box2)
		{
			for (int x = 0; x < box1.first.cols; x++) {
				if (box1.first.at<T>(0, x) > box2.second.at<T>(0, x))	return false;
				if (box1.second.at<T>(0, x) < box2.first.at<T>(0, x))	return false;
			}
			return true;
		}
	}

	void CKDNode::findNearestNeighbor(Mat &key, float &minDistance,  pair_mat_t &searchBox) const
	{
		if (m_pLeft)
			if (ifOverlap<byte>(m_pLeft->getBoundingBox(), searchBox))
				m_pLeft->findNearestNeighbor(key, minDistance, searchBox);
		if (m_pRight)
			if (ifOverlap<byte>(m_pRight->getBoundingBox(), searchBox))
				m_pRight->findNearestNeighbor(key, minDistance, searchBox);

		// Leaf
		if (isLeaf()) {
			float distance = mathop::Euclidian<byte, float>(key, m_key) + 0.5f;
			if (distance < minDistance) {
				minDistance = distance;

				searchBox.first = key - minDistance;
				searchBox.second = key + minDistance;

			}
		}
		/*


		if (m_Left != nullptr && regionCrossesRegion(m_Left->getBoundingBox(), minRegion))
			m_Left->FindNearestNeighbor(srcPoint, nearPoint, minDistance, minRegion, Depth + 1);
		if (m_Right != nullptr && regionCrossesRegion(m_Right->getBoundingBox(), minRegion))
			m_Right->FindNearestNeighbor(srcPoint, nearPoint, minDistance, minRegion, Depth + 1);

		// Leaf
		if (mathop::Euclidian<float>(srcPoint, m_point) <= minDistance) {
			nearPoint = m_point;
			minDistance = mathop::Euclidian(srcPoint, nearPoint);

			for (size_t i = 0; i < srcPoint.size(); i++) {
				minRegion.first[i] = srcPoint[i] - minDistance;
				minRegion.second[i] = srcPoint[i] + minDistance;
			}
		}
		*/

	}

}