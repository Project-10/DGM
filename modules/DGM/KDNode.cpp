#include "KDNode.h"
#include "mathop.h"

namespace DirectGraphicalModels
{
	bool	pointIsInRegion(const vec_float_t &point, const std::pair<vec_float_t, vec_float_t> &Region)
	{
		for (size_t i = 0; i < point.size(); i++)
			if (!(Region.first[i] <= point[i] && point[i] <= Region.second[i]))
				return false;

		return true;
	}

	bool	regionCrossesRegion(const std::pair<vec_float_t, vec_float_t> &Region1, const std::pair<vec_float_t, vec_float_t> &Region2)
	{
		for (size_t i = 0; i < Region1.first.size(); i++)
			if (Region1.first[i] > Region2.second[i] || Region1.second[i] < Region2.first[i])
				return false;
		return true;
	}

	// -------------------------------------------------------------

	size_t CKDNode::getNodeCount(bool withInternalNodes) const
	{
		if (isLeaf()) return 1;			// leaf

		size_t res = m_Left->getNodeCount(withInternalNodes);
		if (m_Right != nullptr)
			res += m_Right->getNodeCount(withInternalNodes);
		if (withInternalNodes) res++;
		return res;
	}

	void CKDNode::SearchKdTree(const kd_Box &searchBox, std::vector<vec_float_t> &Points, size_t Depth) const
	{
		if (m_Left != nullptr && regionCrossesRegion(searchBox, m_Left->getBoundingBox()))							
			m_Left->SearchKdTree(searchBox, Points, Depth + 1);
		if (m_Right != nullptr && regionCrossesRegion(searchBox, m_Right->getBoundingBox()))	
			m_Right->SearchKdTree(searchBox, Points, Depth + 1);

		// Leaf
		if (pointIsInRegion(m_point, searchBox))
			Points.push_back(m_point);
	}
	
	void CKDNode::FindNearestNeighbor(const vec_float_t &srcPoint, vec_float_t &nearPoint, float &minDistance, kd_Box &minRegion, size_t Depth) const
	{
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
	}
	
	void CKDNode::FindKNearestNeighbors(const vec_float_t &srcPoint, std::vector<vec_float_t> &nearPoints, const unsigned k, float &minDistance, kd_Box &minRegion, std::unordered_set<vec_float_t, CKDPointHasher> &nearSet, size_t Depth) const
	{
		if (m_Left != nullptr && regionCrossesRegion(m_Left->getBoundingBox(), minRegion))							
			m_Left->FindKNearestNeighbors(srcPoint, nearPoints, k, minDistance, minRegion, nearSet, Depth + 1);
		if (m_Right != nullptr && regionCrossesRegion(m_Right->getBoundingBox(), minRegion))	
			m_Right->FindKNearestNeighbors(srcPoint, nearPoints, k, minDistance, minRegion, nearSet, Depth + 1);

		// Leaf
		if (mathop::Euclidian(srcPoint, m_point) <= minDistance && nearSet.find(m_point) == nearSet.end()) {
			nearSet.erase(nearPoints[k - 1]);
			nearSet.insert(m_point);

			nearPoints[k - 1] = m_point;

			for (unsigned i = k - 1; i > 0; i--)
				if (mathop::Euclidian(srcPoint, nearPoints[i - 1]) > mathop::Euclidian(srcPoint, nearPoints[i]))
					swap(nearPoints[i - 1], nearPoints[i]);
				else
					break;

			minDistance = mathop::Euclidian(srcPoint, nearPoints[k - 1]);

			for (size_t i = 0; i < srcPoint.size(); i++) {
				minRegion.first[i] = srcPoint[i] - minDistance;
				minRegion.second[i] = srcPoint[i] + minDistance;
			}
		}
	}

}