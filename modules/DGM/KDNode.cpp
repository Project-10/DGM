#include "KDNode.h"

namespace DirectGraphicalModels
{
	float	Distance(const vec_float_t &P, const vec_float_t &Q)
	{
		float sum = 0;
		for (size_t i = 0; i < P.size(); i++)
			sum += (P[i] - Q[i]) * (P[i] - Q[i]);
		return sqrtf(sum);
	}

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

	void kd_internal_node::SearchKdTree(const kd_Box &searchBox, std::vector<vec_float_t> &Points, size_t Depth) const
	{
		if (regionCrossesRegion(searchBox, m_Left->getBoundingBox()))							m_Left->SearchKdTree(searchBox, Points, Depth + 1);
		if (m_Right != nullptr && regionCrossesRegion(searchBox, m_Right->getBoundingBox()))	m_Right->SearchKdTree(searchBox, Points, Depth + 1);
	}
	
	void kd_internal_node::FindNearestNeighbor(const vec_float_t &srcPoint, vec_float_t &nearPoint, float &minDistance, kd_Box &minRegion, size_t Depth) const
	{
		if (regionCrossesRegion(m_Left->getBoundingBox(), minRegion))							m_Left->FindNearestNeighbor(srcPoint, nearPoint, minDistance, minRegion, Depth + 1);
		if (m_Right != nullptr && regionCrossesRegion(m_Right->getBoundingBox(), minRegion))	m_Right->FindNearestNeighbor(srcPoint, nearPoint, minDistance, minRegion, Depth + 1);
	}
	
	void kd_internal_node::FindKNearestNeighbors(const vec_float_t &srcPoint, std::vector<vec_float_t> &nearPoints, const unsigned k, float &minDistance, kd_Box &minRegion, std::unordered_set<vec_float_t, CKDPointHasher> &nearSet, size_t Depth) const
	{
		if (regionCrossesRegion(m_Left->getBoundingBox(), minRegion))							m_Left->FindKNearestNeighbors(srcPoint, nearPoints, k, minDistance, minRegion, nearSet, Depth + 1);
		if (m_Right != nullptr && regionCrossesRegion(m_Right->getBoundingBox(), minRegion))	m_Right->FindKNearestNeighbors(srcPoint, nearPoints, k, minDistance, minRegion, nearSet, Depth + 1);
	}

	// -------------------------------------------------------------

	void kd_leaf_node::SearchKdTree(const kd_Box &searchBox, std::vector<vec_float_t> &Points, size_t Depth) const
	{
		if (pointIsInRegion(m_pointCoords, searchBox)) Points.push_back(m_pointCoords);
	}

	void kd_leaf_node::FindNearestNeighbor(const vec_float_t &srcPoint, vec_float_t &nearPoint, float &minDistance, kd_Box &minRegion, size_t Depth) const
	{
		if (Distance(srcPoint, m_pointCoords) <= minDistance) {
			nearPoint = m_pointCoords;
			minDistance = Distance(srcPoint, nearPoint);

			for (size_t i = 0; i < srcPoint.size(); i++) {
				minRegion.first[i] = srcPoint[i] - minDistance;
				minRegion.second[i] = srcPoint[i] + minDistance;
			}
		}
	}
	
	void kd_leaf_node::FindKNearestNeighbors(const vec_float_t &srcPoint, std::vector<vec_float_t> &nearPoints, const unsigned k, float &minDistance, kd_Box &minRegion, std::unordered_set<vec_float_t, CKDPointHasher> &nearSet, size_t Depth) const
	{
		if (Distance(srcPoint, m_pointCoords) <= minDistance && nearSet.find(m_pointCoords) == nearSet.end()) {
			nearSet.erase(nearPoints[k - 1]);
			nearSet.insert(m_pointCoords);

			nearPoints[k - 1] = m_pointCoords;

			for (unsigned i = k - 1; i > 0; i--)
				if (Distance(srcPoint, nearPoints[i - 1]) > Distance(srcPoint, nearPoints[i]))
					swap(nearPoints[i - 1], nearPoints[i]);
				else
					break;

			minDistance = Distance(srcPoint, nearPoints[k - 1]);

			for (size_t i = 0; i < srcPoint.size(); i++) {
				minRegion.first[i] = srcPoint[i] - minDistance;
				minRegion.second[i] = srcPoint[i] + minDistance;
			}
		}
	}
}