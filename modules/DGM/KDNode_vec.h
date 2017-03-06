#pragma once

#include "types.h"
#include <unordered_set>

namespace DirectGraphicalModels { namespace veccc 
{
	using kd_Box = std::pair<vec_float_t, vec_float_t>;

	class CKDPointHasher
	{
	public:
		size_t operator() (const vec_float_t &point) const
		{
			size_t retVal = 0;
			for (float p : point)
				retVal ^= std::hash<float>()(p);			// !!! (1,2) and (2,1) hash to the same value
															//boost::hash_combine(retVal, p);
			return retVal;
		}
	};

	class CKDNode
	{
	public:
		CKDNode(float splitVal, kd_Box &boundingBox, std::shared_ptr<CKDNode> Left = nullptr, std::shared_ptr<CKDNode> Right = nullptr)
			: m_splitVal(splitVal)
			, m_boundingBox(boundingBox)
			, m_Left(Left)
			, m_Right(Right)
		{}

		CKDNode(const vec_float_t &Point)
			: m_point(Point)
			, m_Left(nullptr)
			, m_Right(nullptr)
		{}

		bool						isLeaf(void) const { return m_Left == nullptr; }
		size_t						getTreeHeight(void) const {	return 1 + max(m_Left != nullptr ? m_Left->getTreeHeight() : 0, m_Right != nullptr ? m_Right->getTreeHeight() : 0); }
		size_t						getNodeCount(bool withInternalNodes) const;
		void						SearchKdTree(const kd_Box &searchBox, std::vector<vec_float_t> &Points, size_t Depth) const;
		void						FindNearestNeighbor(const vec_float_t &srcPoint, vec_float_t &nearPoint, float &minDistance, kd_Box &minRegion, size_t Depth) const;
		void						FindKNearestNeighbors(const vec_float_t &srcPoint, std::vector<vec_float_t> &nearPoints, const unsigned k, float &minDistance, kd_Box &minRegion, std::unordered_set<vec_float_t, CKDPointHasher> &nearSet, size_t Depth) const;

		vec_float_t					getPoint(void)			const { return m_point; }
		float						getSplitVal(void)		const { return m_splitVal; }
		kd_Box						getBoundingBox(void)	const { return isLeaf() ? std::make_pair(m_point, m_point) : m_boundingBox; }
		std::shared_ptr<CKDNode>	Left(void)				const { return m_Left; }
		std::shared_ptr<CKDNode>	Right(void)				const { return m_Right; }

		
	public:		
		std::weak_ptr<CKDNode>		m_Next;
		std::weak_ptr<CKDNode>		m_Prev;


	private:
		vec_float_t					m_point;
		float						m_splitVal;
		kd_Box						m_boundingBox;
		std::shared_ptr<CKDNode>	m_Left;
		std::shared_ptr<CKDNode>	m_Right;
	};
} }