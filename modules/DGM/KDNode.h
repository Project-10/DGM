#pragma once

#include "types.h"
#include <unordered_set>

namespace DirectGraphicalModels
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
		virtual bool		isInternal(void) const = 0;
		virtual kd_Box		getBoundingBox(void) const = 0;
		virtual size_t		getTreeHeight(void) const = 0;
		virtual size_t		getNodeCount(bool withInternalNodes) const = 0;
		virtual void		SearchKdTree(const kd_Box &searchBox, std::vector<vec_float_t> &Points, size_t Depth = 0) const = 0;
		virtual void		FindNearestNeighbor(const vec_float_t &srcPoint, vec_float_t &nearPoint, float &minDistance, kd_Box &minRegion, size_t Depth = 0) const = 0;
		virtual void		FindKNearestNeighbors(const vec_float_t &srcPoint, std::vector<vec_float_t> &nearPoints, const unsigned k, float &minDistance, kd_Box &minRegion, std::unordered_set<vec_float_t, CKDPointHasher> &nearSet, size_t Depth) const = 0;
	};

	class kd_internal_node : public CKDNode
	{
	public:
		kd_internal_node(const float splitVal, kd_Box &boundingBox, std::shared_ptr<CKDNode> Left)
			: m_splitVal(splitVal)
			, m_boundingBox(boundingBox)
			, m_Left(Left)
			, m_Right(nullptr)
		{}

		kd_internal_node(const float splitVal, kd_Box &boundingBox, std::shared_ptr<CKDNode> Left, std::shared_ptr<CKDNode> Right)
			: m_splitVal(splitVal)
			, m_boundingBox(boundingBox)
			, m_Left(Left)
			, m_Right(Right)
		{}

		virtual bool		isInternal(void) const { return true; }
		virtual kd_Box		getBoundingBox(void) const { return m_boundingBox; }
		virtual size_t		getTreeHeight(void) const
		{
			return 1 + max(
				m_Left->getTreeHeight(),
				m_Right != nullptr ? m_Right->getTreeHeight() : 0
			);
		}
		virtual size_t		getNodeCount(bool withInternalNodes) const
		{
			return (withInternalNodes ? 1 : 0) +
				m_Left->getNodeCount(withInternalNodes) +
				(m_Right != nullptr ? m_Right->getNodeCount(withInternalNodes) : 0);
		}
		virtual void		SearchKdTree(const kd_Box &searchBox, std::vector<vec_float_t> &Points, size_t Depth) const;
		virtual void		FindNearestNeighbor(const vec_float_t &srcPoint, vec_float_t &nearPoint, float &minDistance, kd_Box &minRegion, size_t Depth) const;
		virtual void		FindKNearestNeighbors(const vec_float_t &srcPoint, std::vector<vec_float_t> &nearPoints, const unsigned k, float &minDistance, kd_Box &minRegion, std::unordered_set<vec_float_t, CKDPointHasher> &nearSet, size_t Depth) const;

		float				getSplitVal(void) const { return m_splitVal; }

		std::shared_ptr<CKDNode> Left()  const { return m_Left; }
		std::shared_ptr<CKDNode> Right() const { return m_Right; }


	private:
		float						m_splitVal;
		kd_Box						m_boundingBox;
		std::shared_ptr<CKDNode>	m_Left;
		std::shared_ptr<CKDNode>	m_Right;
	};

	class kd_leaf_node : public CKDNode
	{
	public:
		kd_leaf_node(const vec_float_t &Point) : m_pointCoords(Point) { }

		virtual bool		isInternal(void) const { return false; }
		virtual kd_Box		getBoundingBox(void) const { return std::make_pair(m_pointCoords, m_pointCoords); }
		virtual size_t		getTreeHeight(void) const { return 1; }
		virtual size_t		getNodeCount(bool) const { return 1; }
		virtual void		SearchKdTree(const kd_Box &searchBox, std::vector<vec_float_t> &Points, size_t Depth) const;
		virtual void		FindNearestNeighbor(const vec_float_t &srcPoint, vec_float_t &nearPoint, float &minDistance, kd_Box &minRegion, size_t Depth) const;
		virtual void		FindKNearestNeighbors(const vec_float_t &srcPoint, std::vector<vec_float_t> &nearPoints, const unsigned k, float &minDistance, kd_Box &minRegion, std::unordered_set<vec_float_t, CKDPointHasher> &nearSet, size_t Depth) const;

		vec_float_t			getPointCoords(void) const { return m_pointCoords; }


	public:
		std::weak_ptr<kd_leaf_node> m_Next;
		std::weak_ptr<kd_leaf_node>	m_Prev;


	private:
		vec_float_t			m_pointCoords;
	};

}