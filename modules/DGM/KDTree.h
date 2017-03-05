// K-Dimensional Tree class interface
// Copied from http://codereview.stackexchange.com/questions/110225/k-d-tree-implementation-in-c11
#pragma once

#include "types.h"

#include "KDNode.h"

namespace DirectGraphicalModels
{
	class kd_tree_iterator;

	class CKDTree
	{
	public:
		CKDTree(void) { }
		CKDTree(std::vector<vec_float_t> &Points) { insert(Points); }
		CKDTree(const CKDTree &obj) = delete;

		bool operator==(const CKDTree rhs) = delete;
		bool operator=(const CKDTree rhs) = delete;

		void	clear(void);

		friend class kd_tree_iterator;
		typedef typename kd_tree_iterator iterator;

		kd_tree_iterator end();
		kd_tree_iterator begin();

		void	insert(std::vector<vec_float_t> &Points);
		void	search(const vec_float_t &minPoint, const vec_float_t &maxPoint, std::vector<vec_float_t> &Points) const;
		bool	FindNearestNeighbor(const vec_float_t &srcPoint, vec_float_t &nearPoint) const;
		bool	FindKNearestNeighbors(const vec_float_t &srcPoint, std::vector<vec_float_t> &nearPoints, const unsigned k) const;
		
		size_t	getTreeHeight(void)								const { return this->m_Root != nullptr ? m_Root->getTreeHeight() : 0; }
		size_t	getNodeCount(bool withInternalNodes = false)	const { return m_Root != nullptr ? m_Root->getNodeCount(withInternalNodes) : 0; }
		void	PrintTree(void)									const { PrintTree(m_Root); }


	protected:
		// --------------------------------
		std::shared_ptr<CKDNode>		m_Root;
		std::weak_ptr<CKDNode>			m_firstLeaf;
		// -----------------------------------------------------------------------------------------------------

		// The routine has a desired side effect of sorting Points
		float							NthCoordMedian(std::vector<vec_float_t> &Points, size_t num);
		std::shared_ptr<CKDNode>		CreateTree(std::vector<vec_float_t> &Points, std::shared_ptr<CKDNode> &Last_Leaf, size_t Depth = 0);
		std::shared_ptr<CKDNode>		ApproxNearestNeighborNode(const vec_float_t &srcPoint) const;
		float							ApproxNearestNeighborDistance(const vec_float_t &srcPoint) const;
		void							PrintTree(std::shared_ptr<CKDNode> node, size_t depth = 0) const;
	};


	//
	// Iterator implementation
	//
	class kd_tree_iterator : public std::iterator<std::output_iterator_tag, void, void, void, void>
	{
	public:
		kd_tree_iterator() {}
		kd_tree_iterator(std::shared_ptr<typename CKDNode> node) : nodePtr(node) {}

		friend bool operator== (const kd_tree_iterator& lhs, const kd_tree_iterator& rhs);
		friend bool operator!= (const kd_tree_iterator& lhs, const kd_tree_iterator& rhs);

		typename vec_float_t &operator*() { return this->nodePtr->getPoint(); }

		kd_tree_iterator& operator++()
		{
			this->nodePtr = this->nodePtr->m_Next.lock();
			return *this;
		}

		kd_tree_iterator operator++(int)
		{
			kd_tree_iterator tmp(*this);
			operator++();
			return tmp;
		}

	private:
		std::shared_ptr<typename CKDNode> nodePtr;
	};


}