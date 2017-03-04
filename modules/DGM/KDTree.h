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

		void clear(void) { m_Root.reset(); m_firstLeaf.reset(); }

		friend class kd_tree_iterator;
		typedef typename kd_tree_iterator iterator;

		kd_tree_iterator end();
		kd_tree_iterator begin();

		void			insert(std::vector<vec_float_t> &Points);
		void			search(const vec_float_t &minPoint, const vec_float_t &maxPoint, std::vector<vec_float_t> &Points) const
		{
			Points.clear();

			if (m_Root == nullptr)
				return;

			kd_Box sorted;

			for (size_t coord = 0; coord < minPoint.size(); coord++) {
				sorted.first[coord] = min(minPoint[coord], maxPoint[coord]);
				sorted.second[coord] = max(minPoint[coord], maxPoint[coord]);
			}

			m_Root->SearchKdTree(sorted, Points);
		}
		bool			FindNearestNeighbor(const vec_float_t &srcPoint, vec_float_t &nearPoint) const
		{
			bool retVal = (m_Root != nullptr);

			if (!m_Root)
				std::fill(nearPoint.begin(), nearPoint.end(), std::numeric_limits<float>::quiet_NaN());
			else
			{
				float minDistance = ApproxNearestNeighborDistance(srcPoint);

				kd_Box minBox;

				for (size_t coord = 0; coord < srcPoint.size(); coord++) {
					minBox.first[coord] = srcPoint[coord] - minDistance;
					minBox.second[coord] = srcPoint[coord] + minDistance;
				}

				m_Root->FindNearestNeighbor(srcPoint, nearPoint, minDistance, minBox);
			}

			return retVal;
		}
		bool			FindKNearestNeighbors(const vec_float_t &srcPoint, std::vector<vec_float_t> &nearPoints, const unsigned k) const;
		
		size_t			getTreeHeight(void) const { return this->m_Root != nullptr ? m_Root->getTreeHeight() : 0; }
		size_t			getNodeCount(bool withInternalNodes = false) const { return m_Root != nullptr ? m_Root->getNodeCount(withInternalNodes) : 0; }
		
		void			PrintTree(void) const { PrintTree(m_Root); }


	protected:
		// --------------------------------
		std::shared_ptr<CKDNode>		m_Root;
		std::weak_ptr<kd_leaf_node>		m_firstLeaf;
		// -----------------------------------------------------------------------------------------------------

		// The routine has a desired side effect of sorting Points
		float							NthCoordMedian(std::vector<vec_float_t> &Points, size_t num)
		{
			sort(Points.begin(), Points.end(), [num](vec_float_t &A, vec_float_t &B) { 
				size_t K = A.size();
				return A[num % K] < B[num % K]; 
			});

			float Median = Points[Points.size() / 2][num];

			if (Points.size() % 2 == 0)
				Median = (Median + Points[Points.size() / 2 - 1][num]) / 2.0f;

			if (Median == Points[Points.size() - 1][num] && Median != Points[0][num]) {
				size_t index = Points.size() / 2;
				while (Median == Points[--index][num]);
				Median = (Median + Points[index][num]) / 2.0f;
			}

			return Median;
		}
		std::shared_ptr<CKDNode>		CreateTree(std::vector<vec_float_t> &Points, std::shared_ptr<kd_leaf_node> &Last_Leaf, size_t Depth = 0)
		{
			if (Points.size() == 1)
			{
				std::shared_ptr<kd_leaf_node> retNode(new kd_leaf_node(Points[0]));

				if (Last_Leaf)
				{
					Last_Leaf->m_Next = retNode;
					retNode->m_Prev = Last_Leaf;
				}
				else
					m_firstLeaf = retNode;

				Last_Leaf = retNode;

				return retNode;
			}
			else if (Points.size() == 2)
			{
				size_t K = Points[0].size();
				if (Points[0][Depth % K] == Points[1][Depth % K]) {
					std::shared_ptr<CKDNode> Left = CreateTree(Points, Last_Leaf, Depth + 1);
					std::shared_ptr<kd_internal_node> retNode(new kd_internal_node(Points[0][Depth%K], Left->getBoundingBox(), Left));
					return retNode;
				} else {
					if (Points[0][Depth%K] > Points[1][Depth%K])
						swap(Points[0], Points[1]);

					std::shared_ptr<kd_leaf_node> Left(new kd_leaf_node(Points[0])), Right(new kd_leaf_node(Points[1]));

					if (Last_Leaf) {
						Last_Leaf->m_Next = Left;
						Left->m_Prev = Last_Leaf;
					}
					else
						m_firstLeaf = Left;

					Left->m_Next = Right;
					Right->m_Prev = Left;

					Last_Leaf = Right;

					kd_Box boundingBox;

					for (size_t i = 0; i < Points[0].size(); i++) {
						boundingBox.first[i] = min(Points[0][i], Points[1][i]);
						boundingBox.second[i] = max(Points[0][i], Points[1][i]);
					}

					std::shared_ptr<kd_internal_node> retNode(new
						kd_internal_node((Points[0][Depth%K] + Points[1][Depth%K]) / 2.0f, boundingBox, Left, Right));

					return retNode;
				}
			}
			else
			{
				size_t K = Points[0].size();
				float Median = NthCoordMedian(Points, Depth % K);

				std::vector<vec_float_t> subtreePoints;
				std::shared_ptr<CKDNode> Left, Right;

				subtreePoints.reserve(Points.size() / 2);

				for (size_t index = 0; index < Points.size() && Points[index][Depth%K] <= Median; index++)
					subtreePoints.push_back(Points[index]);

				if (subtreePoints.size() > 0)
					Left = CreateTree(subtreePoints, Last_Leaf, Depth + 1);

				size_t insertedPoints = subtreePoints.size();
				size_t remainedPoints = Points.size() - subtreePoints.size();

				subtreePoints.resize(remainedPoints); subtreePoints.shrink_to_fit();

				for (size_t index = insertedPoints; index < Points.size(); index++)
					subtreePoints[index - insertedPoints] = Points[index];

				if (subtreePoints.size() > 0)
					Right = CreateTree(subtreePoints, Last_Leaf, Depth + 1);

				subtreePoints.resize(0); subtreePoints.shrink_to_fit();

				kd_Box boundingBox;

				if (Right)
				{
					kd_Box lbb = Left->getBoundingBox();
					kd_Box rbb = Right->getBoundingBox();

					for (size_t i = 0; i < lbb.first.size(); i++) {
						boundingBox.first[i] = min(lbb.first[i], rbb.first[i]);
						boundingBox.second[i] = max(lbb.second[i], rbb.second[i]);
					}
				}
				else
					boundingBox = Left->getBoundingBox();

				std::shared_ptr<kd_internal_node> retNode(new kd_internal_node(Median, boundingBox, Left, Right));

				return retNode;
			}
		}
		std::shared_ptr<kd_leaf_node>	ApproxNearestNeighborNode(const vec_float_t &srcPoint) const
		{
			unsigned int Depth = 0;
			std::shared_ptr<CKDNode> Node(m_Root);

			while (Node->isInternal()) {
				std::shared_ptr<kd_internal_node> iNode = std::static_pointer_cast<kd_internal_node>(Node);

				if (srcPoint[Depth++ % srcPoint.size()] <= iNode->getSplitVal() || iNode->Right() == nullptr)
					Node = iNode->Left();
				else
					Node = iNode->Right();
			}

			std::shared_ptr<kd_leaf_node> lNode = std::static_pointer_cast<kd_leaf_node>(Node);

			return lNode;
		}
		float							ApproxNearestNeighborDistance(const vec_float_t &srcPoint) const;
		void							PrintTree(std::shared_ptr<CKDNode> node, unsigned int depth = 0) const
		{
			for (unsigned i = 0; i < depth; i++)
				std::cout << " ";

			if (node == nullptr)
				std::cout << "null" << std::endl;
			else
			{
				if (node->isInternal()) {
					std::shared_ptr<kd_internal_node> iNode = std::static_pointer_cast<kd_internal_node>(node);

					//std::cout << "Split val is " << iNode->getSplitVal() << " for axis #" << depth % K + 1 << std::endl;

					PrintTree(iNode->Left(), depth + 1);
					PrintTree(iNode->Right(), depth + 1);
				} else {
					std::shared_ptr<kd_leaf_node> lNode = std::static_pointer_cast<kd_leaf_node>(node);

					std::cout << "Point is (";

					for (auto p : lNode->getPointCoords())	
						std::cout << p << " ";

					std::cout << ")" << std::endl;
				}
			}
		}
	};


	//
	// Iterator implementation
	//
	class kd_tree_iterator : public std::iterator<std::output_iterator_tag, void, void, void, void>
	{
	public:
		kd_tree_iterator() {}
		kd_tree_iterator(std::shared_ptr<typename kd_leaf_node> node) : nodePtr(node) {}

		friend bool operator== (const kd_tree_iterator& lhs, const kd_tree_iterator& rhs);
		friend bool operator!= (const kd_tree_iterator& lhs, const kd_tree_iterator& rhs);

		typename vec_float_t &operator*() { return this->nodePtr->getPointCoords(); }

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
		std::shared_ptr<typename kd_leaf_node> nodePtr;
	};


}