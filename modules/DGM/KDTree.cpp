#include "KDTree.h"
#include "mathop.h"

namespace DirectGraphicalModels
{
	bool isValid(vec_float_t &point) {
		for (float p : point)
			if (isnan(p) ||
				p == std::numeric_limits<float>::infinity() ||
				p == -std::numeric_limits<float>::infinity())
				return false;
		return true;
	}

	bool operator== (const kd_tree_iterator &lhs, const kd_tree_iterator &rhs)
	{
		return (lhs.nodePtr == rhs.nodePtr);
	}

	bool operator!= (const kd_tree_iterator &lhs, const kd_tree_iterator &rhs)
	{
		return !(lhs == rhs);
	}

	// -------------------------------------------------------------

	void CKDTree::insert(std::vector<vec_float_t> &Points)
	{
		clear();

		for (size_t i = Points.size() - 1; i >= 0; i--)
			if (!isValid(Points[i]))
				Points.erase(Points.begin() + i);

		if (Points.size() > 0) {
			sort(Points.begin(), Points.end());
			std::vector<vec_float_t>::iterator it = unique(Points.begin(), Points.end());
			Points.resize(distance(Points.begin(), it));

			std::shared_ptr<kd_leaf_node> dummyLeaf;

			m_Root = CreateTree(Points, dummyLeaf);
		}
	}

	bool CKDTree::FindKNearestNeighbors(const vec_float_t &srcPoint, std::vector<vec_float_t> &nearPoints, const unsigned k) const
	{
		nearPoints.clear();
		nearPoints.reserve(k);

		if (!m_Root) return false;

		std::shared_ptr<kd_leaf_node> nNode = ApproxNearestNeighborNode(srcPoint),
			pNode = nNode->m_Prev.lock();

		nearPoints.push_back(nNode->getPointCoords());

		nNode = nNode->m_Next.lock();

		while (nearPoints.size() < k && (nNode || pNode))
		{
			if (nNode) {
				nearPoints.push_back(nNode->getPointCoords());
				nNode = nNode->m_Next.lock();
			}

			if (pNode && nearPoints.size() < k) {
				nearPoints.push_back(pNode->getPointCoords());
				pNode = pNode->m_Prev.lock();
			}
		}

		sort(nearPoints.begin(), nearPoints.end(), [srcPoint](vec_float_t &A, vec_float_t &B) { return mathop::Euclidian(srcPoint, A) < mathop::Euclidian(srcPoint, B); });

		float minDistance;

		if (nearPoints.size() < k)
		{
			vec_float_t infinityPoint;
			std::fill(infinityPoint.begin(), infinityPoint.end(), std::numeric_limits<float>::infinity());

			nearPoints.resize(k, infinityPoint);

			minDistance = std::numeric_limits<float>::infinity();
		}
		else
			minDistance = mathop::Euclidian(srcPoint, nearPoints[k - 1]);

		kd_Box MinBox;

		for (size_t i = 0; i < srcPoint.size(); i++) {
			MinBox.first[i] = srcPoint[i] - minDistance;
			MinBox.second[i] = srcPoint[i] + minDistance;
		}

		std::unordered_set<vec_float_t, CKDPointHasher> nearSet(nearPoints.begin(), nearPoints.end());

		m_Root->FindKNearestNeighbors(srcPoint, nearPoints, k, minDistance, MinBox, nearSet, 0);

		for (signed i = k - 1; i > 0 && !isValid(nearPoints[i]); i--)
			nearPoints.erase(nearPoints.begin() + i);

		return true;
	}

	float CKDTree::ApproxNearestNeighborDistance(const vec_float_t &srcPoint) const
	{
		std::shared_ptr<kd_leaf_node> node = ApproxNearestNeighborNode(srcPoint);
		return mathop::Euclidian(srcPoint, node->getPointCoords());
	}

	kd_tree_iterator CKDTree::end()
	{
		CKDTree::iterator retVal;
		return retVal;
	}

	kd_tree_iterator CKDTree::begin()
	{
		if (!this->m_Root)	return end();
		else				return m_firstLeaf.lock();
	}

}