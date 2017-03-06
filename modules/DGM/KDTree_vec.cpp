#include "KDTree_vec.h"
#include "mathop.h"

namespace DirectGraphicalModels { namespace veccc 
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

	void CKDTree::clear(void) 
	{ 
		m_Root.reset(); 
		m_firstLeaf.reset(); 
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

	void CKDTree::insert(std::vector<vec_float_t> &Points)
	{
		clear();

		// delete non-valid points
		for (size_t i = Points.size() - 1; i >= 0; i--)
			if (!isValid(Points[i]))
				Points.erase(Points.begin() + i);

		if (Points.size() > 0) {
			sort(Points.begin(), Points.end());
			std::vector<vec_float_t>::iterator it = unique(Points.begin(), Points.end());
			Points.resize(distance(Points.begin(), it));
			std::shared_ptr<CKDNode> dummyLeaf;
			m_Root = CreateTree(Points, dummyLeaf);
		}
	}

	void CKDTree::search(const vec_float_t &minPoint, const vec_float_t &maxPoint, std::vector<vec_float_t> &Points) const
	{
		Points.clear();

		if (m_Root == nullptr) return;

		kd_Box sorted;

		for (size_t coord = 0; coord < minPoint.size(); coord++) {
			sorted.first[coord] = min(minPoint[coord], maxPoint[coord]);
			sorted.second[coord] = max(minPoint[coord], maxPoint[coord]);
		}

		m_Root->SearchKdTree(sorted, Points, 0);
	}

	bool CKDTree::FindNearestNeighbor(const vec_float_t &srcPoint, vec_float_t &nearPoint) const
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

			m_Root->FindNearestNeighbor(srcPoint, nearPoint, minDistance, minBox, 0);
		}

		return retVal;
	}

	bool CKDTree::FindKNearestNeighbors(const vec_float_t &srcPoint, std::vector<vec_float_t> &nearPoints, const unsigned k) const
	{
		nearPoints.clear();
		nearPoints.reserve(k);

		if (!m_Root) return false;

		std::shared_ptr<CKDNode> nNode = ApproxNearestNeighborNode(srcPoint),
			pNode = nNode->m_Prev.lock();

		nearPoints.push_back(nNode->getPoint());

		nNode = nNode->m_Next.lock();

		while (nearPoints.size() < k && (nNode || pNode))
		{
			if (nNode) {
				nearPoints.push_back(nNode->getPoint());
				nNode = nNode->m_Next.lock();
			}

			if (pNode && nearPoints.size() < k) {
				nearPoints.push_back(pNode->getPoint());
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

	// =============================== Protected ===============================

	float CKDTree::NthCoordMedian(std::vector<vec_float_t> &Points, size_t num)
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

	std::shared_ptr<CKDNode> CKDTree::CreateTree(std::vector<vec_float_t> &Points, std::shared_ptr<CKDNode> &Last_Leaf, size_t Depth)
	{
		if (Points.size() == 1) {
			std::shared_ptr<CKDNode> retNode(new CKDNode(Points[0]));

			if (Last_Leaf) {
				Last_Leaf->m_Next = retNode;
				retNode->m_Prev = Last_Leaf;
			} else
				m_firstLeaf = retNode;

			Last_Leaf = retNode;

			return retNode;
		} else if (Points.size() == 2) {
			size_t K = Points[0].size();
			if (Points[0][Depth % K] == Points[1][Depth % K]) {
				std::shared_ptr<CKDNode> Left = CreateTree(Points, Last_Leaf, Depth + 1);
				std::shared_ptr<CKDNode> retNode(new CKDNode(Points[0][Depth%K], Left->getBoundingBox(), Left));
				return retNode;
			}
			else {
				if (Points[0][Depth%K] > Points[1][Depth%K])
					swap(Points[0], Points[1]);

				std::shared_ptr<CKDNode> Left(new CKDNode(Points[0])), Right(new CKDNode(Points[1]));

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

				std::shared_ptr<CKDNode> retNode(new CKDNode((Points[0][Depth%K] + Points[1][Depth%K]) / 2.0f, boundingBox, Left, Right));

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

			if (Right) {
				kd_Box lbb = Left->getBoundingBox();
				kd_Box rbb = Right->getBoundingBox();

				for (size_t i = 0; i < lbb.first.size(); i++) {
					boundingBox.first[i] = min(lbb.first[i], rbb.first[i]);
					boundingBox.second[i] = max(lbb.second[i], rbb.second[i]);
				}
			}
			else
				boundingBox = Left->getBoundingBox();

			std::shared_ptr<CKDNode> retNode(new CKDNode(Median, boundingBox, Left, Right));

			return retNode;
		}
	}

	std::shared_ptr<CKDNode> CKDTree::ApproxNearestNeighborNode(const vec_float_t &srcPoint) const
	{
		unsigned int Depth = 0;
		std::shared_ptr<CKDNode> Node(m_Root);

		while (!Node->isLeaf()) {
			std::shared_ptr<CKDNode> iNode = std::static_pointer_cast<CKDNode>(Node);

			if (srcPoint[Depth++ % srcPoint.size()] <= iNode->getSplitVal() || iNode->Right() == nullptr)
				Node = iNode->Left();
			else
				Node = iNode->Right();
		}

		std::shared_ptr<CKDNode> lNode = std::static_pointer_cast<CKDNode>(Node);

		return lNode;
	}

	float CKDTree::ApproxNearestNeighborDistance(const vec_float_t &srcPoint) const
	{
		std::shared_ptr<CKDNode> node = ApproxNearestNeighborNode(srcPoint);
		return mathop::Euclidian(srcPoint, node->getPoint());
	}

	void CKDTree::PrintTree(std::shared_ptr<CKDNode> node, size_t depth) const
	{
		for (size_t i = 0; i < depth; i++) std::cout << " ";
		
		if (node == nullptr) std::cout << "null" << std::endl;
		else {
			if (!node->isLeaf()) {
				std::shared_ptr<CKDNode> iNode = std::static_pointer_cast<CKDNode>(node);

				//std::cout << "Split val is " << iNode->getSplitVal() << " for axis #" << depth % K + 1 << std::endl;

				PrintTree(iNode->Left(), depth + 1);
				PrintTree(iNode->Right(), depth + 1);
			}
			else {
				std::shared_ptr<CKDNode> lNode = std::static_pointer_cast<CKDNode>(node);

				std::cout << "Point is (";
				for (auto p : lNode->getPoint())	std::cout << p << " ";
				std::cout << ")" << std::endl;
			}
		}
	}

} }