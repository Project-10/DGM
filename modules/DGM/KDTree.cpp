#include "KDTree.h"
#include "random.h"
#include "parallel.h"
#include "mathop.h"

namespace DirectGraphicalModels
{
	namespace {
		template<typename T>
		pair_mat_t getBoundingBox(Mat &data)
		{
			Mat min, max;

			data.row(0).copyTo(min);
			data.row(0).copyTo(max);
			T * pMin = min.ptr<T>(0);
			T * pMax = max.ptr<T>(0);
			for (int y = 1; y < data.rows; y++) {				// samples
				T * pData = data.ptr<T>(y);
				for (int x = 0; x < data.cols; x++) {			// dimensions
					if (pMin[x] > pData[x]) pMin[x] = pData[x];
					if (pMax[x] < pData[x]) pMax[x] = pData[x];
				} // x: dimenstions
			} // y: samples

			return std::make_pair(min, max);
		}

		template<typename T>
		int getSplitDimension(pair_mat_t &boundingBox)
		{
			int		res = 0;
			Mat		diff = boundingBox.second - boundingBox.first;		// diff = max - min
			T		max = diff.at<T>(0, 0);
			Mat		maxMasc(1, diff.cols, CV_8UC1);						// binary maxv-alue masc 
			for (int x = 0; x < diff.cols; x++) {						// dimensions
				if (max == diff.at<T>(0, x)) {
					maxMasc.at<byte>(0, x) = 1;
				}
				else if (max > diff.at<T>(0, x)) {
					maxMasc.at<byte>(0, x) = 0;
				}
				else if (max < diff.at<T>(0, x)) {
					maxMasc.at<byte>(0, x) = 1;
					maxMasc(Rect(0, 0, x, 1)).setTo(0);
					max = diff.at<T>(0, x);
					res = x;
				}
			} // x: dimensions

			int nMaxs = countNonZero(maxMasc);
			if (nMaxs == 1) return res;

			// Randomly choose one of the maximums
			int n = random::u<int>(1, nMaxs);
			for (int x = 0; x < diff.cols; x++) {						// dimensions
				if (maxMasc.at<byte>(0, x) == 1) n--;
				if (n == 0) return x;
			} // x: dimensions
		}
	

	}

	 
	/// @todo Delete dublicated keys before building the tree
	void CKDTree::build(Mat &keys, Mat &values)
	{
		if (keys.empty()) {
			DGM_WARNING("The data is empty");
			return;
		}
		DGM_ASSERT_MSG(keys.type() == CV_8UC1, "Incorrect type of the keys");
		DGM_ASSERT_MSG(values.type() == CV_8UC1, "Incorrect type of the values");
		DGM_ASSERT_MSG(keys.rows == values.rows, "The amount of keys (%d) does not crrespond to the amount of values (%d)", keys.rows, values.rows);
		
		pair_mat_t boundingBox = getBoundingBox<byte>(keys);
		hconcat(keys, values, keys);							// keys = [keys; data]

		// TODO: delete dublicated entries
		//std::sort(keys.begin(), keys.end());
		//std::vector<vec_float_t>::iterator it = std::unique(keys.begin(), keys.end());
		//Points.resize(std::distance(keys.begin(), it));

		m_root = buildTree(keys, boundingBox);
	}

	std::shared_ptr<const CKDNode> CKDTree::findNearestNeighbor(const Mat &key) const
	{
		if (!m_root) {
			DGM_WARNING("The k-D tree is not built");
			return nullptr;
		}
		
		std::shared_ptr<const CKDNode> nearestNode = findNearestNode(key);
		float searchRadius = mathop::Euclidian<byte, float>(key, nearestNode->getKey()) + 0.5f;

		pair_mat_t searchBox;
		searchBox.first  = key - searchRadius;
		searchBox.second = key + searchRadius;

		m_root->findNearestNeighbor(key, searchBox, searchRadius, nearestNode);

		return nearestNode;
	}

	std::vector<std::shared_ptr<const CKDNode>> CKDTree::FindNearestNeighbors(const Mat &key, size_t k) const
	{
		const float searchRadius_extension = 2.0f;
		
		std::vector<std::shared_ptr<const CKDNode>> nearestNeighbors;
		nearestNeighbors.reserve(k);

		if (!m_root) {
			DGM_WARNING("The k-D tree is not built");
			return nearestNeighbors;
		}

		std::shared_ptr<const CKDNode> nearestNode = findNearestNode(key);
		nearestNeighbors.push_back(nearestNode);
		float searchRadius = mathop::Euclidian<byte, float>(key, nearestNode->getKey());
		if (k > 1) searchRadius *= searchRadius_extension;
		searchRadius += 0.5f;
		//if (k > 1) searchRadius = 255 * sqrtf(static_cast<float>(key.cols)); // infinity

		pair_mat_t searchBox;
		searchBox.first  = key - searchRadius;
		searchBox.second = key + searchRadius;

		m_root->findNearestNeighbors(key, k, searchBox, searchRadius, nearestNeighbors);
	
		return nearestNeighbors;
	}

	std::shared_ptr<const CKDNode> CKDTree::findNearestNode(const Mat &key) const
	{
		std::shared_ptr<CKDNode> node(m_root);

		while (!node->isLeaf()) {
			std::shared_ptr<CKDNode> n = std::static_pointer_cast<CKDNode>(node);
			if (key.at<byte>(0, n->getSplitDim()) < n->getSplitVal())	node = n->Left();
			else														node = n->Right();
		}

		return std::static_pointer_cast<CKDNode>(node);
	}


	// ----------------------------------------- Private -----------------------------------------
	// data_i = [key,val]: k + 1 entries 
	// left = [0; splitVal)
	// right = [splitVal; end]
	std::shared_ptr<CKDNode> CKDTree::buildTree(Mat &data, pair_mat_t &boundingBox)
	{
		if (data.rows == 1) {
			std::shared_ptr<CKDNode> res(new CKDNode(data(Rect(0, 0, data.cols - 1, 1)), data.at<byte>(0, data.cols - 1)));
			return res;
		}
		else if (data.rows == 2) {
			//pair_mat_t boundingBox = getBoundingBox<byte>(data);
			int	 splitDim = getSplitDimension<byte>(boundingBox);
			byte splitVal = (data.at<byte>(0, splitDim) + data.at<byte>(1, splitDim)) / 2;
			std::shared_ptr<CKDNode> left, right;
			if (data.at<byte>(0, splitDim) < data.at<byte>(1, splitDim)) {
				left  = std::shared_ptr<CKDNode>(new CKDNode(data.row(0)(Rect(0, 0, data.cols - 1, 1)), data.at<byte>(0, data.cols - 1)));
				right = std::shared_ptr<CKDNode>(new CKDNode(data.row(1)(Rect(0, 0, data.cols - 1, 1)), data.at<byte>(1, data.cols - 1)));
			}
			else {
				left  = std::shared_ptr<CKDNode>(new CKDNode(data.row(1)(Rect(0, 0, data.cols - 1, 1)), data.at<byte>(1, data.cols - 1)));
				right = std::shared_ptr<CKDNode>(new CKDNode(data.row(0)(Rect(0, 0, data.cols - 1, 1)), data.at<byte>(0, data.cols - 1)));
			}
			std::shared_ptr<CKDNode> res(new CKDNode(boundingBox, splitVal, splitDim, left, right));
			return res;
		}
		else {
			//pair_mat_t boundingBox = getBoundingBox<byte>(data);
			int	 splitDim = getSplitDimension<byte>(boundingBox);
			//if (splitDim == 1) printf("Achtung\n");
			parallel::sortRows<byte>(data, splitDim);
			int	 splitIdx = data.rows / 2;
			byte splitVal = data.at<byte>(splitIdx, splitDim);

			pair_mat_t boundingBoxLeft, boundingBoxRight;
			boundingBox.first.copyTo(boundingBoxLeft.first);
			boundingBox.second.copyTo(boundingBoxLeft.second);
			boundingBox.first.copyTo(boundingBoxRight.first);
			boundingBox.second.copyTo(boundingBoxRight.second);

			boundingBoxLeft.second.at<byte>(0, splitDim) = splitVal > 0 ? splitVal - 1 : 0;
			boundingBoxRight.first.at<byte>(0, splitDim) = splitVal;

			std::shared_ptr<CKDNode> left = buildTree(data(Rect(0, 0, data.cols, splitIdx)), boundingBoxLeft);
			std::shared_ptr<CKDNode> right = buildTree(data(Rect(0, data.rows / 2, data.cols, data.rows - data.rows / 2)), boundingBoxRight);
			std::shared_ptr<CKDNode> res(new CKDNode(boundingBox, splitVal, splitDim, left, right));
			return res;
		}
	}


}