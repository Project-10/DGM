#include "KDTree.h"
#include "random.h"
#include "parallel.h"
#include "mathop.h"

namespace DirectGraphicalModels
{

	
	/*float NthCoordMedian(std::vector<vec_float_t> &Points, size_t num)
	{
		sort(Points.begin(), Points.end(), [num](vec_float_t &A, vec_float_t &B) {
			return A[num] < B[num];
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
	}*/

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
	
		std::shared_ptr<CKDNode> createTree1(Mat &data, pair_mat_t &boundingBox)
		{
			if (data.rows == 1) {
				std::shared_ptr<CKDNode> res(new CKDNode(data, 77));
				return res;
			}
			else if (data.rows == 2) {
				//pair_mat_t boundingBox = getBoundingBox<byte>(data);
				int	 splitDim = getSplitDimension<byte>(boundingBox);
				byte splitVal = (data.at<byte>(0, splitDim) + data.at<byte>(1, splitDim)) / 2;
				std::shared_ptr<CKDNode> left, right;
				if (data.at<byte>(0, splitDim) < data.at<byte>(1, splitDim)) {
					left = std::shared_ptr<CKDNode>(new CKDNode(data.row(0), 77));
					right = std::shared_ptr<CKDNode>(new CKDNode(data.row(1), 77));
				}
				else {
					left = std::shared_ptr<CKDNode>(new CKDNode(data.row(1), 77));
					right = std::shared_ptr<CKDNode>(new CKDNode(data.row(0), 77));
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

				std::shared_ptr<CKDNode> left = createTree1(data(Rect(0, 0, data.cols, splitIdx)), boundingBoxLeft);
				std::shared_ptr<CKDNode> right = createTree1(data(Rect(0, data.rows / 2, data.cols, data.rows - data.rows / 2)), boundingBoxRight);
				std::shared_ptr<CKDNode> res(new CKDNode(boundingBox, splitVal, splitDim, left, right));
				return res;
			}
		}
	}

	 
	// left = [0; splitVal)
	// right = [splitVal; end]
	std::shared_ptr<CKDNode> CKDTree::createTree(Mat &data)
	{
		if (data.empty()) {
			DGM_WARNING("The data is empty");
			return nullptr;
		}

		DGM_ASSERT_MSG(data.type() == CV_8UC1, "Incorrect type of the data");
		
		pair_mat_t boundingBox = getBoundingBox<byte>(data);
		return createTree1(data, boundingBox);
	}

	float CKDTree::findNearestNeighbor(std::shared_ptr<CKDNode> root, Mat &key)
	{
		std::shared_ptr<CKDNode> nearestNode = findNearestNode(root, key);
		Mat						 nearestKey	 = nearestNode->getKey();
		float					 minDistance = mathop::Euclidian<byte, float>(key, nearestKey) + 0.5f;

		pair_mat_t searchBox;
		searchBox.first  = key - minDistance;
		searchBox.second = key + minDistance;

		root->findNearestNeighbor(key, minDistance, searchBox);
//		m_Root->FindNearestNeighbor(srcPoint, nearPoint, minDistance, minBox, 0);

		return minDistance;
	}

	std::shared_ptr<CKDNode> CKDTree::findNearestNode(std::shared_ptr<CKDNode> root, Mat &key)
	{
		std::shared_ptr<CKDNode> node(root);

		while (!node->isLeaf()) {
			std::shared_ptr<CKDNode> n = std::static_pointer_cast<CKDNode>(node);
			if (key.at<byte>(0, n->getSplitDim()) < n->getSplitVal())	node = n->Left();
			else														node = n->Right();
		}

		return std::static_pointer_cast<CKDNode>(node);
	}


	// ----------------------------------------- Private -----------------------------------------
	


}