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

		int getSplitDimension(Mat &data)
		{
			if (data.empty()) {
				DGM_WARNING("The data is empty");
				return -1;
			}

			DGM_ASSERT_MSG(data.type() == CV_8UC1, "Incorrect type of the data");

			pair_mat_t boundingBox = getBoundingBox<byte>(data);

			int		res = 0;
			Mat		diff = boundingBox.second - boundingBox.first;	// diff = max - min
			byte	max = diff.at<byte>(0, 0);
			Mat		maxMasc(1, data.cols, CV_8UC1);						// binary maxv-alue masc 
			for (int x = 0; x < data.cols; x++) {						// dimensions
				if (max == diff.at<byte>(0, x)) {
					maxMasc.at<byte>(0, x) = 1;
				}
				else if (max > diff.at<byte>(0, x)) {
					maxMasc.at<byte>(0, x) = 0;
				}
				else if (max < diff.at<byte>(0, x)) {
					maxMasc.at<byte>(0, x) = 1;
					maxMasc(Rect(0, 0, x, 1)).setTo(0);
					max = diff.at<byte>(0, x);
					res = x;
				}
			} // x: dimensions

			int nMaxs = countNonZero(maxMasc);
			if (nMaxs == 1) return res;

			// Randomly choose one of the maximums
			int n = random::u<int>(1, nMaxs);
			for (int x = 0; x < data.cols; x++) {						// dimensions
				if (maxMasc.at<byte>(0, x) == 1) n--;
				if (n == 0) return x;
			} // x: dimensions
		}
	}

	 
	std::shared_ptr<CKDNode> CKDTree::createTree(Mat &data)
	{
		if (data.rows == 1) {
			std::shared_ptr<CKDNode> res(new CKDNode(data, 77));
			return res;
		}
		else if (data.rows == 2) {
			int	 splitDim = getSplitDimension(data);
			byte splitVal = (data.at<byte>(0, splitDim) + data.at<byte>(1, splitDim)) / 2;
			std::shared_ptr<CKDNode> left, right;
			if (data.at<byte>(0, splitDim) < data.at<byte>(1, splitDim)) {
				left  = std::shared_ptr<CKDNode>(new CKDNode(data.row(0), 77));
				right = std::shared_ptr<CKDNode>(new CKDNode(data.row(1), 77));
			}
			else {
				left  = std::shared_ptr<CKDNode>(new CKDNode(data.row(1), 77));
				right = std::shared_ptr<CKDNode>(new CKDNode(data.row(0), 77));
			}
			std::shared_ptr<CKDNode> res(new CKDNode(splitVal, splitDim, left, right));
			return res;
		}
		else {
			int	 splitDim = getSplitDimension(data);	// Possible optimization: do not re-calculate the bounding box
			//if (splitDim == 1) printf("Achtung\n");
			parallel::sortRows<byte>(data, splitDim);
			byte splitVal = data.at<byte>(data.rows / 2, splitDim);

			std::shared_ptr<CKDNode> left =  createTree(data(Rect(0, 0, data.cols, data.rows / 2)));
			std::shared_ptr<CKDNode> right = createTree(data(Rect(0, data.rows / 2, data.cols, data.rows - data.rows / 2)));
			std::shared_ptr<CKDNode> res(new CKDNode(splitVal, splitDim, left, right));
			return res;
		}
	}

}