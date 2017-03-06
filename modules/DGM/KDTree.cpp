#include "KDTree.h"
#include "mathop.h"

namespace DirectGraphicalModels
{

	float NthCoordMedian(std::vector<vec_float_t> &Points, size_t num)
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
	}


	CKDNode * CKDTree::createTree(Mat &vData)
	{
		if (vData.rows == 1) {
			CKDNode *root = new CKDNode(vData);
			return root;
		}
		else if (vData.rows == 2) {
			byte Median = 0; // NthCoordMedian(vData, 0);
			CKDNode *Left = new CKDNode(vData.row(0));
			CKDNode *Right = new CKDNode(vData.row(1));
			CKDNode *root = new CKDNode(Median, Left, Right);
			return root;
		}
		else {
			byte Median = 0; // NthCoordMedian(vData, 0);

			CKDNode *Left  = new CKDNode(vData.row(0));
			CKDNode *Right = new CKDNode(vData.row(1));
			CKDNode *root  = new CKDNode(Median, Left, Right);

			return root;
		}

		return NULL;
	}

}