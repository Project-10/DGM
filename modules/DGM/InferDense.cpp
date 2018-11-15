#include "InferDense.h"
#include "densecrf/edgepotential.h"

namespace DirectGraphicalModels
{
	namespace {
		template<typename T>
		void normalize(const Mat &src, Mat dst) {
			if(dst.empty()) dst = Mat(src.size(), src.type());
			for (int y = 0; y < src.rows; y++) {
				const T *pSrc = src.ptr<T>(y);
				T *pDst = dst.ptr<T>(y);
				T sum = 0;
				for (int x = 0; x < src.cols; x++) sum += pSrc[x];
				if (sum > DBL_EPSILON)
					for (int x = 0; x < src.cols; x++) pDst[x] = pSrc[x] / sum;
			} // y
		}
		
		template<typename T>
		void myexp(const Mat &src, Mat &dst)
		{
			for (int y = 0; y < src.rows; y++) {             
				const T *pSrc  = src.ptr<float>(y);
				T *pDst = dst.ptr<float>(y);

				// Find the max and subtract it so that the exp doesn't explodeh
				T max = pSrc[0];
				for (int x = 1; x < src.cols; x++)
					if (pSrc[x] > max) max = pSrc[x];

				for (int x = 0; x < src.cols; x++)
					pDst[x] = expf(pSrc[x] - max);
			} // y
		}
	}
	
	void CInferDense::infer(unsigned int nIt)
	{
		// ====================================== Initialization ======================================
		const int rows = getGraphDense().getNodes().rows;
		const int cols = getGraphDense().getNodes().cols;

		Mat tmp;
		Mat next = Mat(rows, cols, CV_32FC1);

		Mat nodes0 = getGraphDense().getNodes().clone();

		// =================================== Calculating potentials ==================================	
		for (unsigned int i = 0; i < nIt; i++) {
#ifdef DEBUG_PRINT_INFO
            if (i == 0) printf("\n");
            if (i % 5 == 0) printf("--- It: %d ---\n", i);
#endif
			normalize<float>(getGraphDense().getNodes(), getGraphDense().getNodes());	
			next.setTo(1);
			// Add up all pairwise potentials
            for (auto &edgePotModel : getGraphDense().getEdgeModels()) {
				edgePotModel->apply(getGraphDense().getNodes(), tmp);					// tmp = f(pot_i)
				exp(tmp, tmp);
				multiply(next, tmp, next);												// next *= exp(tmp)
			}

			multiply(nodes0, next, getGraphDense().getNodes());							// pot_(i+1) = pot_0 * next
		} // iter
	}
}
