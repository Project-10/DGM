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

				// Find the max and subtract it so that the exp doesn't explode
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
		Mat nodePotentials	= getGraphDense().getNodePotentials();
		Mat	nodePotentials0	= nodePotentials.clone();
		Mat	temp			= Mat(nodePotentials.size(), nodePotentials.type());
		Mat	tmp;

		// =================================== Calculating potentials ==================================	
		for (unsigned int i = 0; i < nIt; i++) {
#ifdef DEBUG_PRINT_INFO
			if (i == 0) printf("\n");
			if (i % 5 == 0) printf("--- It: %d ---\n", i);
#endif
			normalize<float>(nodePotentials, nodePotentials);
			
			// Add up all pairwise potentials
			temp.setTo(1);
			for (auto &edgePotModel : getGraphDense().getEdgeModels()) {
				edgePotModel->apply(nodePotentials, tmp);					// tmp = f(pot_i)
				multiply(temp, tmp, temp);									// temp *= exp(tmp)
			}

			multiply(nodePotentials0, temp, nodePotentials);				// pot_(i+1) = pot_0 * next
		} // iter
	}
}
