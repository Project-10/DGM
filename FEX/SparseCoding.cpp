#include "SparseCoding.h"
#include "SparseDictionary.h"
#include "macroses.h"

namespace DirectGraphicalModels { namespace fex
{

Mat CSparseCoding::get(const Mat &img, const Mat &D, SqNeighbourhood nbhd)
{
	DGM_ASSERT_MSG(!D.empty(), "The dictionary must me trained or loaded before using this function");

	const word		nWords		= D.rows;
	const int		blockSize	= static_cast<int>(sqrt(D.cols));
	const int		dataWidth	= img.cols - blockSize + 1;
	const int		dataHeight	= img.rows - blockSize + 1;
	const float		lambda		= 5e-5f;							// regularisation parameter (on features)
	const float		epsilon		= 1e-5f;							// L1-regularisation epsilon |x| ~ sqrt(x^2 + epsilon)
			
	DGM_ASSERT_MSG(nbhd.leftGap + nbhd.rightGap == nbhd.upperGap + nbhd.lowerGap, "The Neighbourhood must be a square for this method");
	DGM_ASSERT(blockSize == nbhd.leftGap + nbhd.rightGap + 1);
	DGM_ASSERT_MSG(nWords <= CV_CN_MAX, "The number of words %d exceeds the maximal allowed number of channels %d", nWords, CV_CN_MAX);

	Mat X = img2data(img, blockSize);

	Mat *pTemp = new Mat[nWords];
	for (word w = 0; w < nWords; w++)
		pTemp[w] = Mat(img.size(), CV_8UC1, cvScalar(0));

#ifdef USE_PPL
	concurrency::parallel_for(0, dataHeight, 1, [&] (int y) {
#else
	for (int y = 0; y < dataHeight; y++) {
#endif
		Mat _W, W;
		for (int x = 0; x < dataWidth; x += 1) {
			int s = y * dataWidth + x;										// sample index
			Mat sample = X.row(s);											// sample as a row-vector

			gemm(D, sample.t(), 1.0, Mat(), 0.0, _W);						// W = D x sample^T
			W = _W.t();
			for (int w = 0; w < W.cols; w++)
				W.col(w) /= norm(D.row(w), NORM_L2);

			calculate_W(sample, D, W, lambda, epsilon, 200);

			// double minVal, maxVal;
			// minMaxLoc(W, &minVal, &maxVal);
			// printf("[%f; %f]\n", minVal, maxVal);
			
			for (word w = 0; w < nWords; w++) {
				pTemp[w].at<byte>(y + nbhd.upperGap, x + nbhd.leftGap) = linear_mapper(W.at<float>(0, w), -1.0f, 1.0f);
				//pTemp[w].at<float>(y + nbhd.upperGap, x + nbhd.leftGap) = W.at<float>(0, w);
			}
		}
	}
#ifdef USE_PPL
	);
#endif

	Mat res;
	merge(pTemp, nWords, res);

	delete[] pTemp;

	return res;
}
} }