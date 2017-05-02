#include "SparseCoding.h"
#include "SparseDictionary.h"
#include "LinearMapper.h"
#include "macroses.h"

namespace DirectGraphicalModels { namespace fex
{
Mat CSparseCoding::get(const Mat &img, const Mat &D, SqNeighbourhood nbhd)
{
	const word	  nWords	= D.rows;
	DGM_ASSERT_MSG(nWords <= CV_CN_MAX, "The number of words %d exceeds the maximal allowed number of channels %d. Use get_v() function instead.", nWords, CV_CN_MAX);

	Mat			res;
	vec_mat_t	vFeatures	= get_v(img, D, nbhd);
	merge(vFeatures, res);
	return res;
}

vec_mat_t CSparseCoding::get_v(const Mat &img, const Mat &D, SqNeighbourhood nbhd)
{
	DGM_ASSERT_MSG(!D.empty(), "The dictionary must me trained or loaded before using this function");

	const word		nWords = D.rows;
	const int		blockSize = static_cast<int>(sqrt(D.cols));
	const int		dataWidth = img.cols - blockSize + 1;
	const int		dataHeight = img.rows - blockSize + 1;

	DGM_ASSERT_MSG(nbhd.leftGap + nbhd.rightGap == nbhd.upperGap + nbhd.lowerGap, "The Neighbourhood must be a square for this method");
	DGM_ASSERT(blockSize == nbhd.leftGap + nbhd.rightGap + 1);

	Mat X = img2data(img, blockSize);
	int normalizer = (X.depth() == CV_8U) ? 255 : 65535;

	vec_mat_t res(nWords);
	for (word w = 0; w < nWords; w++)
		res[w] = Mat(img.size(), CV_8UC1, cvScalar(0));

#ifdef ENABLE_PPL
	concurrency::parallel_for(0, dataHeight, 1, [&](int y) {
#else
	for (int y = 0; y < dataHeight; y++) {
#endif
		Mat _W, W;
		for (int x = 0; x < dataWidth; x += 1) {
			int s = y * dataWidth + x;										// sample index
			Mat sample = X.row(s);											// sample as a row-vector
			sample.convertTo(sample, CV_32FC1, 1.0 / normalizer);

			gemm(D, sample.t(), 1.0, Mat(), 0.0, _W);						// W = D x sample^T
			W = _W.t();
			for (int w = 0; w < W.cols; w++)
				W.col(w) /= norm(D.row(w), NORM_L2);

			calculate_W(sample, D, W, SC_LAMBDA, SC_EPSILON, 200, SC_LRATE_W);

			for (word w = 0; w < nWords; w++) 
				res[w].at<byte>(y + nbhd.upperGap, x + nbhd.leftGap) = linear_mapper<byte>(W.at<float>(0, w), -1.0f, 1.0f);
		}
	}
#ifdef ENABLE_PPL
	);
#endif
	return res;
}
} }