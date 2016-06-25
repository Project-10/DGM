#include "SparseCode.h"
#include "SparseCodeDictionary.h"
#include "macroses.h"

namespace DirectGraphicalModels { namespace fex 
{

	Mat CSparseCode::get(const Mat &img, CSparseCodeDictionary *pDict, SqNeighbourhood nbhd)
	{
		const Mat		dict		= pDict->get();
		const int		nWords		= pDict->getNumWords();
		const int		blockSize	= pDict->getBlockSize();
		const int		dataWidth	= img.cols - blockSize + 1;
		const int		dataHeight	= img.rows - blockSize + 1;
		const double	lambda		= 5e-5;									// L1-regularisation parameter (on features)
		const double	epsilon		= 1e-5;									// L1-regularisation epsilon |x| ~ sqrt(x^2 + epsilon)
		const double	gamma		= 1e-2;									// L2-regularisation parameter (on basis)

		DGM_ASSERT_MSG(nbhd.leftGap + nbhd.rightGap == nbhd.upperGap + nbhd.lowerGap, "The Neighbourhood must be a square for this method");	
		DGM_ASSERT_MSG(pDict->isTrained(), "The dictionary must me trained or loaded before using this function");
		DGM_ASSERT(blockSize == nbhd.leftGap + nbhd.rightGap + 1);

		const Mat X = img2data(img, blockSize);		// TODO: rewrite this
		
		Mat *pTemp = new Mat[nWords];
		for (int w = 0; w < nWords; w++) 
			pTemp[w] = Mat(img.size(), CV_8UC1, cvScalar(0));


		double min = 0;
		double max = 0;

		// for (int s = 0; s < nSamples; s++) {
#ifdef USE_PPL
		concurrency::parallel_for(0, dataHeight, 1, [&] (int y) {
#else
		for (int y = 0; y < dataHeight; y++ ) {
#endif
			for (int x = 0; x < dataWidth; x++) {
				int s = y * dataWidth + x;										// sample index

				Mat sample = X.col(s);											// sample
				Mat H;
				gemm(pDict->get(), sample, 1.0, Mat(), 0.0, H, GEMM_1_T);		// H = dict^T x sample

				for (int j = 0; j < H.rows; j++) 
					H.row(j) = H.row(j) / norm(dict.col(j), NORM_L2);

				double cost = trainH(sample, dict, H, lambda, epsilon, gamma, 200);
				//printf("Sample: %d, cost value = %f\n", s, cost);

				for (int w = 0; w < nWords; w++) {
					//if (min > H.at<double>(w, 0)) min = H.at<double>(w, 0);
					//if (max < H.at<double>(w, 0)) max = H.at<double>(w, 0);
					
					
					pTemp[w].at<byte>(y, x) = static_cast<byte>(127.0 + 25 * H.at<double>(w, 0));
					//printf("%d ", pTemp[w].at<byte>(y, x));
				}
				//printf("\n");
				//printf("[%f; %f]\n", min, max);
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