#include "SparseCoding.h"
#include "SparseDictionary.h"
#include "macroses.h"

namespace DirectGraphicalModels { namespace fex 
{

	Mat CSparseCoding::get(const Mat &img, const Mat &dict, SqNeighbourhood nbhd)
	{
		DGM_ASSERT_MSG(!dict.empty(), "The dictionary must me trained or loaded before using this function");
		
		const int		nWords		= dict.cols;
		const int		blockSize	= static_cast<int>(sqrt(dict.rows));
		const int		dataWidth	= img.cols - blockSize + 1;
		const int		dataHeight	= img.rows - blockSize + 1;
		const double	epsilon		= 1e-5;							// L1-regularisation epsilon |x| ~ sqrt(x^2 + epsilon)
		const double	lambdaH		= 5e-5;							// regularisation parameter (on features)

		DGM_ASSERT_MSG(nbhd.leftGap + nbhd.rightGap == nbhd.upperGap + nbhd.lowerGap, "The Neighbourhood must be a square for this method");	
		DGM_ASSERT(blockSize == nbhd.leftGap + nbhd.rightGap + 1);

		// Converting to one channel image
		Mat I;
		if (img.channels() != 1) cvtColor(img, I, CV_RGB2GRAY);
		else img.copyTo(I);

		// Prepare data
		Mat X(blockSize * blockSize, dataWidth * dataHeight, CV_64FC1);
		for (int y = 0; y < dataHeight; y++)
			for (register int x = 0; x < dataWidth; x++) {
				int s = y * dataWidth + x;										// sample index
				Mat sample = I(cvRect(x, y, blockSize, blockSize)).clone().reshape(0, 1).t();
				sample.convertTo(X.col(s), X.type(), 1.0 / 255);
				s++;
			}

		Mat *pTemp = new Mat[nWords];
		for (int w = 0; w < nWords; w++) 
			pTemp[w] = Mat(I.size(), CV_8UC1, cvScalar(0));

		double min = 0;
		double max = 0;

#ifdef USE_PPL
		concurrency::parallel_for(0, dataHeight, [&] (int y) {
#else
		for (int y = 0; y < dataHeight; y++) {
#endif
			for (int x = 0; x < dataWidth; x++) {
				int s = y * dataWidth + x;										// sample index
				Mat sample = X.col(s);											// sample as a column-vector
				Mat H;
				gemm(dict, sample, 1.0, Mat(), 0.0, H, GEMM_1_T);				// H = dict^T x sample

				for (int j = 0; j < H.rows; j++)
					H.row(j) = H.row(j) / norm(dict.col(j), NORM_L2);

				double cost = calculateH(sample, dict, H, epsilon, lambdaH, 200);
				//printf("Sample: %d, cost value = %f\n", s, cost);

				for (int w = 0; w < nWords; w++) {
					//if (min > H.at<double>(w, 0)) min = H.at<double>(w, 0);
					//if (max < H.at<double>(w, 0)) max = H.at<double>(w, 0);

					pTemp[w].at<byte>(y + nbhd.upperGap, x + nbhd.leftGap) = linear_mapper(static_cast<float>(H.at<double>(w, 0)), -2.5f, 2.5f);
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