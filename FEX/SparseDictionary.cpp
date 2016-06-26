#include "SparseDictionary.h"
#include "macroses.h"

namespace DirectGraphicalModels { namespace fex
{
	void CSparseDictionary::train(const Mat &X, int nWords, int batch, unsigned int nIt)
	{
		const int		nSamples	= X.cols;
		const int		nFeatures	= X.rows;

		const double	epsilon		= 1e-5;		// 1e-5;  // L1-regularisation epsilon |x| ~ sqrt(x^2 + epsilon)
		const double	lambdaH		= 5e-4;		// 5e-5;  // L1-regularisation parameter (on features)
		const double	lambdaDict	= 1e-2;		// 1e-2;  // L2-regularisation parameter (on basis)

		DGM_ASSERT_MSG(batch <= nSamples, "The batch number %d exceeds the length of the training data %d", batch, nSamples);

		RNG rng;
		if (!m_dict.empty()) m_dict.release();
		m_dict = Mat(nFeatures, nWords, CV_64FC1);
		rng.fill(m_dict, RNG::NORMAL, 0, 1);
		m_dict = m_dict * 0.12;

		Mat H;

		//Gradient Checking...
		//    mat randx = x.cols(5, 10);
		//    h = dict.t() * randx;
		//    for(int i = 0; i < h.n_rows; i++){
		//        h.row(i) = h.row(i) / norm(dict.col(i), 2);
		//    }
		for (unsigned int i = 0; i < nIt; i++) {
			printf("--- It: %d ---\n", i);

			int randomNum = ((long)rand() + (long)rand()) % (nSamples - batch);
			Mat randx = X(cvRect(randomNum, 0, batch, nFeatures));
			gemm(m_dict, randx, 1.0, Mat(), 0.0, H, GEMM_1_T);				// H = dict^T x randx;

			for (int j = 0; j < H.rows; j++) H.row(j) = H.row(j) / norm(m_dict.col(j), NORM_L2);

			calculateH(randx, m_dict, H, epsilon, lambdaH, 800);
			calculateDict(randx, m_dict, H, epsilon, lambdaDict, 800);

			//std::string str = "D:\\Dictionaries\\dict_";
			//str += std::to_string(i / 5);
			//str += ".txt";
			//if (i % 5 == 0) save(str);
		}
	}

	void CSparseDictionary::save(const std::string &fileName) const
	{
		FILE *pFile = fopen(fileName.c_str(), "wb");
		fwrite(&m_dict.rows, sizeof(int), 1, pFile);			// nFeatures
		fwrite(&m_dict.cols, sizeof(int), 1, pFile);			// nWords
		fwrite(m_dict.data, sizeof(double), m_dict.rows * m_dict.cols, pFile);
		fclose(pFile);
	}

	void CSparseDictionary::load(const std::string &fileName)
	{
		int nFeatures;
		int nWords;

		FILE *pFile = fopen(fileName.c_str(), "rb");
		fread(&nFeatures, sizeof(int), 1, pFile);
		fread(&nWords, sizeof(int), 1, pFile);

		if (!m_dict.empty()) m_dict.release();
		m_dict = Mat(nFeatures, nWords, CV_64FC1);

		fread(m_dict.data, sizeof(double), nFeatures * nWords, pFile);

		fclose(pFile);
	}

	Mat CSparseDictionary::decode(const Mat &X, CvSize imgSize) const
	{
		DGM_ASSERT_MSG(!m_dict.empty(), "The dictionary must me trained or loaded before using this function");

		const int		blockSize = static_cast<int>(sqrt(m_dict.rows));
		const double	epsilon = 1e-5;		// L1-regularisation epsilon |x| ~ sqrt(x^2 + epsilon)
		const double	lambdaH = 5e-5;		// regularisation parameter (on features)

		Mat res(imgSize, CV_64FC1, cvScalar(0));
		Mat cover(imgSize, CV_64FC1, cvScalar(0));

		// for (int s = 0; s < nSamples; s++) {
#ifdef USE_PPL
		concurrency::parallel_for(0, imgSize.height - blockSize + 1, blockSize, [&](int y) {
#else
		for (int y = 0; y < imgSize.height - blockSize + 1; y += blockSize) {
#endif
			for (int x = 0; x < imgSize.width - blockSize + 1; x += blockSize) {

				int s = y * (imgSize.width - blockSize + 1) + x;				// sample index

				Mat sample = X.col(s);											// sample
				Mat H;
				gemm(m_dict, sample, 1.0, Mat(), 0.0, H, GEMM_1_T);				// H = dict^T x sample

				for (int j = 0; j < H.rows; j++) H.row(j) = H.row(j) / norm(m_dict.col(j), NORM_L2);

				double cost = calculateH(sample, m_dict, H, epsilon, lambdaH, 800);
				//printf("Sample: %d, cost value = %f\n", s, cost);

				Mat tmp;
				gemm(m_dict, H, 1.0, Mat(), 0.0, tmp);							// tmp = dict x H
				tmp = tmp.reshape(0, blockSize);

				res(cvRect(x, y, blockSize, blockSize)) += tmp;
				cover(cvRect(x, y, blockSize, blockSize)) += 1.0;
			}
		}
#ifdef USE_PPL
		);
#endif
		res /= cover;
		return res;
		}

	// =================================================================================== static

	Mat CSparseDictionary::img2data(const Mat &img, int blockSize)
	{
		DGM_IF_WARNING(blockSize % 2 == 0, "The block size is even");

		// Converting to one channel image
		Mat I;
		if (img.channels() != 1) cvtColor(img, I, CV_RGB2GRAY);
		else img.copyTo(I);

		const int	dataWidth	= img.cols - blockSize + 1;
		const int	dataHeight	= img.rows - blockSize + 1;

		Mat res(blockSize * blockSize, dataWidth * dataHeight, CV_64FC1);

		for (register int y = 0; y < dataHeight; y++)
			for (register int x = 0; x < dataWidth; x++) {
				int s = y * dataWidth + x;															// sample index
				Mat sample = I(cvRect(x, y, blockSize, blockSize)).clone().reshape(0, 1).t();		// sample as a column-vector
				sample.convertTo(res.col(s), res.type(), 1.0 / 255);								// res.col(s) = sample
			} // x
		return res;
	}

	Mat CSparseDictionary::data2img(const Mat &X, CvSize imgSize)
	{
		Mat res(imgSize, CV_64FC1, cvScalar(0));
		Mat cover(imgSize, CV_64FC1, cvScalar(0));

		const int	blockSize	= static_cast<int>(sqrt(X.rows));
		const int	dataWidth	= res.cols - blockSize + 1;
		const int	dataHeight	= res.rows - blockSize + 1;

		for (int y = 0; y < dataHeight; y++)
			for (int x = 0; x < dataWidth; x++) {
				int s = y * dataWidth + x;															// sample index
					
				Mat sample = X.col(s).t();															// smple as a row-vector
				sample = sample.reshape(0, blockSize);												// square sample - data patch

				res(cvRect(x, y, blockSize, blockSize)) += sample;
				cover(cvRect(x, y, blockSize, blockSize)) += 1.0;
			} // x
		res /= cover;
		return res;
	}

	Mat CSparseDictionary::shuffleCols(const Mat &X)
	{
		std::vector<int> seeds;
		for (int x = 0; x < X.cols; x++) 
			seeds.push_back(x);

		randShuffle(seeds);

		Mat res(X.size(), X.type());
		for (int x = 0; x < X.cols; x++)
			X.col(seeds[x]).copyTo(res.col(x));

		return res;
	}
	
	// =================================================================================== protected

	double CSparseDictionary::calculateCost(cost_type cType, const Mat &X, const Mat &dict, const Mat &H, Mat &grad, double epsilon, double lambda)
	{
		const int nSamples = X.cols;

		Mat delta;
		gemm(dict, H, 1.0, X, -1.0, delta);					// delta = dict x H - X	
		reduce(delta, delta, 1, CV_REDUCE_AVG);
		pow(delta, 2, delta);
		double cost = sum(delta)[0];

		Mat sparsityMatrix;
		pow(H, 2, sparsityMatrix);
		sparsityMatrix += epsilon;
		sqrt(sparsityMatrix, sparsityMatrix);				// sparsityMatrix = sqrt(H^2 + epsilon)


		if (cType == COST_H) {
			// ----- Grad -----
			Mat p1;
			gemm(dict, H, 1.0, Mat(), 0.0, p1);				// p1 = dict x H

			Mat p2;
			gemm(dict, X, 1.0, Mat(), 0.0, p2, GEMM_1_T);	// p2 = dict^T x X

			Mat p3;
			gemm(dict, p1, 2.0, p2, -2.0, p3, GEMM_1_T);	// p3 = 2 * (dict^T x p1) - 2 * p2 

			grad = p3 / nSamples;
			grad += lambda * (H / sparsityMatrix);


			// ----- Cost -----
			Mat sparsityVector;
			reduce(sparsityMatrix, sparsityVector, 1, CV_REDUCE_AVG);
			cost += lambda * sum(sparsityVector)[0];
		} 
		else { // DICT_COST
			// ----- Grad -----
			Mat p1;
			gemm(H, H, 1.0, Mat(), 0.0, p1, GEMM_2_T);		// p1 = H x H^T

			Mat p2;
			gemm(X, H, 1.0, Mat(), 0.0, p2, GEMM_2_T);		// p2 = X x H^T

			Mat p3;
			gemm(dict, p1, 2.0, p2, -2.0, p3);				// p3 = 2 * (dict x p1) - 2 * p2

			grad = p3 / nSamples;
			grad += 2 * lambda * dict;

			// ----- Cost -----
			Mat dict2;
			pow(dict, 2, dict2);
			cost += lambda * sum(dict2)[0];
		}

		return cost;
	}

	double CSparseDictionary::calculateDict(const Mat &X, Mat &dict, const Mat &H, double epsilon, double lambda, unsigned int nIt)
	{
		// define the velocity vectors.
		Mat dictGrad(dict.size(), CV_64FC1, cvScalar(0));
		Mat inc_dict(dict.size(), CV_64FC1, cvScalar(0));

		const double	lrate = 0.05;				//Learning rate for weights 
		const double	weightcost = 0.0002;
		const double	initialmomentum = 0.5;
		const double	finalmomentum = 0.9;
		double			momentum;
		double			cost;

		for (unsigned int i = 0; i < nIt; i++) {
			momentum = (i > 10) ? finalmomentum : initialmomentum;
			cost = calculateCost(COST_DICT, X, dict, H, dictGrad, epsilon, lambda);
			// update weights 
			inc_dict = momentum * inc_dict + lrate * (dictGrad - weightcost * dict);
			dict -= inc_dict;
		}
		printf("training dict, Cost function value = %f\n", cost);
		return cost;
	}

	double CSparseDictionary::calculateH(const Mat &X, const Mat& dict, Mat &H, double epsilon, double lambda,  unsigned int nIt)
	{
		// define the velocity vectors.
		Mat hGrad(H.size(), CV_64FC1, cvScalar(0));
		Mat inc_h(H.size(), CV_64FC1, cvScalar(0));

		const double lrate = 0.05;						//Learning rate for weights 
		const double weightcost = 0.0002;
		const double initialmomentum = 0.5;
		const double finalmomentum = 0.9;
		double		 momentum;
		double		 cost;

		for (unsigned int i = 0; i < nIt; i++) {
			momentum = (i > 10) ? finalmomentum : initialmomentum;
			cost = calculateCost(COST_H, X, dict, H, hGrad, epsilon, lambda);
			// update weights 
			inc_h = momentum * inc_h + lrate * (hGrad - weightcost * H);
			H -= inc_h;
		} // i
		//printf("training H, Cost function value = %f\n", cost);
		return cost;
	}

} }
