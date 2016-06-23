#include "SC.h"

namespace DirectGraphicalModels { namespace fex 
{
	const int DICT_SIZE = 49;
	const int block_size = 8;

	void CSC::trainDictionary(Mat &X, int dictsize, int batch, unsigned int nIt)
	{
		const int		nSamples = X.cols;
		const int		nFeatures = X.rows;

		const double	lambda = 5e-4;		// 5e-5;  // L1-regularisation parameter (on features)
		const double	epsilon = 1e-5;		// 1e-5;  // L1-regularisation epsilon |x| ~ sqrt(x^2 + epsilon)
		const double	gamma = 1e-2;		// 1e-2;  // L2-regularisation parameter (on basis)

		RNG rng;
		Mat H;
		if (!m_dict.empty()) m_dict.release();
		m_dict = Mat(nFeatures, dictsize, CV_64FC1);
		rng.fill(m_dict, RNG::NORMAL, 0, 1);
		m_dict = m_dict * 0.12;

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

			trainingH(randx, H, lambda, epsilon, gamma, 800);
			trainingDict(randx, m_dict, H, lambda, epsilon, gamma, 800);

			std::string str = "dict_";
			str += std::to_string(i / 5);
			str += ".txt";
			if (i % 5 == 0) saveDictionary(str);
		}
	}

	Mat CSC::decoder(const Mat &X, CvSize imgSize) const 
	{
		const double	lambda = 5e-5;		// L1-regularisation parameter (on features)
		const double	epsilon = 1e-5;		// L1-regularisation epsilon |x| ~ sqrt(x^2 + epsilon)
		const double	gamma = 1e-2;		// L2-regularisation parameter (on basis)

		Mat res(imgSize, CV_64FC1, cvScalar(0));
		Mat cover(imgSize, CV_64FC1, cvScalar(0));

		// for (int s = 0; s < nSamples; s++) {
#ifdef USE_PPL
		concurrency::parallel_for(0, imgSize.height - block_size + 1, block_size, [&](int y) {
#else
		for (int y = 0; y < imgSize.height - block_size + 1; y += block_size) {
#endif
			for (int x = 0; x < imgSize.width - block_size + 1; x += block_size) {

				int s = y * (imgSize.width - block_size + 1) + x;				// sample index

				Mat sample = X.col(s);											// sample
				Mat H;
				gemm(m_dict, sample, 1.0, Mat(), 0.0, H, GEMM_1_T);				// H = dict^T x sample

				for (int j = 0; j < H.rows; j++) H.row(j) = H.row(j) / norm(m_dict.col(j), NORM_L2);

				double cost = trainingH(sample, H, lambda, epsilon, gamma, 800);
				//printf("Sample: %d, cost value = %f\n", s, cost);

				Mat tmp;
				gemm(m_dict, H, 1.0, Mat(), 0.0, tmp);							// tmp = dict x H
				tmp = tmp.reshape(0, block_size);

				res(cvRect(x, y, block_size, block_size)) += tmp;
				cover(cvRect(x, y, block_size, block_size)) += 1.0;
			}
		}
#ifdef USE_PPL
		);
#endif
		res /= cover;
		return res;
	}

	void CSC::saveDictionary(const std::string &fileName) const
	{
		FILE *pOut = fopen(fileName.c_str(), "w");

		for (int y = 0; y < m_dict.rows; y++) {
			for (int x = 0; x < m_dict.cols; x++) {
				fprintf(pOut, "%lf", m_dict.at<double>(y, x));
				if (x == m_dict.cols - 1) fprintf(pOut, "\n");
				else fprintf(pOut, " ");
			}
		}
		fclose(pOut);
	}

	void CSC::loadDictionary(const std::string &fileName)
	{
		if (!m_dict.empty()) m_dict.release();
		m_dict = Mat(block_size * block_size, DICT_SIZE, CV_64FC1);

		FILE *pFile = fopen(fileName.c_str(), "r");
		double val;
		for (int counter = 0; ; counter++) {
			if (fscanf(pFile, "%lf", &val) == EOF) break;
			m_dict.at<double>(counter / DICT_SIZE, counter % DICT_SIZE) = val;
		}
		fclose(pFile);
	}

	// =================================================================================== static
	
	Mat CSC::renderDict(Mat &dict)
	{
		const int margin = 2;
		const int dictLen = dict.cols;
		int width = static_cast<int>(sqrt(dictLen));	
		int height = dictLen / width;
		CvSize imgSize = cvSize(width * block_size + (width + 1) * margin, height * block_size + (height + 1) * margin);

		Mat res(imgSize, CV_8UC1, cvScalar(0));

		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++) {
				int y0 = margin + y * (block_size + margin);
				int x0 = margin + x * (block_size + margin);

				for (int j = 0; j < block_size; j++)
					for (int i = 0; i < block_size; i++)
						res.at<byte>(y0 + j, x0 + i) = static_cast<byte>(127 + 500 * dict.at<double>(j * block_size + i, y * width + x));

			}

		return res;
	}

	Mat CSC::img2data(const Mat &img)
	{
		const int	dataWidth = img.cols - block_size + 1;
		const int	dataHeight = img.rows - block_size + 1;

		Mat res(block_size * block_size, dataWidth * dataHeight, CV_64FC1);

		int sample = 0;
		for (register int y = 0; y < dataHeight; y++)
			for (register int x = 0; x < dataWidth; x++) {
				for (int j = 0; j < block_size; j++)
					for (int i = 0; i < block_size; i++)
						res.at<double>(j * block_size + i, sample) = static_cast<double>(img.at<byte>(y + j, x + i)) / 255.0;
				sample++;
			}
		return res;
	}

	Mat CSC::data2img(Mat &X, CvSize imgSize)
	{
		Mat res(imgSize, CV_64FC1, cvScalar(0));
		Mat cover(imgSize, CV_64FC1, cvScalar(0));

		for (int y = 0; y < imgSize.height - block_size + 1; y += block_size)
			for (int x = 0; x < imgSize.width - block_size + 1; x += block_size) {

				int s = y * (imgSize.width - block_size + 1) + x;				// sample index

				Mat tmp = X.col(s).t();
				tmp = tmp.reshape(0, block_size);

				res(cvRect(x, y, block_size, block_size)) += tmp;
				cover(cvRect(x, y, block_size, block_size)) += 1.0;
			}
		res /= cover;
		return res;
	}

	Mat CSC::shuffleCols(const Mat &matrix)
	{
		std::vector<int> seeds;
		for (int x = 0; x < matrix.cols; x++) seeds.push_back(x);

		randShuffle(seeds);

		Mat res(matrix.size(), matrix.type());
		for (int x = 0; x < matrix.cols; x++)
			matrix.col(seeds[x]).copyTo(res.col(x));

		return res;
	}

	// =================================================================================== protected

	double CSC::getSparseCodingCost(const Mat &X, const Mat &H, Mat &dictGrad, Mat &hGrad, double lambda, double epsilon, double gamma, sc_cost cond) const
	{
		const int nSamples = X.cols;

		Mat delta;
		gemm(m_dict, H, 1.0, X, -1.0, delta);				// delta = dict x H - X	
		reduce(delta, delta, 1, CV_REDUCE_AVG);
		pow(delta, 2, delta);
		double cost = sum(delta)[0];

		Mat sparsityMatrix;
		pow(H, 2, sparsityMatrix);
		sparsityMatrix += epsilon;
		sqrt(sparsityMatrix, sparsityMatrix);			// sparsityMatrix = sqrt(H^2 + epsilon)


		if (cond == H_COST) {
			Mat sparsityVector;
			reduce(sparsityMatrix, sparsityVector, 1, CV_REDUCE_AVG);
			cost += lambda * sum(sparsityVector)[0];
		}
		else {
			Mat dict2;
			pow(m_dict, 2, dict2);
			cost += gamma * sum(dict2)[0];
		}

		if (!hGrad.empty()) {
			Mat p1;
			gemm(m_dict, H, 1.0, Mat(), 0.0, p1);			// p1 = dict x H

			Mat p2;
			gemm(m_dict, X, 1.0, Mat(), 0.0, p2, GEMM_1_T);	// p2 = dict^T x X

			Mat p3;
			gemm(m_dict, p1, 2.0, p2, -2.0, p3, GEMM_1_T);	// p3 = 2 * (dict^T x p1) - 2 * p2 

			hGrad = p3 / nSamples;
			hGrad += lambda * (H / sparsityMatrix);
		}

		if (!dictGrad.empty()) {
			Mat p1;
			gemm(H, H, 1.0, Mat(), 0.0, p1, GEMM_2_T);		// p1 = H x H^T

			Mat p2;
			gemm(X, H, 1.0, Mat(), 0.0, p2, GEMM_2_T);		// p2 = X x H^T

			Mat p3;
			gemm(m_dict, p1, 2.0, p2, -2.0, p3);			// p3 = 2 * (dict x p1) - 2 * p2

			dictGrad = p3 / nSamples;
			dictGrad += 2 * gamma * m_dict;
		}


		return cost;
	}

	double CSC::trainingDict(const Mat &X, Mat &dict, const Mat &H, double lambda, double epsilon, double gamma, unsigned int nIt)
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
			cost = getSparseCodingCost(X, H, dictGrad, Mat(), lambda, epsilon, gamma, DICT_COST);
			// update weights 
			inc_dict = momentum * inc_dict + lrate * (dictGrad - weightcost * dict);
			dict -= inc_dict;
		}
		printf("training dict, Cost function value = %f\n", cost);
		return cost;
	}

	double CSC::trainingH(const Mat &X, Mat &H, double lambda, double epsilon, double gamma, unsigned int nIt) const 
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
			cost = getSparseCodingCost(X, H, Mat(), hGrad, lambda, epsilon, gamma, H_COST);
			// update weights 
			inc_h = momentum * inc_h + lrate * (hGrad - weightcost * H);
			H -= inc_h;
		} // i
		printf("training H, Cost function value = %f\n", cost);
		return cost;
	}

} }