#include "SparseDictionary.h"
#include "macroses.h"
#include "DGM\parallel.h"
#include "DGM\random.h"

namespace DirectGraphicalModels { namespace fex
{
// =================================================================================== public
void CSparseDictionary::train(const Mat &X, word nWords, dword batch, unsigned int nIt, float lRate, const std::string &fileName)
{
	const dword		nSamples  = X.rows;
	const int		sampleLen = X.cols;

	// Assertions
	DGM_ASSERT_MSG((X.depth() == CV_8U) || (X.depth() == CV_16U), "The depth of argument X is not supported");
	if (batch > nSamples) {
		DGM_WARNING("The batch number %d exceeds the length of the training data %d", batch, nSamples);
		batch = nSamples;
	}

	// 1. Initialize dictionary D randomly
	if (!m_D.empty()) m_D.release();
	m_D = random::N(cvSize(sampleLen, nWords), CV_32FC1, 0.0f, 0.3f);  

	Mat		_W, W;					// Weights matrix (Size: nStamples x nWords)
	float	cost;

	// 2. Repeat until convergence
	for (unsigned int i = 0; i < nIt; i++) {								// iterations
#ifdef DEBUG_PRINT_INFO
		if (i == 0) printf("\n");
		printf("--- It: %d ---\n", i);
#endif
		// 2.1 Select a random mini-batch of 2000 patches
		dword rndRow = random::u<dword>(0, MAX(1, nSamples - batch) - 1);
		int normalizer = (X.depth() == CV_8U) ? 255 : 65535;
		Mat _X = X(cvRect(0, rndRow, sampleLen, batch));
		_X.convertTo(_X, CV_32FC1, 1.0 / normalizer);
		
		// 2.2 Initialize W
		parallel::gemm(m_D, _X.t(), 1.0, Mat(), 0.0, _W);					// _W = (D x _X^T);
		W = _W.t();															// _W = (D x _X^T)^T;
		for (word w = 0; w < W.cols; w++)
			W.col(w) /= norm(m_D.row(w), NORM_L2);					

#ifdef DEBUG_PRINT_INFO
		printf("Cost: ");
		cost = calculateCost(_X, m_D, W, SC_LAMBDA, SC_EPSILON, SC_GAMMA);
		printf("%f -> ", cost);
#endif
		
		// 2.3. Find the W, that minimizes J(D, W) for the D found in the previos step
		// argmin J(W) = ||W x D - X||^{2}_{2} + \lambda||W||_1
		calculate_W(_X, m_D, W, SC_LAMBDA, SC_EPSILON, 800, SC_LRATE_W);
#ifdef DEBUG_PRINT_INFO		
		cost = calculateCost(_X, m_D, W, SC_LAMBDA, SC_EPSILON, SC_GAMMA);
		printf("%f -> ", cost);
#endif

		// 2.4 Solve for the D that minimizes J(D, W) for the W found in the previous step
		// argmin J(D) = ||W x D - X||^{2}_{2} + \gamma||D||^{2}_{2}
		calculate_D(_X, m_D, W, SC_GAMMA, 800, lRate);
		cost = calculateCost(_X, m_D, W, SC_LAMBDA, SC_EPSILON, SC_GAMMA);
#ifdef DEBUG_PRINT_INFO	
		printf("%f\n", cost);
#endif
		DGM_ASSERT_MSG(!isnan(cost), "Training is unstable. Try reducing the learning rate for dictionary.");

		// 2.5 Saving intermediate dictionary
		if (!fileName.empty()) {
			std::string str = fileName + std::to_string(i / 5);
			str += ".dic";
			if (i % 5 == 0) save(str);
		}
	} // i
}

void CSparseDictionary::save(const std::string &fileName) const
{
	FILE *pFile = fopen(fileName.c_str(), "wb");
	fwrite(&m_D.rows, sizeof(int), 1, pFile);			// nWords
	fwrite(&m_D.cols, sizeof(int), 1, pFile);			// sampleLen
	fwrite(m_D.data, sizeof(float), m_D.rows * m_D.cols, pFile);
	fclose(pFile);
}

void CSparseDictionary::load(const std::string &fileName)
{
	int sampleLen;
	int nWords;

	FILE *pFile = fopen(fileName.c_str(), "rb");	
	DGM_ASSERT_MSG(pFile, "Can't load data from %s", fileName.c_str());
	
	fread(&nWords, sizeof(int), 1, pFile);
	fread(&sampleLen, sizeof(int), 1, pFile);

	if (!m_D.empty()) m_D.release();
	m_D = Mat(nWords, sampleLen, CV_32FC1);

	fread(m_D.data, sizeof(float), nWords * sampleLen, pFile);

	fclose(pFile);
}

#ifdef DEBUG_MODE	// --- Debugging ---
Mat CSparseDictionary::TEST_decode(const Mat &X, CvSize imgSize) const
{
	DGM_ASSERT_MSG(!m_D.empty(), "The dictionary must me trained or loaded before using this function");

	const int	blockSize	= getBlockSize();
	const int	dataWidth	= imgSize.width  - blockSize + 1;
	const int	dataHeight	= imgSize.height - blockSize + 1;
	const int	sampleType	= X.type();
	const int	normalizer	= (X.depth() == CV_8U) ? 255 : 65535;

	const float	lambda		= 5e-5f;		// L1-regularisation parameter (on features)
	const float	epsilon		= 1e-5f;		// L1-regularisation epsilon |x| ~ sqrt(x^2 + epsilon)

	Mat res(imgSize, CV_32FC1, cvScalar(0));
	Mat cover(imgSize, CV_32FC1, cvScalar(0));

#ifdef ENABLE_PPL
	concurrency::parallel_for(0, dataHeight, blockSize, [&](int y) {
#else
	for (int y = 0; y < dataHeight; y += blockSize) {
#endif
		Mat _W, W;
		Mat tmp;
		for (int x = 0; x < dataWidth; x += blockSize) {

			int s = y * dataWidth + x;										// sample index
			Mat sample = X.row(s);											// sample
			sample.convertTo(sample, CV_32FC1, 1.0 / normalizer);

			gemm(m_D, sample.t(), 1.0, Mat(), 0.0, _W);						// _W = (D x sample^T)
			W = _W.t();														// W = (D x sample^T)^T
			for (int w = 0; w < W.cols; w++)
				W.col(w) /= norm(m_D.row(w), NORM_L2);

			// argmin J(W) = ||W x D - X||^{2}_{2} + \lambda||W||_1
			calculate_W(sample, m_D, W, lambda, epsilon, 800);

			gemm(W, m_D, 1.0, Mat(), 0.0, tmp);								// tmp = W x D
			tmp = tmp.reshape(0, blockSize);

			res(cvRect(x, y, blockSize, blockSize))   += tmp;
			cover(cvRect(x, y, blockSize, blockSize)) += 1.0;
		}
	}
#ifdef ENABLE_PPL
	);
#endif
	res /= cover;
	res.convertTo(res, sampleType, normalizer);
	return res;
}
#endif				// --- --------- ---

// =================================================================================== static

Mat CSparseDictionary::img2data(const Mat &img, int blockSize, float varianceThreshold)
{
	DGM_IF_WARNING(blockSize % 2 == 0, "The block size is even");

	varianceThreshold = sqrtf(varianceThreshold);
	// Converting to one channel image
	Mat I;
	if (img.channels() != 1) cvtColor(img, I, CV_RGB2GRAY);
	else img.copyTo(I);

	const int	dataHeight = img.rows - blockSize + 1;
	const int	dataWidth = img.cols - blockSize + 1;

	Mat res;
	Mat sample;

	for (int y = 0; y < dataHeight; y++)
		for (int x = 0; x < dataWidth; x++) {
			sample = I(cvRect(x, y, blockSize, blockSize)).clone().reshape(0, 1);			// sample as a row-vector

			Scalar stddev;
			meanStdDev(sample, Mat(), stddev);
			float variance = (float) stddev[0];
			//printf("variance = %f\n", variance);

			if (variance >= varianceThreshold)
				res.push_back(sample);
		} // x
	return res;
}

Mat CSparseDictionary::data2img(const Mat &X, CvSize imgSize)
{
	Mat res(imgSize, CV_32FC1, cvScalar(0));
	Mat cover(imgSize, CV_32FC1, cvScalar(0));

	const int	blockSize = static_cast<int>(sqrt(X.cols));
	const int	dataWidth = res.cols - blockSize + 1;
	const int	dataHeight = res.rows - blockSize + 1;

	for (int s = 0; s < X.rows; s++) {
		Mat sample = X.row(s);															// smple as a row-vector
		sample = sample.reshape(0, blockSize);											// square sample - data patch

		int y = s / dataWidth;
		int x = s % dataWidth;

		res(cvRect(x, y, blockSize, blockSize)) += sample;
		cover(cvRect(x, y, blockSize, blockSize)) += 1.0;
	}
	res /= cover;

	res.convertTo(res, X.type(), 1);
	return res;
}

// =================================================================================== protected

// J(W) = ||W x D - X||^{2}_{2} + \lambda||W||_1
void CSparseDictionary::calculate_W(const Mat &X, const Mat &D, Mat &W, float lambda, float epsilon, unsigned int nIt, float lRate)
{
	// Define the velocity vectors
	Mat gradient;
	Mat incriment(W.size(), W.type(), cvScalar(0));

	for (unsigned int i = 0; i < nIt; i++) {
		float momentum = (i <= 10) ? 0.5f : 0.9f;
		gradient = calculateGradient(GRAD_W, X, D, W, lambda, epsilon, 0);
		incriment = momentum * incriment + lRate * (gradient - 2e-4f * W);
		W -= incriment;
	} // i
}

// J(D) = ||W x D - X||^{2}_{2} + \gamma||D||^{2}_{2}
void CSparseDictionary::calculate_D(const Mat &X, Mat &D, const Mat &W, float gamma, unsigned int nIt, float lRate)
{
	// define the velocity vectors
	Mat gradient;
	Mat incriment(D.size(), D.type(), cvScalar(0));

	for (unsigned int i = 0; i < nIt; i++) {
		float momentum = (i <= 10) ? 0.5f : 0.9f;
		gradient = calculateGradient(GRAD_D, X, D, W, 0, 0, gamma);
		incriment = momentum * incriment + lRate * (gradient - 2e-4f * D);
		D -= incriment;
	} // i
}

Mat CSparseDictionary::calculateGradient(grad_type gType, const Mat &X, const Mat &D, const Mat &W, float lambda, float epsilon, float gamma)
{
	const int	nSamples = X.rows;

	Mat temp;
	parallel::gemm(W, D, 2.0f / nSamples, X, -2.0f / nSamples, temp);				// temp = (2.0 / nSamples) * (W x D - X)
	Mat gradient;
	Mat sparsityMatrix;

	switch (gType) {
	case GRAD_W:	// 2 * (W x D - X) x D^T / nSamples + lambda * W / sqrt(W^2 + epsilon)
		multiply(W, W, sparsityMatrix);
		sparsityMatrix += epsilon;
		sqrt(sparsityMatrix, sparsityMatrix);										// sparsityMatrix = sqrt(W^2 + epsilon)
		parallel::gemm(temp, D.t(), 1.0, W / sparsityMatrix, lambda, gradient);
		break;
	case GRAD_D:	// 2 * W^T x (W x D - X) / nSamples + 2 * gamma * D
		parallel::gemm(W.t(), temp, 1.0, D, 2 * gamma, gradient);
		break;
	}

	return gradient;
}

float CSparseDictionary::calculateCost(const Mat &X, const Mat &D, const Mat &W, float lambda, float epsilon, float gamma)
{
	Mat temp;

	parallel::gemm(W, D, 1.0, X, -1.0, temp);		// temp =  W x D - X	
	reduce(temp, temp, 0, CV_REDUCE_AVG);
	multiply(temp, temp, temp);						// temp = (W x D - X)^2
	float J1 = static_cast<float>(sum(temp)[0]);

	multiply(W, W, temp);							// temp = W^2
	temp += epsilon;								// temp = W^2 + epsilon
	sqrt(temp, temp);								// temp = sqrt(W^2 + epsilon)
	reduce(temp, temp, 0, CV_REDUCE_AVG);
	float J2 = lambda * static_cast<float>(sum(temp)[0]);

	multiply(D, D, temp);							// temp = D^2
	float J3 = gamma * static_cast<float>(sum(temp)[0]);

	float cost = J1 + J2 + J3;
	return cost;
}

} }
