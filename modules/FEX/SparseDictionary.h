// Sparse Dictionary class interface
// Written by Sergey G. Kosov in 2016 for Project X (based on Xingdi (Eric) Yuan implementation)
#pragma once

#include "types.h"

namespace DirectGraphicalModels { namespace fex
{
	const float SC_LRATE_W = 5e-2f;		///< Learning rate (speed) for weights \f$W\f$
	const float	SC_LRATE_D = 1e-2f;		///< Learning rate (speed) for dictionary \f$D\f$

	const float	SC_LAMBDA  = 5e-5f;		///< \f$\lambda\f$:  L1-regularisation parameter (on features)
	const float	SC_EPSILON = 1e-5f;		///< \f$\epsilon\f$: L1-regularisation epsilon \f$ \left\|x\right\|_1 \approx  \sqrt{x^2 + \epsilon} \f$
	const float	SC_GAMMA   = 1e-2f;		///< \f$\gamma\f$:   L2-regularisation parameter (on dictionary words)


	// ================================ Sparse Dictionary Class ==============================
	/**
	* @brief Sparse Dictionary Learning class
	* @details This class performs the <a href="https://en.wikipedia.org/wiki/Sparse_dictionary_learning">Sparse dictionary learning</a>,
	* i.e. estimation of dictionary words (bases) \f$\vec{d}_i\in D\f$, such that arbitrary data vector\f$\vec{x}\f$ could be represended via a linear combination:
	* \f[ \begin{align}
	*   \sum^{nWords}_{i=1} w_{i,j} \cdot \vec{d}_i = \vec{x}_j, & & \forall j \in [0; nSamples), & \text{  or} \\
	*	\vec{w}_j \times D = \vec{x}_j, & & \forall j \in [0; nSamples), &
	* \end{align} \f]
	* where \f$D\f$ is the dictionary.<br>
	* The implementation is based on <a href="http://ufldl.stanford.edu/wiki/index.php/Sparse_Coding:_Autoencoder_Interpretation">Sparse Coding: Autoencoder Interpretation</a> article,
	* where the task of the dictionary learning turns into the following minimization problem:
	* \f[ \text{arg}\,\min\limits_{D,W} J(D, W) = \left\| W \times D - X \right\|^{2}_{2} + \lambda\left\|W\right\|_1 + \gamma\left\|D\right\|^{2}_{2}, \f]
	* where \f$D\in\mathbb{R}^{sampleLen \times nWords}\f$, \f$W\in\mathbb{R}^{nWords \times nSamples}\f$ is the matrix, containing weighting coefficients for every word and every sample
	* and \f$X\in\mathbb{R}^{sampleLen \times nSamples}\f$ contains the training data as row-vectors samples.<br>
	* In order to minimize \f$J(D, W)\f$ we use the <a href="https://en.wikipedia.org/wiki/Gradient_descent">Gradient Descent</a> algorithm.  
	* We also use \f$\sum_{i,j}\sqrt{w^{2}_{i,j} + \epsilon}\f$ in place of \f$\left\|W\right\|_1\f$ to make \f$J(D, W)\f$ differentiable at \f$W = 0\f$.<br>
	* In order to train the dictionary, one may use the code:
	* @code
	* using namespace DirectGraphicalModels;
	* using namespace DirectGraphicalModels::fex;
	*
	*	CSparseCoding *sparseCoding = new CSparseCoding(img);
	*	Mat X = CSparseDictionary::img2data(img, blockSize);	// sampleLen = blockSize * blockSize
	*	parallel::shuffleRows(X);
	*	sparseCoding->train(X, nWords);
	*	sparseCoding->save("dictionary.dic");
	* @endcode
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CSparseDictionary
	{
	public:
		CSparseDictionary(void) : m_D(Mat()) { }
		virtual ~CSparseDictionary(void) {}

		/**
		* @brief Trains dictionary \f$D\f$
		* @details This function creates and trains new dictionary \f$D\f$ on data \f$X\f$
		* @param X Training data \f$X\f$: Mat(size nSamples x sampleLen; type CV_8UC1 or CV_16UC1)
		* > May be derived from an image with img2data() fucntion
		* @param nWords Length of the dictionary (number of words)
		* @param batch The number of randomly chosen samples from \b X to be used in every distinct iteration of training
		* > This parameter must be smaller or equal to the number of samples in training data \f$X\f$
		* @param nIt Number of iterations
		* @param lRate Learning rate parameter, which is charged with the speed of convergence
		* @param fileName Path and file name to store intermediate dictionaries \f$D\f$ (every 5 iterations).
		* If specified the resulting file name will be the follows: \b fileName<it/5>.dic
		*/
		DllExport void train(const Mat &X, word nWords, dword batch = 2000, unsigned int nIt = 1000, float lRate = SC_LRATE_D, const std::string &fileName = std::string());
		/**
		* @brief Saves dictionary \f$D\f$ into a binary file
		* @param fileName Full file name
		*/
		DllExport void save(const std::string &fileName) const;
		/**
		* @brief Loads dictionary \f$D\f$ from the file
		* @param fileName Full file name
		*/
		DllExport void load(const std::string &fileName);
		/**
		* @brief Checks whether the dictionary has been trained or loaded
		* @retval true if the dictionary has been trained or loaded
		* @retval false otherwise
		*/
		DllExport bool empty(void) const { return m_D.empty(); }
		/**
		* @brief Returns dictionary \f$D\f$
		* @returns Dictionary \f$D\f$: Mat(size: nWords x sampleLen; type: CV_32FC1)
		*/
		DllExport Mat getDictionary(void) const { return m_D; }
		/**
		* @brief Returns dictionary \f$D\f$ from file
		* @param fileName Full file name
		* @returns Dictionary \f$D\f$: Mat(size: nWords x sampleLen; type: CV_32FC1)
		*/			
		DllExport static Mat getDictionary(const std::string &fileName)
		{
			CSparseDictionary me;
			me.load(fileName);
			return me.getDictionary();
		}
		/**
		* @brief Returns size of the block, i.e. \f$\sqrt{sampleLen}\f$
		* @returns \b blockSize
		*/
		DllExport int getBlockSize(void) const { return empty() ? 0 : static_cast<int>(sqrt(m_D.cols)); }
		/**
		* @brief Returns the number words in dictionary \f$D\f$
		* @returns \b nWords
		*/
		DllExport word getNumWords(void) const { return empty() ? 0 : static_cast<word>(m_D.rows); }

#ifdef DEBUG_MODE	// --- Debugging ---
		/**
		* @brief Decodes an image from the data \f$X\f$
		* @details This is a debug function, which restores an image from the data \f$X\f$ with the current dictionary \f$D\f$.<br>
		* Thus the code:
		* @code
		*	using namespace DirectGraphicalModels::fex;
		*
		*	Mat img = imread("image.jpg");
		*	Mat X = CSparseDictionary::img2data(img, sparseDictionary.getBlockSize());
		*	CSparseDictionary sparseDictionary;
		*	sparseDictionary.load("dictionary.dic");
		*	Mat res = sparseDictionary.TEST_decode(X, img.size());
		* @endcode
		* should produce \a res very close to \a img.
		* > This function supports PPL
		* @param X Test data \f$X\f$
		* @param imgSize Size of the resulting image
		* @returns Decoded image: Mat(size: \b imgSize; type: CV_8UC1)
		*/
		DllExport Mat TEST_decode(const Mat &X, CvSize imgSize) const;
#endif				// --- --------- ---

		/**
		* @brief Converts image into data \f$X\f$
		* @details This functions generates a set of data samples (\b blockSize x \b blockSize patches) from a single image.
		* The extracted pathces are overlapping, thus the maximal number of data samples is: nMaxSamples = (img.width - \b blockSize + 1) x (img.height - \b blockSize + 1)
		* > It is recommended to suffle the samples with parallel::shuffleRows() function before training dictionary with train()
		* @param img The input image (1 or 3 channels, 8 or 16 bit image)
		* @param blockSize Size of the quadratic patch
		* > In order to use this calss with fex::CSparseCoding::get() the size of the block should be odd
		* @param varianceThreshold The extracted samples with variance greater or equal to \b varianceThreshold will be included to \f$X\f$. 
		* > If \b varianceThreshold = 0 all the samples are included, thus nSamples = nMaxSamples
		* @returns Data \f$X\f$: Mat(size: nSamples x \b blockSize^2; type: CV_8UC1 or CV_16UC1)
		*/
		DllExport static Mat img2data(const Mat &img, int blockSize, float varianceThreshold = 0.0f);
		/**
		* @brief Converts data \f$X\f$ into an image
		* @details This function performs reverse transformation of img2data() function, thus the code
		* @code
		*		Mat X   = CSparseDictionary::img2data(img, blockSize, 0);
		*		Mat res = CSparseDictionary::data2img(X, img.size());
		* @endcode
		* gives \a res identical to \a img.
		* @param X The input data \f$X\f$ (depth: 8 or 16 bit)
		* @param imgSize The size of the image to return
		* @returns Resulting image: Mat(size: \b imgSize; type: CV_8UC1 or CV_16UC1)
		*/
		DllExport static Mat data2img(const Mat &X, CvSize imgSize);


	protected:
		/**
		* @brief Evaluates weighting coefficients matrix \f$W\f$
		* @details Finds the \f$W\f$, that minimizes \f$J(D, W)\f$ for the given \f$D\f$:
		* \f[ \text{arg}\,\min\limits_{W} J(D, W) = \left\| W \times D - X \right\|^{2}_{2} + \lambda\sum_{i,j}{\sqrt{w^{2}_{i,j} + \epsilon}} \f]
		* @param[in] X Training data \f$X\f$: Mat(size nSamples x sampleLen; type CV_32FC1)
		* @param[in] D Dictionary \f$D\f$:  Mat(size nWords x sampleLen; type CV_32FC1)
		* @param[in,out] W  Weighting coefficients \f$W\f$:  Mat(size nSamples x nWords; type CV_32FC1)
		* @param[in] lambda Regularisation parameter \f$\lambda\f$
		* @param[in] epsilon L1-regularisation parameter: \f$\epsilon\f$
		* @param[in] nIt Number of iterations
		* @param[in] lRate Learning rate parameter, which is charged with the speed of convergence
		*/
		DllExport static void calculate_W(const Mat &X, const Mat& D, Mat &W, float lambda, float epsilon, unsigned int nIt = 800, float lRate = SC_LRATE_W);
		/**
		* @brief Evaluates dictionary \f$D\f$
		* @details Finds the \f$D\f$, that minimizes \f$J(D, W)\f$ for the given \f$W\f$:
		* \f[ \text{arg}\,\min\limits_{D} J(D, W) = \left\| W \times D - X \right\|^{2}_{2} + \gamma\left\|D\right\|^{2}_{2}, \f]
		* @param[in] X Training data \f$X\f$: Mat(size nSamples x sampleLen; type CV_32FC1)
		* @param[in,out] D Dictionary \f$D\f$:  Mat(size nWords x sampleLen; type CV_32FC1)
		* @param[in] W Weighting coefficients \f$W\f$:  Mat(size nSamples x nWords; type CV_32FC1)
		* @param[in] gamma Regularisation parameter: \f$\gamma\f$
		* @param[in] nIt Number of iterations
		* @param[in] lRate Learning rate parameter, which is charged with the speed of convergence
		*/
		DllExport static void calculate_D(const Mat &X, Mat &D, const Mat &W, float gamma, unsigned int nIt = 800, float lRate = SC_LRATE_D);


	private:
		Mat		m_D;					///< The dictionary \f$D\f$: Mat(size: nWords x sampleLen; type: CV_32FC1); 


	protected:
		enum grad_type { GRAD_D, GRAD_W };
		/**
		* @brief Calculates the gradient matrices \f$\frac{\partial J(D, W)}{\partial D}\f$ and \f$\frac{\partial J(D, W)}{\partial W}\f$
		* @details This function calculates the gradient matrices:
		* \f[\begin{align}
		* \frac{\partial J(D, W)}{\partial D} &= \frac{2}{nSapmles} \cdot W^\top\times [W \times D - X]  + 2\cdot\gamma\cdot D,                    &\text{  if gType = GRAD_D} \\
		* \frac{\partial J(D, W)}{\partial W} &= \frac{2}{nSapmles} \cdot [W \times D - X] \times D^\top + \lambda\frac{W}{\sqrt{W^2 + \epsilon}}, &\text{  if gType = GRAD_W} 
		* \end{align}\f]
		* @param gType Gradient type
		* @param X Training data \f$X\f$: Mat(size nSamples x sampleLen; type CV_32FC1)
		* @param D Dictionary \f$D\f$:  Mat(size nWords x sampleLen; type CV_32FC1)
		* @param W Weighting coefficients \f$W\f$:  Mat(size nSamples x nWords; type CV_32FC1)
		* @param lambda Regularisation parameter \f$\lambda\f$
		* @param epsilon L1-regularisation parameter: \f$\epsilon\f$
		* @param gamma Regularisation parameter: \f$\gamma\f$
		* @returns Gradient matrix
		*/
		static Mat calculateGradient(grad_type gType, const Mat &X, const Mat &D, const Mat &W, float lambda, float epsilon, float gamma);
		/**
		* @brief Calculates the value of \f$J(D, W)\f$ function
		* @param X Training data \f$X\f$: Mat(size nSamples x sampleLen; type CV_32FC1)
		* @param D Dictionary \f$D\f$:  Mat(size nWords x sampleLen; type CV_32FC1)
		* @param W Weighting coefficients \f$W\f$:  Mat(size nSamples x nWords; type CV_32FC1)
		* @param lambda Regularisation parameter \f$\lambda\f$
		* @param epsilon L1-regularisation parameter: \f$\epsilon\f$
		* @param gamma Regularisation parameter: \f$\gamma\f$
		* @returns The value of \f$J(D, W)\f$ function
		*/
		static float calculateCost(const Mat &X, const Mat &D, const Mat &W, float lambda, float epsilon, float gamma);
	};

} }

