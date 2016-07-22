// Sparse Dictionary class interface
// Written by Sergey G. Kosov in 2016 for Project X (based on Xingdi (Eric) Yuan implementation)
#pragma once

#include "types.h"

namespace DirectGraphicalModels { namespace fex
{
	// ================================ Sparse Dictionary Class ==============================
	/**
	* @brief Sparse Dictionary Learning class 
	* @details This class performs the <a href="https://en.wikipedia.org/wiki/Sparse_dictionary_learning">Sparse dictionary learning</a>, 
	* i.e. estimation of dictionary words (bases) \f$ d_i\f$, such that the data \f$ \vec{x} \f$ might be reconstructed through a linear combination of them:
	* \f[ \begin{align} 
	*   \sum^{nWords}_{i=1} w_{i,j} \cdot \vec{d}_i = \vec{x}_j, & & \forall j \in [0; nSamples), & \text{ or} \\
	*	\vec{w}_j \times D = \vec{x}_j, & & \forall j \in [0; nSamples) &
	* \end{align} \f]
	* The implementation is based on <a href="http://ufldl.stanford.edu/wiki/index.php/Sparse_Coding:_Autoencoder_Interpretation">Sparse Coding: Autoencoder Interpretation</a> article:
	* \f[ argmin_{D, W} J(D, W) = \left\| W \times D - X \right\|^{2}_{2} + \lambda\sum_{i,j}{\sqrt{w^{2}_{i,j} + \epsilon}} + \gamma\left\|D\right\|^{2}_{2}, \f]
	* where \f$X\in\mathbb{R}^{sampleLen \times nSamples}\f$ is the matrix, containing training data samples as row-vectors,
	* \f$D\in\mathbb{R}^{sampleLen \times nWords}\f$ is the dictionary and \f$W\in\mathbb{R}^{nWords \times nSamples}\f$ is the matrix, containing weighting coefficients.<br>
	
	* In order to train the dictionary, one may use the code:
	* @code
	*	using namespace DirectGraphicalModels::fex;	
	*
	*	CSparseCoding *sparseCoding = new CSparseCoding(img);
	*	Mat data = CSparseDictionary::img2data(img, 7);
	*	data = CSparseDictionary::shuffleRows(data);
	*	sparseCoding->train(data, nWords, 1000, 1000);
	*	sparseCoding->save("dictionary.dic");
	* @endcode
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CSparseDictionary
	{
	public:
		CSparseDictionary(void) : m_D(Mat()) {}
		virtual ~CSparseDictionary(void) {}

		/**
		* @brief Train dictionary \f$D\f$
		* @details This function creates and trains new dictionary \f$D\f$ on data \f$X\f$
		* @param X Training data \f$X\f$: Mat(size sampleLen x nSamples; type CV_64FC1)
		* @param nWords Length of the dictionary (number of words)
		* @param batch The number of randomly chosen samples from \b X to be used in every distinct iteration of training
		* > This parameter must be smaller or equal to the number of samples in training data \f$X\f$
		* @param nIt Number of iterations
		*/
		DllExport void train(const Mat &X, word nWords, dword batch = 2000, unsigned int nIt = 1000);
		/**
		* @brief Save dictionary \f$D\f$ into a file
		* @param fileName File name
		*/
		DllExport void save(const std::string &fileName) const;
		/**
		* @brief Load dictionary \f$D\f$ from the file
		* @param fileName File name
		*/
		DllExport void load(const std::string &fileName);
		/**
		* @brief Return dictionary \f$D\f$
		* @returns Dictionary \f$D\f$: Mat(size: blockSize^2 x nWords; type: CV_64FC1)
		*/
		DllExport Mat getDictionary(void) const { return m_D; }
		/**
		* @brief Returns the words' size in dictionary
		* @returns blockSize
		*/
		DllExport int getBlockSize(void) { return m_D.empty() ? 0 : static_cast<int>(sqrt(m_D.cols)); }
		
		/**
		* @brief
		* @param X Test data \f$X\f$
		* @param imgSize
		*/
		DllExport Mat decode(const Mat &X, CvSize imgSize) const;


		/**
		* @brief Converts image into data \f$X\f$
		* @details This functions generates a set of data patches (\b blockSize x \b blockSize) from a single image. 
		* The extracted pathces are overlapping, thus the total number of data samples is: nSamples = (img.width - blockSize + 1) x (img.height - blockSize + 1)
		* > It is recommended to suffle the samples with shuffleRows() function before dictionary training
		* @param img The input image
		* @param blockSize Size of the quadratic patch
		* > In order to use this calss with fex::CSparseCoding::get() the size of the block should be odd
		* @param varianceThreshold 
		* @returns Data \f$X\f$: Mat(size: blockSize^2 x nSamples; type: CV_64FC1)
		*/
		DllExport static Mat img2data(const Mat &img, int blockSize, double varianceThreshold = 0.0);
		/**
		* @brief Converts data \f$X\f$ into an image
		* @details This function performs reverse transformation of img2data() function, thus the code
		* @code
		*		Mat data = CSparseDictionary::img2data(img, 7);
		*		Mat res  = CSparseDictionary::data2img(data, img.size());
		* @endcode
		* gives \a res identical to \a img.
		* @param data The input data \f$\mathbb{X}\f$
		* @param imgSize The size of the image to return
		* @returns Resulting image: Mat(size: \b imgSize; type: CV_8UC1)
		*/
		DllExport static Mat data2img(const Mat &X, CvSize imgSize);
		/**
		* @brief Randomly shuffles the rows of the input matrix
		* @param data The input data
		* @returns Copy of \b X with suffled columns
		*/
		DllExport static Mat shuffleRows(const Mat &X);

	
	protected:
		/**
		* @brief Evaluates weighting coefficients matrix \f$W\f$
		* @details Find the \f$W\f$, that minimizes \f$J(D, W)\f$ for the given \f$D\f$: 
		* \f[ argmin_{W} J(D, W) = \left\| W \times D - X \right\|^{2}_{2} + \lambda\sum_{i,j}{\sqrt{w^{2}_{i,j} + \epsilon}} \f]
		* @param[in] X Training data \f$X\f$: Mat(size nSamples x sampleLen; type CV_64FC1)
		* @param[in] D Dictionary \f$D\f$:  Mat(size nWords x sampleLen; type CV_64FC1)
		* @param[in,out] W  Weighting coefficients \f$W\f$:  Mat(size nSamples x nWords; type CV_64FC1)
		* @param[in] lambda Regularisation parameter \f$\lambda\f$
		* @param[in] epsilon L1-regularisation parameter: \f$\epsilon\f$
		* @param[in] nIt Number of iterations
		*/
		static void calculate_W(const Mat &X, const Mat& D, Mat &W, double lambda, double epsilon, unsigned int nIt = 800);
		/**
		* @brief Evaluates dictionary \f$D\f$
		* @details Solve for the \f$D\f$ that minimizes \f$J(D, W)\f$ for the given \f$W\f$:
		* \f[ argmin_{D} J(D, W) = \left\| W \times D - X \right\|^{2}_{2} + \gamma\left\|D\right\|^{2}_{2}, \f]
		* @param[in] X Training data \f$X\f$: Mat(size nSamples x sampleLen; type CV_64FC1)
		* @param[in,out] D Dictionary \f$D\f$:  Mat(size nWords x sampleLen; type CV_64FC1)
		* @param[in] W Weighting coefficients \f$W\f$:  Mat(size nSamples x nWords; type CV_64FC1)
		* @param[in] gamma Regularisation parameter: \f$\gamma\f$
		* @param[in] nIt Number of iterations
		*/
		static void calculate_D(const Mat &X, Mat &D, const Mat &W, double gamma, unsigned int nIt = 800);



	protected:
		Mat		m_D;					///< The dictionary \f$\mathbb{D}\f$: Mat(size: blockSize^2 x nWords; type: CV_64FC1); 


	protected:
		enum grad_type { GRAD_D, GRAD_W };
		/**
		* @brief Calculates the gradient matrix
		* @param[in] gType
		* @returns Gradient matrix
		*/
		static Mat calculateGradient(grad_type gType, const Mat &X, const Mat &D, const Mat &W, double lambda, double epsilon, double gamma);
		/**
		* @brief Calculates the value of the cost function and \b grad matrix
		* @param[in] X Training data \f$\mathbb{X}\f$: Mat(size blockSize^2 x nSamples; type CV_64FC1)
		* @param[in] dict Dictionary \f$\mathbb{D}\f$:  Mat(size blockSize^2 x nWords; type CV_64FC1)
		* @param[in] H Weighting coefficients \f$\vec{h}\f$:  Mat(size blockSize^2 x 1; type CV_64FC1)
		* @param[out] grad (hGrad or dictGrad, depending on \b cType)
		* @param[in] epsilon L1-regularisation parameter: \f$|h|\approx\sqrt{h^2 + \epsilon}\f$
		* @param[in] lambda Regularisation parameter (for hCost or gradCost, depending on \b cType)
		* @returns The value of the cost function
		*/
		static double calculateCost(const Mat &X, const Mat &D, const Mat &W, double lambda, double epsilon, double gamma);
	};

} }

