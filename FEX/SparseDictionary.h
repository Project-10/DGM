// Sparse Dictionary class interface
// Written by Sergey G. Kosov in 2016 for Project X (based on Xingdi (Eric) Yuan implementation)
#pragma once

#include "types.h"

namespace DirectGraphicalModels { namespace fex
{
	// ================================ Sparse Dictionary Class ==============================
	/**
	* @brief Sparse Dictionary Learning class 
	* @details This class performs the <a href="https://en.wikipedia.org/wiki/Sparse_dictionary_learning">Sparse dictionary learning</a>:
	* \f[ argmin_{\mathbb{D}, \vec{h}_i} \sum^{nSamples}_{i=1}{\left\| \vec{x}_i - \mathbb{D}\times\vec{h}_i\right\|^{2}_{2} + \lambda\left\|\vec{h}_i\right\|_0 }, \f]
	* where \f$\vec{x}_i\in\mathbb{X}\in\mathbb{R}^{nSamples \times blockSize^2}\f$ is a data sample,
	* \f$\mathbb{D}\in\mathbb{R}^{nWords \times blockSize^2}\f$ is the dictionary and \f$\vec{h}_i\f$ are weighting coefficients.<br>
	* The class is based on <a href="http://eric-yuan.me/sc">Xingdi (Eric) Yuan implementation</a> 
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CSparseDictionary
	{
	public:
		CSparseDictionary(void) : m_dict(Mat()) {}
		virtual ~CSparseDictionary(void) {}

		/**
		* @brief Train dictionary \f$\mathbb{D}\f$
		* @details This function creates and trains new dictionary \f$\mathbb{D}\f$ on data \f$\mathbb{X}\f$
		* @param X Training data \f$\mathbb{X}\f$: Mat(size blockSize^2 x nSamples; type CV_64FC1)
		* @param nWords Length of the dictionary (number of words)
		* @param batch The number of randomly chosen samples from \b X to be used in every distinct iteration of training
		* > This parameter must be smaller or equal to the number of samples in training data \f$\mathbb{X}\f$
		* @param nIt Number of iterations
		*/
		DllExport void train(const Mat &X, int nWords, int batch, unsigned int nIt = 1000);
		/**
		* @brief Save dictionary \f$\mathbb{D}\f$ into a file
		* @param fileName File name
		*/
		DllExport void save(const std::string &fileName) const;
		/**
		* @brief Load dictionary \f$\mathbb{D}\f$ from the file
		* @param fileName File name
		*/
		DllExport void load(const std::string &fileName);
		/**
		* @brief Return dictionary \f$\mathbb{D}\f$
		* @returns Dictionary \f$\mathbb{D}\f$: Mat(size: blockSize^2 x nWords; type: CV_64FC1)
		*/
		DllExport Mat get(void) const { return m_dict; }
		/**
		* @brief Returns the words' size in dictionary
		* @returns blockSize
		*/
		DllExport int getBlockSize(void) { return m_dict.empty() ? 0 : static_cast<int>(sqrt(m_dict.rows)); }
		
		/**
		* @brief
		* @param X Test data \f$\mathbb{X}\f$
		* @param imgSize
		*/
		DllExport Mat decode(const Mat &X, CvSize imgSize) const;


		/**
		* @brief Converts image into data \f$\mathbb{X}\f$
		* @details This functions generates a set of data patches (\b blockSize x \b blockSize) from a single image. 
		* The extracted pathces are overlapping, thus the total number of data samples is: nSamples = (img.width - blockSize + 1) x (img.height - blockSize + 1)
		* > It is recommended to suffle the samples with shuffleCols() function before dictionary training
		* @param img The input image
		* @param blockSize Size of the quadratic patch
		* > In order to use this calss with fex::CSparseCode the size of the block should be odd
		* @returns Dictionary \f$\mathbb{X}\f$: Mat(size: blockSize^2 x nSamples; type: CV_64FC1)
		*/
		DllExport static Mat img2data(const Mat &img, int blockSize);
		/**
		* @brief Converts data \f$\mathbb{X}\f$ into an image
		* @details This function performs reverse transformation of img2data() function
		* @param X The input data \f$\mathbb{X}\f$
		* @param imgSize The size of the image to return
		* @returns Resulting image: Mat(size: \b imgSize; type: CV_8UC1)
		*/
		DllExport static Mat data2img(const Mat &X, CvSize imgSize);
		/**
		* @brief Randomly shuffles the columns of the input matrix
		* @param X The input data
		* @returns Copy of \b X with suffled columns
		*/
		DllExport static Mat shuffleCols(const Mat &X);

	
	protected:
		/**
		* @brief Evaluates dictionary \f$\mathbb{D}\f$ with given coefficients \f$\vec{h}\f$
		* @param[in] X Training data \f$\mathbb{X}\f$: Mat(size blockSize^2 x nSamples; type CV_64FC1)
		* @param[in,out] dict Dictionary \f$\mathbb{D}\f$:  Mat(size blockSize^2 x nWords; type CV_64FC1)
		* @param[in] H Weighting coefficients \f$\vec{h}\f$:  Mat(size blockSize^2 x 1; type CV_64FC1)
		* @param[in] epsilon L1-regularisation parameter: \f$|h|\approx\sqrt{h^2 + \epsilon}\f$
		* @param[in] lambda Regularisation parameter
		* @param[in] nIt Number of iterations
		* @returns The value of the cost function
		*/
		static double calculateDict(const Mat &X, Mat &dict, const Mat &H, double epsilon, double lambda, unsigned int nIt = 800);
		/**
		* @brief Evaluates coefficients \f$\vec{h}\f$ with given dictionary \f$\mathbb{D}\f$
		* @param[in] X Training data \f$\mathbb{X}\f$: Mat(size blockSize^2 x nSamples; type CV_64FC1)
		* @param[in] dict Dictionary \f$\mathbb{D}\f$:  Mat(size blockSize^2 x nWords; type CV_64FC1)
		* @param[in,out] H  Weighting coefficients \f$\vec{h}\f$:  Mat(size blockSize^2 x 1; type CV_64FC1)
		* @param[in] epsilon L1-regularisation parameter: \f$|h|\approx\sqrt{h^2 + \epsilon}\f$
		* @param[in] lambda Regularisation parameter 
		* @param[in] nIt Number of iterations
		* @returns The value of the cost function
		*/
		static double calculateH(const Mat &X, const Mat& dict, Mat &H, double epsilon, double lambda, unsigned int nIt = 800);


	protected:
		Mat		m_dict;				///< The dictionary \f$\mathbb{D}\f$: Mat(size: blockSize^2 x nWords; type: CV_64FC1); 


	private:
		enum cost_type { COST_DICT, COST_H };
		/**
		* @brief Calculates the value of the cost function and \b grad matrix
		* @param[in] cType
		* @param[in] X Training data \f$\mathbb{X}\f$: Mat(size blockSize^2 x nSamples; type CV_64FC1)
		* @param[in] dict Dictionary \f$\mathbb{D}\f$:  Mat(size blockSize^2 x nWords; type CV_64FC1)
		* @param[in] H Weighting coefficients \f$\vec{h}\f$:  Mat(size blockSize^2 x 1; type CV_64FC1)
		* @param[out] grad (hGrad or dictGrad, depending on \b cType)
		* @param[in] epsilon L1-regularisation parameter: \f$|h|\approx\sqrt{h^2 + \epsilon}\f$
		* @param[in] lambda Regularisation parameter (for hCost or gradCost, depending on \b cType)
		* @returns The value of the cost function
		*/
		static double calculateCost(cost_type cType, const Mat &X, const Mat &dict, const Mat &H, Mat &grad, double epsilon, double lambda);
	};

} }

