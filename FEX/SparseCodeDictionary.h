// Sparse Coding Dictionary class interface
// Written by Sergey G. Kosov in 2016 for Project X (based on Xingdi (Eric) Yuan implementation: http://eric-yuan.me/sc/)
#pragma once

#include "types.h"

namespace DirectGraphicalModels { namespace fex
{
	// ================================ Sparse Coding Dictionary Class ==============================
	/**
	* @brief Sparse Coding Dictionary class 
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CSparseCodeDictionary
	{
	public:
		CSparseCodeDictionary(void) : m_dict(Mat()) {}
		virtual ~CSparseCodeDictionary(void) {}

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
		* @brief Check weather dictionary \f$\mathbb{D}\f$ is available
		* @retval true If dictionary \f$\mathbb{D}\f$ is trained or loaded from a file
		* @retval false otherwise
		*/
		DllExport bool isTrained(void) { return !m_dict.empty(); }
		/**
		* @brief Returns the words' size in dictionary
		* @returns blockSize
		*/
		DllExport int getBlockSize(void) { return m_dict.empty() ? 0 : static_cast<int>(sqrt(m_dict.rows)); }
		/**
		* @brief Returns the number of words in dictionary
		* @returns nWords
		*/
		DllExport int getNumWords(void) { return m_dict.empty() ? 0 : m_dict.cols; }
		
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
		enum sc_cost { DICT_COST, H_COST };
		
		/**
		* @brief
		* @param[in] X
		* @param[in] dict
		* @param[in] H
		* @param[out] grad
		* @param[in] lambda
		* @param[in] epsilon
		* @param[in] gamma
		* @param[in] cond
		* @returns
		*/
		static double getSparseCodingCost(const Mat &X, const Mat &dict, const Mat &H, Mat &grad, double lambda, double epsilon, double gamma, sc_cost cond);
		/**
		* @brief
		* @param[in] X
		* @param[in,out] dict
		* @param[in] H
		* @param[in] lambda
		* @param[in] epsilon
		* @param[in] gamma
		* @param[in] nIt
		* @returns
		*/
		static double trainDict(const Mat &X, Mat &dict, const Mat &H, double lambda, double epsilon, double gamma, unsigned int nIt = 800);
		/**
		* @brief
		* @param[in] X
		* @param[in] dict 
		* @param[in,out] H
		* @param[in] lambda
		* @param[in] epsilon
		* @param[in] gamma
		* @param[in] nIt
		* @returns
		*/
		static double trainH(const Mat &X, const Mat& dict, Mat &H, double lambda, double epsilon, double gamma, unsigned int nIt = 800);


	private:
		Mat		m_dict;				///< The dictionary \f$\mathbb{D}\f$: Mat(size: blockSize^2 x nWords; type: CV_64FC1); 

	};

} }

