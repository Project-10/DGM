// Sparse Coding feature extraction class interface
// Written by Sergey G. Kosov in 2016 for Project X (based on Xingdi (Eric) Yuan implementation) http://eric-yuan.me/sc/
#pragma once

#include "types.h"
#include "SquareNeighborhood.h"

namespace DirectGraphicalModels { namespace fex 
{
	/**
	*/
	enum sc_cost {
		DICT_COST,		///<
		H_COST			///<  
	};


	// ================================ SC Class ==============================
	/**
	* @brief Sparse Coding feature extraction class. (http://www.scholarpedia.org/article/Sparse_coding)
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CSparseCode 
	{
	public:
		/**
		* @brief Constructor.
		*/
		DllExport CSparseCode(/*const Mat &img*/) : /*CBaseFeatureExtractor(img),*/ m_dict(Mat()) {}
		DllExport virtual ~CSparseCode(void) {}

		/**
		* @brief
		* @param X
		* @param nWords Length of the dictionary
		* @param batch
		* @param nIt Number of iterations
		*/
		DllExport void trainDictionary(const Mat &X, int nWords, int batch, unsigned int nIt = 1000);
		/**
		* @brief
		* @param X
		* @param imgSize 
		* @returns
		*/
		DllExport Mat decoder(const Mat &X, CvSize imgSize) const;
		
		DllExport Mat get(const Mat &img, SqNeighbourhood nbhd = sqNeighbourhood(3));
		
		/**
		* @brief Saves Dictionary \f$\mathbb{D}\f$ into a file
		* @param fileName File name
		*/
		DllExport void saveDictionary(const std::string &fileName) const;
		/**
		* @brief Loads Dictionary \f$\mathbb{D}\f$ from the file
		* @param fileName File name
		*/
		DllExport void loadDictionary(const std::string &fileName);
		/**
		* @brief Returns the Dictionary \f$\mathbb{D}\f$
		* @returns Dictionary \f$\mathbb{D}\f$
		*/
		DllExport Mat getDictionary(void) const { return m_dict; }
		/**
		* @brief 
		* @param img
		* @param blockSize
		* @returns x : ( blockSize^2 X nStamples )
		*/
		DllExport static Mat img2data(const Mat &img, int blockSize);
		/**
		* @brief
		* @param X
		* @param imgSize
		* @returns
		*/
		DllExport static Mat data2img(const Mat &X, CvSize imgSize);
		/**
		* @brief
		* @param matrix
		* @returns
		*/
		DllExport static Mat shuffleCols(const Mat &matrix);


	protected:
		/**
		* @brief
		* @param[in] X
		* @param[in] H
		* @param[out] grad
		* @param[in] lambda
		* @param[in] epsilon
		* @param[in] gamma
		* @param[in] cond
		* @returns
		*/
		double getSparseCodingCost(const Mat &X, const Mat &H, Mat &grad, double lambda, double epsilon, double gamma, sc_cost cond) const;
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
		double trainingDict(const Mat &X, Mat &dict, const Mat &H, double lambda, double epsilon, double gamma, unsigned int nIt = 800);
		/**
		* @brief
		* @param[in] X
		* @param[in,out] H
		* @param[in] lambda
		* @param[in] epsilon
		* @param[in] gamma
		* @param[in] nIt
		* @returns
		*/
		double trainingH(const Mat &X, Mat &H, double lambda, double epsilon, double gamma, unsigned int nIt = 800) const;


	private:
		Mat		m_dict;				//< The dictionary 		 ( blockSize^2 X nWords )

	};
} }
