// Sparse Coding feature extraction class interface
// Written by Sergey G. Kosov in 2016 for Project X (based on Xingdi (Eric) Yuan implementation) http://eric-yuan.me/sc/
#pragma once

#include "BaseFeatureExtractor.h"

/**
Usage example:
@code
int main()
{
	Mat img = imread("..\\test_img_org.jpg", 0);

	// Test
	//Mat data = img2data(img);
	//Mat res = data2img(data, img.size());
	//res.convertTo(res, CV_8UC1, 255);
	//imshow("Result", res);
	//cvWaitKey();
	//return 0;


	if (true) {	// Train dict
		Mat data = img2data(img);
		Mat trainX = shuffleCols(data);					// my addition

		batch = 1000;
		Mat dict = dictionaryLearning(trainX, DICT_SIZE, 1000);
	}

	if (false) {	// Visualize dict
		Mat dict;
		//VideoWriter outputVideo;
		//outputVideo.open("dict_car.avi", CV_FOURCC_DEFAULT, 24, cvSize(300, 300), true);

		//int idx = 2;
		for (int idx = 0; idx < 200; idx++)
		{
			std::string path = "dict_";
			path += std::to_string(idx);
			path += ".txt";
			dict = readDict(path);
			Mat dictImg = renderDict(dict);
			resize(dictImg, dictImg, cvSize(300, 300), 0, 0, 0);
			imshow("Dictionary", dictImg);
			//outputVideo << dictImg;
			cvWaitKey(33);
		}
		cvWaitKey();
	}

	if (true) {		// Reconstruction
		Mat data = img2data(img);
		Mat dict = readDict("dict_199.txt");
		Mat res = decoder(data, dict, img.size());

		res.convertTo(res, CV_8UC1, 255);

		imshow("Result", res);
		cvWaitKey();
	}

	return 0;
}
@endcode
*/


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
	* Usage example:
	* @code
		int main()
		{
			Mat img = imread("..\\test_img_org.jpg", 0);

			// Test
			//Mat data = img2data(img);
			//Mat res = data2img(data, img.size());
			//res.convertTo(res, CV_8UC1, 255);
			//imshow("Result", res);
			//cvWaitKey();
			//return 0;


			if (true) {	// Train dict
				Mat data = img2data(img);
				Mat trainX = shuffleCols(data);					// my addition

				batch = 1000;
				Mat dict = dictionaryLearning(trainX, DICT_SIZE, 1000);
			}

			if (false) {	// Visualize dict
				Mat dict;
				//VideoWriter outputVideo;
				//outputVideo.open("dict_car.avi", CV_FOURCC_DEFAULT, 24, cvSize(300, 300), true);

				//int idx = 2;
				for (int idx = 0; idx < 200; idx++)
				{
					std::string path = "dict_";
					path += std::to_string(idx);
					path += ".txt";
					dict = readDict(path);
					Mat dictImg = renderDict(dict);
					resize(dictImg, dictImg, cvSize(300, 300), 0, 0, 0);
					imshow("Dictionary", dictImg);
					//outputVideo << dictImg;
				cvWaitKey(33);
				}
				cvWaitKey();
			}

			if (true) {		// Reconstruction
				Mat data = img2data(img);
				Mat dict = readDict("dict_199.txt");
				Mat res = decoder(data, dict, img.size());

				res.convertTo(res, CV_8UC1, 255);

				imshow("Result", res);
				cvWaitKey();
			}

			return 0;
		}
	@endcode
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CSC : public CBaseFeatureExtractor
	{
	public:
		/**
		* @brief Constructor.
		* @param img Input image of type \b CV_8UC3.
		*/
		DllExport CSC(const Mat &img) : CBaseFeatureExtractor(img), m_dict(Mat()) {}
		DllExport virtual ~CSC(void) {}

		/**
		* @brief
		* @param X
		* @param dictSize
		* @param batch
		* @param nIt Number of iterations
		*/
		DllExport void trainDictionary(Mat &X, int dictSize, int batch, unsigned int nIt = 1000);
		/**
		* @brief
		* @param X
		* @param imgSize 
		* @returns
		*/
		DllExport Mat decoder(const Mat &X, CvSize imgSize) const;

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
		*/
		DllExport static Mat renderDict(Mat &dict);
		/**
		* @brief Reads training data from image
		* @param path Path to the JPEG file with image
		* @returns x : ( block_size^2 X DATA_SIZE )
		*/
		DllExport static Mat img2data(const Mat &img);
		/**
		*/
		DllExport static Mat data2img(Mat &X, CvSize imgSize);
		/**
		*/
		DllExport static Mat shuffleCols(const Mat &matrix);


	protected:
		/**
		* @brief
		* @param[in] X
		* @param[in] H
		* @param[out] dictGrad
		* @param[out] hGrad
		* @param[in] lambda
		* @param[in] epsilon
		* @param[in] gamma
		* @param[in] cond
		* @returns
		*/
		double getSparseCodingCost(const Mat &X, const Mat &H, Mat &dictGrad, Mat &hGrad, double lambda, double epsilon, double gamma, sc_cost cond) const;
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
		Mat		m_dict;				//< The dictionary 		 ( block_size^2 X DICT_SIZE )

	};
} }
