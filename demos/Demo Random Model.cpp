// Example "Random Model" training on 2D features
#include "DGM.h"
#include "VIS.h"
#include "FEX.h"
#include "DGM\timer.h"

using namespace DirectGraphicalModels;
using namespace DirectGraphicalModels::vis;
using namespace DirectGraphicalModels::fex;

void print_help(char *argv0)
{
	printf("Usage: %s node_training_model training_image groundtruth_image output_image\n", argv0);

	printf("\nNode training models:\n");
	printf("0: Bayes\n");
	printf("1: Gaussian Mixture Model\n");
	printf("2: OpenCV Gaussian Mixture Model\n");
	printf("3: Nearest Neighbor\n");
	printf("4: OpenCV Nearest  Neighbor\n");
	printf("5: Support Vector Machines\n");
	printf("6: OpenCV Random Forest\n");
	printf("7: MicroSoft Random Forest\n");
	printf("8: OpenCV Artificial Neural Network\n");
}

// merges some classes in one
Mat shrinkStateImage(const Mat &gt, byte nStates)
{
	// assertions
	if (gt.type() != CV_8UC1) return Mat();

	Mat res;
	gt.copyTo(res);

	for (auto it = res.begin<byte>(); it != res.end<byte>(); it++)
		if (*it < 3) *it = 0;
		else if (*it < 4) *it = 1;
		else *it = 2;
//		*it = (*it + 1) % nStates;

	return res;
}

int main(int argc, char *argv[])
{
	const CvSize		imgSize		= cvSize(400, 400);
	const unsigned int	nStates		= 3;	 		
	const unsigned int	nFeatures	= 2;		// {ndvi, saturation}

	if (argc != 5) {
		print_help(argv[0]);
		return 0;
	}
	
	// Reading parameters and images
	int nodeModel	= atoi(argv[1]);
	Mat img			= imread(argv[2], 1); resize(img, img, imgSize, 0, 0, INTER_LANCZOS4);	// training image 
	Mat gt			= imread(argv[3], 0); resize(gt, gt, imgSize, 0, 0, INTER_NEAREST);	    // groundtruth for training
	gt				= shrinkStateImage(gt, nStates);										// reduce the number of classes in gt to nStates

	float Z;																				// the value of partition function
	CTrainNode	* nodeTrainer = NULL;
	switch(nodeModel) {
		case 0: nodeTrainer = new CTrainNodeNaiveBayes(nStates, nFeatures);	Z = 2e34f; break;
		case 1: nodeTrainer = new CTrainNodeGMM(nStates, nFeatures);		Z = 1.0f; break;
		case 2: nodeTrainer = new CTrainNodeCvGMM(nStates, nFeatures);		Z = 1.0f; break;
		case 3: nodeTrainer = new CTrainNodeKNN(nStates, nFeatures);		Z = 1.0f; break;
		case 4: nodeTrainer = new CTrainNodeCvKNN(nStates, nFeatures);		Z = 1.0f; break;
		case 5: nodeTrainer = new CTrainNodeCvSVM(nStates, nFeatures);		Z = 1.0f; break;
		case 6: nodeTrainer = new CTrainNodeCvRF(nStates, nFeatures);		Z = 1.0f; break;
#ifdef USE_SHERWOOD
		case 7: nodeTrainer = new CTrainNodeMsRF(nStates, nFeatures);		Z = 1.0f; break;
#endif
		case 8: nodeTrainer = new CTrainNodeCvANN(nStates, nFeatures);		Z = 1.0f; break;
		default: printf("Unknown node_training_model is given\n"); print_help(argv[0]); return 0;
	}
	CMarkerHistogram marker(nodeTrainer, DEF_PALETTE_3);

	//	---------- Features Extraction ----------
	vec_mat_t featureVector;
	fex::CCommonFeatureExtractor fExtractor(img);
	featureVector.push_back(fExtractor.getNDVI(0).autoContrast().get());
	featureVector.push_back(fExtractor.getSaturation().invert().get());

	//	---------- Training ----------
	Timer::start("Training... ");
	nodeTrainer->addFeatureVec(featureVector, gt);
	nodeTrainer->train();
	Timer::stop();

	//	---------- Visualization ----------
	if (nodeModel == 0) {
		imshow("histogram 1d", marker.drawHistogram());
		imshow("histogram 2d", marker.drawHistogram2D());
	}

	Timer::start("Classifying...");
	Mat classMap = marker.drawClassificationMap2D(Z);
	Timer::stop();
	imwrite(argv[4], classMap);

	imshow("class map 2d", classMap);
	cvWaitKey(0*1000);

	return 0;
}




/*

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#define	NTRAINING_SAMPLES	100			// Number of training samples per class
#define FRAC_LINEAR_SEP		0.9f	    // Fraction of samples which compose the linear separable part

using namespace cv;
using namespace cv::ml;
using namespace std;

static void help()
{
	cout << "\n--------------------------------------------------------------------------" << endl
		<< "This program shows Support Vector Machines for Non-Linearly Separable Data. " << endl
		<< "Usage:" << endl
		<< "./non_linear_svms" << endl
		<< "--------------------------------------------------------------------------" << endl
		<< endl;
}

int main()
{
	help();

	// Data for visual representation
	const int WIDTH = 512, HEIGHT = 512;
	Mat I = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);

	//--------------------- 1. Set up training data randomly ---------------------------------------
	Mat trainData(2 * NTRAINING_SAMPLES, 2, CV_32FC1);
	Mat labels(2 * NTRAINING_SAMPLES, 1, CV_32SC1);

	RNG rng(100); // Random value generation class

				  // Set up the linearly separable part of the training data
	int nLinearSamples = (int)(FRAC_LINEAR_SEP * NTRAINING_SAMPLES);

	//! [setup1]
	// Generate random points for the class 1
	Mat trainClass = trainData.rowRange(0, nLinearSamples);
	// The x coordinate of the points is in [0, 0.4)
	Mat c = trainClass.colRange(0, 1);
	rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(0.4 * WIDTH));
	// The y coordinate of the points is in [0, 1)
	c = trainClass.colRange(1, 2);
	rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(HEIGHT));

	// Generate random points for the class 2
	trainClass = trainData.rowRange(2 * NTRAINING_SAMPLES - nLinearSamples, 2 * NTRAINING_SAMPLES);
	// The x coordinate of the points is in [0.6, 1]
	c = trainClass.colRange(0, 1);
	rng.fill(c, RNG::UNIFORM, Scalar(0.6*WIDTH), Scalar(WIDTH));
	// The y coordinate of the points is in [0, 1)
	c = trainClass.colRange(1, 2);
	rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(HEIGHT));
	//! [setup1]

	//------------------ Set up the non-linearly separable part of the training data ---------------
	//! [setup2]
	// Generate random points for the classes 1 and 2
	trainClass = trainData.rowRange(nLinearSamples, 2 * NTRAINING_SAMPLES - nLinearSamples);
	// The x coordinate of the points is in [0.4, 0.6)
	c = trainClass.colRange(0, 1);
	rng.fill(c, RNG::UNIFORM, Scalar(0.4*WIDTH), Scalar(0.6*WIDTH));
	// The y coordinate of the points is in [0, 1)
	c = trainClass.colRange(1, 2);
	rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(HEIGHT));
	//! [setup2]
	//------------------------- Set up the labels for the classes ---------------------------------
	labels.rowRange(0, NTRAINING_SAMPLES).setTo(1);  // Class 1
	labels.rowRange(NTRAINING_SAMPLES, 2 * NTRAINING_SAMPLES).setTo(2);  // Class 2

																		 //------------------------ 2. Set up the support vector machines parameters --------------------
																		 //------------------------ 3. Train the svm ----------------------------------------------------
	cout << "Starting training process" << endl;
	//! [init]
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::NU_SVC);
	svm->setC(0.1);
	svm->setNu(0.1);
	svm->setKernel(SVM::INTER);
	svm->setGamma(0.1);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)1e7, 1e-6));
	//! [init]
	//! [train]
	svm->train(trainData, ROW_SAMPLE, labels);
	//! [train]
	cout << "Finished training process" << endl;

	//------------------------ 4. Show the decision regions ----------------------------------------
	//! [show]
	Vec3b green(0, 100, 0), blue(100, 0, 0);
	for (int i = 0; i < I.rows; ++i)
		for (int j = 0; j < I.cols; ++j)
		{
			Mat sampleMat = (Mat_<float>(1, 2) << i, j);
			float response = svm->predict(sampleMat);

			if (response == 1)    I.at<Vec3b>(j, i) = green;
			else if (response == 2)    I.at<Vec3b>(j, i) = blue;
		}
	//! [show]

	//----------------------- 5. Show the training data --------------------------------------------
	//! [show_data]
	int thick = -1;
	int lineType = 8;
	float px, py;
	// Class 1
	for (int i = 0; i < NTRAINING_SAMPLES; ++i)
	{
		px = trainData.at<float>(i, 0);
		py = trainData.at<float>(i, 1);
		circle(I, Point((int)px, (int)py), 3, Scalar(0, 255, 0), thick, lineType);
	}
	// Class 2
	for (int i = NTRAINING_SAMPLES; i <2 * NTRAINING_SAMPLES; ++i)
	{
		px = trainData.at<float>(i, 0);
		py = trainData.at<float>(i, 1);
		circle(I, Point((int)px, (int)py), 3, Scalar(255, 0, 0), thick, lineType);
	}
	//! [show_data]

	//------------------------- 6. Show support vectors --------------------------------------------
	//! [show_vectors]
	thick = 2;
	lineType = 8;
	Mat sv = svm->getUncompressedSupportVectors();

	for (int i = 0; i < sv.rows; ++i)
	{
		const float* v = sv.ptr<float>(i);
		circle(I, Point((int)v[0], (int)v[1]), 6, Scalar(128, 128, 128), thick, lineType);
	}
	//! [show_vectors]

	imwrite("result.png", I);	                   // save the Image
	imshow("SVM for Non-Linear Training Data", I); // show it to the user
	waitKey(0);
}
*/