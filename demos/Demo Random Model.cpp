// Example "Random Model" training on 2D features
#include "DGM.h"
#include "VIS.h"
#include "FEX.h"
#include "DGM/timer.h"

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
	printf("5: OpenCV Random Forest\n");
	printf("6: MicroSoft Random Forest\n");
	printf("7: OpenCV Artificial Neural Network\n");
	printf("8: OpenCV Support Vector Machines\n");
}

// merges some classes in one
Mat shrinkStateImage(const Mat &gt, byte nStates)
{
	// assertions
	if (gt.type() != CV_8UC1) return Mat();

	Mat res = gt.clone();

	for (byte& val: static_cast<Mat_<byte>>(res)) 
		if (val < 3)		val = 0;
		else if (val < 4)	val = 1;
		else				val = 2;
	//	val = (val + 1) % nStates;

	return res;
}

int main(int argc, char *argv[])
{
	const Size	imgSize		= Size(400, 400);
	const byte	nStates		= 3;	 		
	const word	nFeatures	= 2;				// {ndvi, saturation}

	if (argc != 5) {
		print_help(argv[0]);
		return 0;
	}
	
	// Reading parameters and images
	int nodeModel	= atoi(argv[1]);
	Mat img			= imread(argv[2], 1); resize(img, img, imgSize, 0, 0, INTER_LANCZOS4);	// training image 
	Mat gt			= imread(argv[3], 0); resize(gt, gt, imgSize, 0, 0, INTER_NEAREST);	    // groundtruth for training
	gt				= shrinkStateImage(gt, nStates);										// reduce the number of classes in gt to nStates

	float Z =  1.0f;																		// the value of partition function
	if (nodeModel == 0)						Z = 2e34f;										// for Bayes model
	if (nodeModel == 6 || nodeModel == 7)	Z = 0.0f;										// for MicroSoft Random Forest and OpenCV Artificial Neural Network

	auto nodeTrainer = CTrainNode::create(nodeModel, nStates, nFeatures);
	CMarkerHistogram marker(*nodeTrainer, DEF_PALETTE_3);

	//	---------- Features Extraction ----------
	vec_mat_t featureVector;
	CCommonFeatureExtractor fExtractor(img);
	featureVector.push_back(fExtractor.getNDVI(0).autoContrast().get());
	featureVector.push_back(fExtractor.getSaturation().invert().get());

	for (int y = 0; y < gt.rows; y++)
		for (int x = 0; x < gt.cols; x++)
			if (gt.at<byte>(y, x) == 1) {
				float val = (float) featureVector[0].at<byte>(y, x);
				val = val - 10;
				featureVector[0].at<byte>(y, x) = (byte) MAX(0.0f, val + 0.5f);
			}
			else if (gt.at<byte>(y, x) == 2) {
				float val = (float) featureVector[1].at<byte>(y, x);
				val = val + 0;
				featureVector[1].at<byte>(y, x) = (byte) MAX(0.0f, val + 0.5f);
			}

	//	---------- Training ----------
	Timer::start("Training... ");
	nodeTrainer->addFeatureVecs(featureVector, gt);
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
	
	waitKey(1000);

	return 0;
}
