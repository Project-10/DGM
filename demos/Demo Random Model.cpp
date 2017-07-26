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