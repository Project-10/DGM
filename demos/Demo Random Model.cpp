// Example "Random Model" training on 2D features
#include "DGM.h"
#include "VIS.h"
#include "FEX.h"

using namespace DirectGraphicalModels;
using namespace DirectGraphicalModels::vis;
using namespace DirectGraphicalModels::fex;

void print_help(void)
{
	printf("Usage: \"Demo Random Model.exe\" node_training_model edge_training_model training_image_features training_groundtruth_image testing_image_features testing_groundtruth_image original_image output_image\n");

	printf("\nNode training models:\n");
	printf("0: Naive Bayes\n");
	printf("1: Gaussian Mixture Model\n");
	printf("2: OpenCV Gaussian Mixture Model\n");
	printf("3: Nearest Neighbor\n");
	printf("4: OpenCV Random Forest\n");
	printf("5: MicroSoft Random Forest\n");
}

// merges some classes in one
Mat shrinkStateImage(Mat &img, byte nStates)
{
	// assertions
	if (img.channels() != 1) return Mat();

	Mat res;
	img.copyTo(res);

	for (register int y = 0; y < img.rows; y++) {
		byte *pImg = img.ptr<byte>(y);
		byte *pRes = res.ptr<byte>(y);
		for (register int x = 0; x < img.cols; x++) {
			switch (pImg[x]) {
				case 0: pRes[x] = 0; break;
				case 1: pRes[x] = 0; break;
				case 2: pRes[x] = 0; break;
				case 3: pRes[x] = 1; break;
				case 4: pRes[x] = 2; break;
				case 5: pRes[x] = 2; break;
			}
			
			
			//if (pImg[x] <= 2) pRes[x] = 0;
			//else if (pImg[x] <= 3) pRes[x] = 1;
			//else pRes[x] = 2;
			//if (pImg[x] < nStates / 3) pRes[x] = 0;
			//else if (pImg[x] < 2 * nStates / 3) pRes[x] = 1;
			//else pRes[x] = 2;
		} // x
	} // y

	return res;
}

int main(int argv, char *argc[])
{
	const CvSize		imgSize = cvSize(100, 100);
	const int			width = imgSize.width;
	const int			height = imgSize.height;
	const unsigned int	nStates = 3;	 		
	const unsigned int	nFeatures = 2;

	if (argv != 4) {
		print_help();
		return 0;
	}
	
	// Reading parameters and images
	int nodeModel = 3; // atoi(argc[1]);
	Mat train_img = imread(argc[2], 1); resize(train_img, train_img, imgSize, 0, 0, INTER_LANCZOS4);	// training image feature vector
	Mat train_gt  = imread(argc[3], 0); resize(train_gt, train_gt, imgSize, 0, 0, INTER_NEAREST);	// groundtruth for training
	train_gt = shrinkStateImage(train_gt, nStates);


	float Z;
	CTrainNode	* nodeTrainer = NULL;
	switch(nodeModel) {
		case 0: nodeTrainer = new CTrainNodeNaiveBayes(nStates, nFeatures);	Z = 5e34f; break;
		case 1: nodeTrainer = new CTrainNodeGMM(nStates, nFeatures);		Z = 1.0f;break;
		case 2: nodeTrainer = new CTrainNodeCvGMM(nStates, nFeatures);		Z = 1.0f; break;
		case 3: nodeTrainer = new CTrainNodeKNN(nStates, nFeatures);		Z = 1.0f; break;
		case 4: nodeTrainer = new CTrainNodeCvRF(nStates, nFeatures);		Z = 1.0f; break;
#ifdef USE_SHERWOOD
		case 5: nodeTrainer = new CTrainNodeMsRF(nStates, nFeatures);		Z = 1.0f; break;
#endif
		default: printf("Unknown node_training_model is given\n"); print_help(); return 0;
	}
	CMarkerHistogram marker(nodeTrainer, DEF_PALETTE_3);

	vec_mat_t train_fv;
	fex::CCommonFeatureExtractor fExtractor(train_img);
	train_fv.push_back(fExtractor.getNDVI(0).autoContrast().get());
	train_fv.push_back(fExtractor.getSaturation().invert().get());

	//	---------- Training ----------
	printf("Training... ");
	int64 ticks = getTickCount();
	
	nodeTrainer->addFeatureVec(train_fv, train_gt);
	nodeTrainer->train();
//	dynamic_cast<CTrainNodeNaiveBayes *>(nodeTrainer)->smooth(10);

	ticks = getTickCount() - ticks;
	printf("Done! (%fms)\n", ticks * 1000 / getTickFrequency());


	//	---------- Visualization ----------
	printf("Preparing Histogram...");
	ticks = getTickCount();
	
	Mat hist1D = marker.drawHistogram();
	Mat hist2D = marker.drawHistogram2D();
	Mat clMap = marker.drawClassificationMap2D(Z);
	
	ticks = getTickCount() - ticks;
	printf("Done! (%fms)\n", ticks * 1000 / getTickFrequency());

	imshow("histogram 1d", hist1D);
	imshow("histogram 2d", hist2D);
	imshow("class map 2d", clMap);
//	imwrite("D:\\output.jpg", hist2D);

	cvWaitKey();



	return 0;
}