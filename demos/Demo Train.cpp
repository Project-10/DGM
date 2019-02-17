// Example "Training" 2D-case with model training
#include "DGM.h"
#include "VIS.h"
#include "DGM/timer.h"

using namespace DirectGraphicalModels;
using namespace DirectGraphicalModels::vis;

void print_help(char *argv0)
{
	printf("Usage: %s node_training_model edge_training_model training_image_features training_groundtruth_image testing_image_features testing_groundtruth_image original_image output_image\n", argv0);

	printf("\nNode training models:\n");
	printf("0: Bayes\n");
	printf("1: Gaussian Mixture Model\n");
	printf("2: OpenCV Gaussian Mixture Model\n");
	printf("3: Nearest Neighbor\n");
	printf("4: OpenCV Nearest Neighbor\n");
	printf("5: OpenCV Random Forest\n");
	printf("6: MicroSoft Random Forest\n");
	printf("7: OpenCV Artificial Neural Network\n");
	printf("8: OpenCV Support Vector Machines\n");

	
	printf("\nEdge training models:\n");
	printf("0: Without Edges\n");
	printf("1: Potts Model\n");
	printf("2: Contrast-Sensitive Potts Model\n");
	printf("3: Contrast-Sensitive Potts Model with Prior\n");
	printf("4: Concatenated Model\n");
}

int main(int argc, char *argv[])
{
	const Size	imgSize		= Size(400, 400);
	const int	width		= imgSize.width;
	const int	height		= imgSize.height;
	const byte	nStates		= 6;				// {road, traffic island, grass, agriculture, tree, car} 	
	const word	nFeatures	= 3;		

	if (argc != 9) {
		print_help(argv[0]);
		return 0;
	}

	// Reading parameters and images
	int nodeModel	= atoi(argv[1]);																	// node training model
	int edgeModel	= atoi(argv[2]);																	// edge training model
	Mat train_fv	= imread(argv[3], 1); resize(train_fv, train_fv, imgSize, 0, 0, INTER_LANCZOS4);	// training image feature vector
	Mat train_gt	= imread(argv[4], 0); resize(train_gt, train_gt, imgSize, 0, 0, INTER_NEAREST);		// groundtruth for training
	Mat test_fv		= imread(argv[5], 1); resize(test_fv,  test_fv,  imgSize, 0, 0, INTER_LANCZOS4);	// testing image feature vector
	Mat test_gt		= imread(argv[6], 0); resize(test_gt,  test_gt,  imgSize, 0, 0, INTER_NEAREST);		// groundtruth for evaluation
	Mat test_img	= imread(argv[7], 1); resize(test_img, test_img, imgSize, 0, 0, INTER_LANCZOS4);	// testing image

	// Preparing parameters for edge trainers
	vec_float_t			vParams = {100, 0.01f};	
	if (edgeModel <= 1 || edgeModel == 4) vParams.pop_back();	// Potts and Concat models need ony 1 parameter
	if (edgeModel == 0) vParams[0] = 1;							// Emulate "No edges"
	else edgeModel--;
	
	auto				nodeTrainer = CTrainNode::create(nodeModel, nStates, nFeatures);
	auto				edgeTrainer = CTrainEdge::create(edgeModel, nStates, nFeatures);
	CGraphPairwise		graph(nStates);
	CGraphPairwiseExt	graphExt(graph);
	CInferLBP			decoder(graph);
	CMarker				marker(DEF_PALETTE_6);
	CCMat				confMat(nStates);

	// ==================== STAGE 1: Building the graph ====================
	Timer::start("Building the Graph... ");
	graphExt.buildGraph(imgSize);
	Timer::stop();

	// ========================= STAGE 2: Training =========================
	Timer::start("Training... ");
	// Node Training (compact notation)
	nodeTrainer->addFeatureVecs(train_fv, train_gt);					

	// Edge Training (comprehensive notation)
	Mat featureVector1(nFeatures, 1, CV_8UC1); 
	Mat featureVector2(nFeatures, 1, CV_8UC1); 	
	for (int y = 1; y < height; y++) {
		byte *pFv1 = train_fv.ptr<byte>(y);
		byte *pFv2 = train_fv.ptr<byte>(y - 1);
		byte *pGt1 = train_gt.ptr<byte>(y);
		byte *pGt2 = train_gt.ptr<byte>(y - 1);
		for (int x = 1; x < width; x++) {
			for (word f = 0; f < nFeatures; f++) featureVector1.at<byte>(f, 0) = pFv1[nFeatures * x + f];		// featureVector1 = fv[x][y]

			for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv1[nFeatures * (x - 1) + f];	// featureVector2 = fv[x-1][y]
			edgeTrainer->addFeatureVecs(featureVector1, pGt1[x], featureVector2, pGt1[x-1]);
			edgeTrainer->addFeatureVecs(featureVector2, pGt1[x-1], featureVector1, pGt1[x]);

			for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv2[nFeatures * x + f];		// featureVector2 = fv[x][y-1]
			edgeTrainer->addFeatureVecs(featureVector1, pGt1[x], featureVector2, pGt2[x]);
			edgeTrainer->addFeatureVecs(featureVector2, pGt2[x], featureVector1, pGt1[x]);
		} // x
	} // y

	nodeTrainer->train(); 
	edgeTrainer->train(); 
	Timer::stop();

	// ==================== STAGE 3: Filling the Graph =====================
	Timer::start("Filling the Graph... ");
	Mat nodePotentials = nodeTrainer->getNodePotentials(test_fv);		// Classification: CV_32FC(nStates) <- CV_8UC(nFeatures)
	graphExt.setGraph(nodePotentials);									// Filling-in the graph nodes
	graphExt.fillEdges(*edgeTrainer, test_fv, vParams);					// Filling-in the graph edges with pairwise potentials
	Timer::stop();

	// ========================= STAGE 4: Decoding =========================
	Timer::start("Decoding... ");
	vec_byte_t optimalDecoding = decoder.decode(100);
	Timer::stop();

	// ====================== Evaluation =======================	
	Mat solution(imgSize, CV_8UC1, optimalDecoding.data());
	confMat.estimate(test_gt, solution);								// compare solution with the groundtruth
	char str[255];
	sprintf(str, "Accuracy = %.2f%%", confMat.getAccuracy());
	printf("%s\n", str);

	// ====================== Visualization =======================
	marker.markClasses(test_img, solution);
	rectangle(test_img, Point(width - 160, height- 18), Point(width, height), CV_RGB(0,0,0), -1);
	putText(test_img, str, Point(width - 155, height - 5), cv::HersheyFonts::FONT_HERSHEY_SIMPLEX, 0.45, CV_RGB(225, 240, 255), 1, cv::LineTypes::LINE_AA);
	imwrite(argv[8], test_img);
	
	imshow("Image", test_img);
	waitKey(1000);

	return 0;
}

