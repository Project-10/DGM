// Example "Training" 2D-case with model training
#include "DGM.h"
#include "VIS.h"

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
	printf("4: OpenCV Random Forest\n");
	printf("5: MicroSoft Random Forest\n");

	
	printf("\nEdge training models:\n");
	printf("0: Without Edges\n");
	printf("1: Potts Model\n");
	printf("2: Contrast-Sensitive Potts Model\n");
	printf("3: Contrast-Sensitive Potts Model with Prior\n");
	printf("4: Concatenated Model\n");
}

int main(int argc, char *argv[])
{
	const CvSize		imgSize		= cvSize(400, 400);
	const int			width		= imgSize.width;
	const int			height		= imgSize.height;
	const unsigned int	nStates		= 6;		// {road, traffic island, grass, agriculture, tree, car} 	
	const unsigned int	nFeatures	= 3;		

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

	CTrainNode		* nodeTrainer	= NULL; 
	CTrainEdge		* edgeTrainer	= NULL;
	CGraphExt		* graph			= new CGraphExt(nStates);
	CInfer			* decoder		= new CInferLBP(graph);
	CMarker			* marker		= new CMarker(DEF_PALETTE_6);
	CCMat			* confMat		= new CCMat(nStates);
	float			  params[]		= {100, 0.01f};						
	size_t			  params_len;

	switch(nodeModel) {
		case 0: nodeTrainer = new CTrainNodeNaiveBayes(nStates, nFeatures);	break;
		case 1: nodeTrainer = new CTrainNodeGMM(nStates, nFeatures);		break;		
		case 2: nodeTrainer = new CTrainNodeCvGMM(nStates, nFeatures);		break;		
		case 3: nodeTrainer = new CTrainNodeKNN(nStates, nFeatures);		break;
		case 4: nodeTrainer = new CTrainNodeCvRF(nStates, nFeatures);		break;		
#ifdef USE_SHERWOOD
		case 5: nodeTrainer = new CTrainNodeMsRF(nStates, nFeatures);		break;
#endif
		default: printf("Unknown node_training_model is given\n"); print_help(argv[0]); return 0;
	}
	switch(edgeModel) {
		case 0: params[0] = 1;	// Emulate "No edges"
		case 1:	edgeTrainer = new CTrainEdgePotts(nStates, nFeatures);		params_len = 1; break;
		case 2:	edgeTrainer = new CTrainEdgePottsCS(nStates, nFeatures);	params_len = 2; break;
		case 3:	edgeTrainer = new CTrainEdgePrior(nStates, nFeatures);		params_len = 2; break;
		case 4:	
			edgeTrainer = new CTrainEdgeConcat<CTrainNodeNaiveBayes, CDiffFeaturesConcatenator>(nStates, nFeatures);
			params_len = 1;
			break;
		default: printf("Unknown edge_training_model is given\n"); print_help(argv[0]); return 0;
	}

	// ==================== STAGE 1: Building the graph ====================
	printf("Building the Graph... ");
	int64 ticks = getTickCount();	
	graph->build(imgSize);
	ticks = getTickCount() - ticks;
	printf("Done! (%fms)\n", ticks * 1000 / getTickFrequency());

	// ========================= STAGE 2: Training =========================
	printf("Training... ");
	ticks = getTickCount();	
	
	// Node Training (compact notation)
	nodeTrainer->addFeatureVec(train_fv, train_gt);					

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

	ticks = getTickCount() - ticks;
	printf("Done! (%fms)\n", ticks * 1000 / getTickFrequency());

	// ==================== STAGE 3: Filling the Graph =====================
	printf("Filling the Graph... ");
	ticks = getTickCount();
	Mat nodePotentials = nodeTrainer->getNodePotentials(test_fv);		// Classification: CV_32FC(nStates) <- CV_8UC(nFeatures)
	graph->setNodes(nodePotentials);									// Filling-in the graph nodes
	graph->fillEdges(edgeTrainer, test_fv, params, params_len);			// Filling-in the graph edges with pairwise potentials
	ticks = getTickCount() - ticks;
	printf("Done! (%fms)\n", ticks * 1000 / getTickFrequency());

	// ========================= STAGE 4: Decoding =========================
	printf("Decoding... ");
	ticks = getTickCount();
	vec_byte_t optimalDecoding = decoder->decode(100);
	ticks =  getTickCount() - ticks;
	printf("Done! (%fms)\n", ticks * 1000 / getTickFrequency());

	// ====================== Evaluation =======================	
	Mat solution(imgSize, CV_8UC1, optimalDecoding.data());
	confMat->estimate(test_gt, solution);								// compare solution with the groundtruth
	char str[255];
	sprintf(str, "Accuracy = %.2f%%", confMat->getAccuracy());
	printf("%s\n", str);

	// ====================== Visualization =======================
	marker->markClasses(test_img, solution);
	rectangle(test_img, Point(width - 160, height- 18), Point(width, height), CV_RGB(0,0,0), -1);
	putText(test_img, str, Point(width - 155, height - 5), FONT_HERSHEY_SIMPLEX, 0.45, CV_RGB(225, 240, 255), 1, CV_AA);
	imwrite(argv[8], test_img);
	
	imshow("Image", test_img);
	cvWaitKey(0 * 1000);

	return 0;
}

