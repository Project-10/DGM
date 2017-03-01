// Example "Training" 2D-case with model training
#include "DGM.h"
#include "VIS.h"

using namespace DirectGraphicalModels;
using namespace DirectGraphicalModels::vis;

void print_help(void)
{
	printf("Usage: \"Demo Train.exe\" node_training_model edge_training_model original_image features_image groundtruth_image output_image\n");

	printf("\nNode training models:\n");
	printf("0: Naive Bayes\n");
	printf("1: Gaussian Model\n");
	printf("2: Gaussian Mixture Model\n");
	printf("3: OpenCV Gaussian Model\n");
	printf("4: OpenCV Gaussian Mixture Model\n");
	printf("5: OpenCV Random Forest\n");
	printf("6: MicroSoft Random Forest\n");
	
	printf("\nEdge training models:\n");
	printf("0: Without Edges\n");
	printf("1: Potts Model\n");
	printf("2: Contrast-Sensitive Potts Model\n");
	printf("3: Contrast-Sensitive Potts Model with Prior\n");
	printf("4: Concatenated Model\n");
}

int main(int argv, char *argc[])
{
	const CvSize		imgSize		= cvSize(400, 400);
	const int			width		= imgSize.width;
	const int			height		= imgSize.height;
	const unsigned int	nStates		= 6;		// {road, traffic island, grass, agriculture, tree, car} 		
	const unsigned int	nFeatures	= 3;		

	if (argv != 7) {
		print_help();
		return 0;
	}

	// Reading parameters and images
	int nodeModel	= atoi(argc[1]);															// node training model
	int edgeModel	= atoi(argc[2]);															// edge training model
	Mat img			= imread(argc[3], 1); resize(img, img, imgSize, 0, 0, INTER_LANCZOS4);		// image
	Mat fv			= imread(argc[4], 1); resize(fv,  fv,  imgSize, 0, 0, INTER_LANCZOS4);		// feature vector
	Mat gt			= imread(argc[5], 0); resize(gt,  gt,  imgSize, 0, 0, INTER_NEAREST);		// groundtruth

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
		case 1: nodeTrainer = new CTrainNodeGM(nStates, nFeatures);			break;
		case 2: nodeTrainer = new CTrainNodeGMM(nStates, nFeatures);		break;		
		case 3: nodeTrainer = new CTrainNodeCvGM(nStates, nFeatures);		break;
		case 4: nodeTrainer = new CTrainNodeCvGMM(nStates, nFeatures);		break;		
		case 5: nodeTrainer = new CTrainNodeCvRF(nStates, nFeatures);		break;		
#ifdef USE_SHERWOOD
		case 6: nodeTrainer = new CTrainNodeMsRF(nStates, nFeatures);		break;
#endif
		case 7: nodeTrainer = new CTrainNodeNearestNeighbor(nStates, nFeatures); break;
		default: printf("Unknown node_training_model is given\n"); print_help(); return 0;
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
		default: printf("Unknown edge_training_model is given\n"); print_help(); return 0;
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
	nodeTrainer->addFeatureVec(fv, gt);					

	// Edge Training (comprehensive notation)
	Mat featureVector1(nFeatures, 1, CV_8UC1); 
	Mat featureVector2(nFeatures, 1, CV_8UC1); 	
	for (int y = 1; y < height; y++) {
		byte *pFv1 = fv.ptr<byte>(y);
		byte *pFv2 = fv.ptr<byte>(y - 1);
		byte *pGt1 = gt.ptr<byte>(y);
		byte *pGt2 = gt.ptr<byte>(y - 1);
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
	graph->fillNodes(nodeTrainer, fv);
	graph->fillEdges(edgeTrainer, fv, params, params_len);
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
	confMat->estimate(gt, solution);																				// compare solution with the groundtruth
	char str[255];
	sprintf(str, "Accuracy = %.2f%%", confMat->getAccuracy());
	printf("%s\n", str);

	// ====================== Visualization =======================
	marker->markClasses(img, solution);
	rectangle(img, Point(width - 160, height- 18), Point(width, height), CV_RGB(0,0,0), -1);
	putText(img, str, Point(width - 155, height - 5), FONT_HERSHEY_SIMPLEX, 0.45, CV_RGB(225, 240, 255), 1, CV_AA);
	imwrite(argc[6], img);
	
	imshow("Image", img);
	cvWaitKey(1000);

	//getchar();

	return 0;
}

