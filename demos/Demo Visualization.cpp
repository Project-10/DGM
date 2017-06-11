// Example "Visualization" 2D-case 
#include "DGM.h"
#include "VIS.h"
using namespace DirectGraphicalModels;
using namespace DirectGraphicalModels::vis;

typedef struct {
	CGraph				* pGraph;
	CMarkerHistogram	* pMarker;
	int					  imgWidth;
} USER_DATA;

// Mouse handlers
void solutiontWindowMouseHandler(int Event, int x, int y, int flags, void *param)
{
	USER_DATA	* pUserData	= static_cast<USER_DATA *>(param);
	if (Event == CV_EVENT_LBUTTONDOWN) {
		Mat			  pot, potImg;
		size_t		  node_id	= pUserData->imgWidth * y + x;

		// Node potential
		pUserData->pGraph->getNode(node_id, pot);
		potImg = pUserData->pMarker->drawPotentials(pot, MARK_BW);
		imshow("Node Potential", potImg);

		// Edge potential
		vec_size_t child_nodes;
		pUserData->pGraph->getChildNodes(node_id, child_nodes);
		if (child_nodes.size() > 0) {
			pUserData->pGraph->getEdge(node_id, child_nodes.at(0), pot);
			potImg = pUserData->pMarker->drawPotentials(pot, MARK_BW);
			imshow("Edge Potential", potImg);
		}

		pot.release();
		potImg.release();
	}
}

void print_help(char *argv0)
{
	printf("Usage: %s original_image features_image groundtruth_image\n", argv0);
}

int main(int argc, char *argv[])
{
	const CvSize		imgSize		= cvSize(400, 400);
	const int			width		= imgSize.width;
	const int			height		= imgSize.height;
	const unsigned int	nStates		= 6;		// {road, traffic island, grass, agriculture, tree, car} 		
	const unsigned int	nFeatures	= 3;		

	if (argc != 4) {
		print_help(argv[0]);
		return 0;
	}

	// Reading parameters and images
	Mat img			= imread(argv[1], 1); resize(img, img, imgSize, 0, 0, INTER_LANCZOS4);		// image
	Mat fv			= imread(argv[2], 1); resize(fv,  fv,  imgSize, 0, 0, INTER_LANCZOS4);		// feature vector
	Mat gt			= imread(argv[3], 0); resize(gt,  gt,  imgSize, 0, 0, INTER_NEAREST);		// groundtruth

	CTrainNode			* nodeTrainer	 = new CTrainNodeNaiveBayes(nStates, nFeatures); 
	CTrainEdge			* edgeTrainer	 = new CTrainEdgePottsCS(nStates, nFeatures);
	float				  params[]		 = {400, 0.001f};						
	size_t				  params_len	 = 2;
	CGraph				* graph			 = new CGraph(nStates); 
	CInfer				* decoder		 = new CInferLBP(graph);
	// Define custom colors in RGB format for our classes (for visualization)
	vec_nColor_t		  palette;
	palette.push_back(std::make_pair(CV_RGB(64,  64,   64), "road"));
	palette.push_back(std::make_pair(CV_RGB(0,  255,  255), "tr. island"));
	palette.push_back(std::make_pair(CV_RGB(0,   255,   0), "grass"));
	palette.push_back(std::make_pair(CV_RGB(200, 135,  70), "agricult."));
	palette.push_back(std::make_pair(CV_RGB(64,  128,   0), "tree"));
	palette.push_back(std::make_pair(CV_RGB(255,   0,   0), "car"));
	// Define feature names for visualization
	vec_string_t		  featureNames	= {"NDVI", "Var. Int.", "Saturation"};	
	CMarkerHistogram	* marker		= new CMarkerHistogram(nodeTrainer, palette, featureNames);
	CCMat				* confMat		= new CCMat(nStates);

	// ==================== STAGE 1: Building the graph ====================
	printf("Building the Graph... ");
	int64 ticks = getTickCount();	
	for (int y = 0; y < height; y++) 
		for (int x = 0; x < width; x++) {
			size_t idx = graph->addNode();
			if (x > 0) 	 graph->addArc(idx, idx - 1);
			if (y > 0) 	 graph->addArc(idx, idx - width); 
		} // x
	ticks = getTickCount() - ticks;
	printf("Done! (%fms)\n", ticks * 1000 / getTickFrequency());

	// ========================= STAGE 2: Training =========================
	printf("Training... ");
	ticks = getTickCount();	
	nodeTrainer->addFeatureVec(fv, gt);										// Only Node Training 		
	nodeTrainer->train();													// Contrast-Sensitive Edge Model requires no training
	ticks = getTickCount() - ticks;
	printf("Done! (%fms)\n", ticks * 1000 / getTickFrequency());

	// ==================== STAGE 3: Filling the Graph =====================
	printf("Filling the Graph... ");
	ticks = getTickCount();
	Mat featureVector1(nFeatures, 1, CV_8UC1); 
	Mat featureVector2(nFeatures, 1, CV_8UC1); 
	Mat nodePot, edgePot;
	for (int y = 0, idx = 0; y < height; y++) {
		byte *pFv1 = fv.ptr<byte>(y);
		byte *pFv2 = (y > 0) ? fv.ptr<byte>(y - 1) : NULL;	
		for (int x = 0; x < width; x++, idx++) {
			for (word f = 0; f < nFeatures; f++) featureVector1.at<byte>(f, 0) = pFv1[nFeatures * x + f];			// featureVector1 = fv[x][y]
			nodePot = nodeTrainer->getNodePotentials(featureVector1);												// node potential
			graph->setNode(idx, nodePot);

			if (x > 0) {
				for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv1[nFeatures * (x - 1) + f];	// featureVector2 = fv[x-1][y]
				edgePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len);		// edge potential
				graph->setArc(idx, idx - 1, edgePot);
			} // if x
			if (y > 0) {
				for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv2[nFeatures * x + f];		// featureVector2 = fv[x][y-1]
				edgePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len);		// edge potential
				graph->setArc(idx, idx - width, edgePot);
			} // if y
		} // x
	} // y
	ticks = getTickCount() - ticks;
	printf("Done! (%fms)\n", ticks * 1000 / getTickFrequency());

	// ========================= STAGE 4: Decoding =========================
	printf("Decoding... ");
	ticks = getTickCount();
	vec_byte_t optimalDecoding = decoder->decode(10);
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
	imshow("Solution", img);
	
	// Feature distribution histograms
	marker->showHistogram();

	// Confusion matrix
	Mat cMat	= confMat->getConfusionMatrix();
	Mat cMatImg	= marker->drawConfusionMatrix(cMat, MARK_BW);
	imshow("Confusion Matrix", cMatImg);

	// Setting up handlers
	USER_DATA userData;
	userData.pGraph		= graph;
	userData.pMarker	= marker;
	userData.imgWidth	= width;
	cvSetMouseCallback("Solution",  solutiontWindowMouseHandler, &userData);

	cvWaitKey();

	return 0;
}

