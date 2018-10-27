// Example "Dense-CRF" 2D-case with model training
#include "DGM.h"
#include "VIS.h"
#include "DGM/timer.h"
#include "DGM/serialize.h"

using namespace DirectGraphicalModels;
using namespace DirectGraphicalModels::vis;

void print_help(char *argv0)
{
	printf("Usage: %s training_image_features training_groundtruth_image testing_image_features testing_groundtruth_image original_image output_image\n", argv0);
}

int main(int argc, char *argv[])
{
	const cv::Size		imgSize = cv::Size(400, 400);
	const int			width = imgSize.width;
	const int			height = imgSize.height;
	const unsigned int	nStates = 6;		// {road, traffic island, grass, agriculture, tree, car} 	
	const unsigned int	nFeatures = 3;

	if (argc != 7) {
		print_help(argv[0]);
		return 0;
	}

	// Reading parameters and images
    Mat train_fv = imread(argv[1], 1); resize(train_fv, train_fv, imgSize, 0, 0, INTER_LANCZOS4);	// training image feature vector
	Mat train_gt = imread(argv[2], 0); resize(train_gt, train_gt, imgSize, 0, 0, INTER_NEAREST);	// groundtruth for training
	Mat test_fv  = imread(argv[3], 1); resize(test_fv,  test_fv,  imgSize, 0, 0, INTER_LANCZOS4);	// testing image feature vector
	Mat test_gt  = imread(argv[4], 0); resize(test_gt,  test_gt,  imgSize, 0, 0, INTER_NEAREST);	// groundtruth for evaluation
	Mat test_img = imread(argv[5], 1); resize(test_img, test_img, imgSize, 0, 0, INTER_LANCZOS4);	// testing image

	CTrainNodeBayes nodeTrainer(nStates, nFeatures);
	CTrainEdgePotts	edgeTrainer(nStates, nFeatures);
//	CGraphExt		* graph = new CGraphExt(nStates);
//	CInfer			* decoder = new CInferLBP(graph);
	CMarker			marker(DEF_PALETTE_6);
	CCMat			confMat(nStates);
//	float			params[] = { 100, 0.01f };
//	size_t			params_len = 1;


	// ==================== STAGE 1: Building the graph ====================
//	Timer::start("Building the Graph... ");
//	graph->build(imgSize);
//	Timer::stop();

	// ========================= STAGE 2: Training =========================
	Timer::start("Training... ");
	// Node Training (compact notation)
	nodeTrainer.addFeatureVecs(train_fv, train_gt);

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
			edgeTrainer.addFeatureVecs(featureVector1, pGt1[x], featureVector2, pGt1[x - 1]);
			edgeTrainer.addFeatureVecs(featureVector2, pGt1[x - 1], featureVector1, pGt1[x]);

			for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv2[nFeatures * x + f];		// featureVector2 = fv[x][y-1]
			edgeTrainer.addFeatureVecs(featureVector1, pGt1[x], featureVector2, pGt2[x]);
			edgeTrainer.addFeatureVecs(featureVector2, pGt2[x], featureVector1, pGt1[x]);
		} // x
	} // y

	nodeTrainer.train();
	edgeTrainer.train();
	Timer::stop();


	// CTrainEdgePotts::getEdgePotentials(100, nStates); // default Potts edge potential

	// ==================== STAGE 3: Filling the Graph =====================
	Timer::start("Filling the Graph... ");
	Mat nodePotentials = nodeTrainer.getNodePotentials(test_fv);		// Classification: CV_32FC(nStates) <- CV_8UC(nFeatures)
	//graph->setNodes(nodePotentials);									// Filling-in the graph nodes
	//graph->fillEdges(edgeTrainer, test_fv, params, params_len);			// Filling-in the graph edges with pairwise potentials

	CGraphDense graph(nStates);
	CGraphDenseExt graphExt(graph);
	CInferDense decoder(graph);
	
	
    // TODO:
	graphExt.setNodes(nodePotentials);
    graphExt.addGaussianEdgeModel(Vec2f::all(100), 3);
    graphExt.addBilateralEdgeModel(test_img, Vec2f::all(10), Vec3f::all(32), 10);
	Timer::stop();


	// ========================= STAGE 4: Decoding =========================
	Timer::start("Decoding... ");
	vec_byte_t optimalDecoding = decoder.decode(100);
	Timer::stop();


	// ====================== Evaluation =======================
	Mat solution(imgSize, CV_8UC1, optimalDecoding.data());
	confMat.estimate(test_gt, solution);
	char str[255];
	sprintf(str, "Accuracy = %.2f%%", confMat.getAccuracy());
	printf("%s\n", str);

	// ====================== Visualization =======================
	marker.markClasses(test_img, solution);
	rectangle(test_img, Point(width - 160, height - 18), Point(width, height), CV_RGB(0, 0, 0), -1);
	putText(test_img, str, Point(width - 155, height - 5), FONT_HERSHEY_SIMPLEX, 0.45, CV_RGB(225, 240, 255), 1, cv::LineTypes::LINE_AA);
	imwrite(argv[6], test_img);
	
	imshow("Image", test_img);
	cv::waitKey();

	return 0;
}
