// Example "Dense-CRF" 2D-case with model training
#include "DGM.h"
#include "VIS.h"

using namespace DirectGraphicalModels;
using namespace DirectGraphicalModels::vis;

void print_help(char *argv0)
{
	printf("Usage: %s training_image_features training_groundtruth_image testing_image_features testing_groundtruth_image original_image\n", argv0);
}

int main(int argc, char *argv[])
{
	const Size	imgSize		= Size(400, 400);
	const byte	nStates		= 6;				// {road, traffic island, grass, agriculture, tree, car} 	
	const word	nFeatures	= 3;

	if (argc != 6) {
		print_help(argv[0]);
		return 0;
	}

	// Reading parameters and images
    Mat train_fv = imread(argv[1], 1); resize(train_fv, train_fv, imgSize, 0, 0, INTER_LANCZOS4);	// training image feature vector
	Mat train_gt = imread(argv[2], 0); resize(train_gt, train_gt, imgSize, 0, 0, INTER_NEAREST);	// groundtruth for training
	Mat test_fv  = imread(argv[3], 1); resize(test_fv,  test_fv,  imgSize, 0, 0, INTER_LANCZOS4);	// testing image feature vector
	Mat test_gt  = imread(argv[4], 0); resize(test_gt,  test_gt,  imgSize, 0, 0, INTER_NEAREST);	// groundtruth for evaluation
	Mat test_img = imread(argv[5], 1); resize(test_img, test_img, imgSize, 0, 0, INTER_LANCZOS4);	// testing image

	auto	nodeTrainer = CTrainNode::create(Bayes, nStates, nFeatures);
	auto	graphKit	= CGraphKit::create(GraphType::dense, nStates);
	CMarker	marker(DEF_PALETTE_6);
	CCMat	confMat(nStates);

	// Initializing Powell search class and parameters
	const vec_float_t vInitParams  = { 100.0f, 300.0f, 3.0f, 10.0f };
	const vec_float_t vInitDeltas  = {  10.0f,  10.0f, 1.0f,  1.0f };
	vec_float_t vParams = vInitParams;									// Actual model parameters

	CPowell powell(vParams.size());
	powell.setInitParams(vInitParams);
	powell.setDeltas(vInitDeltas);

	// ========================= Training Node Potentials=========================
	nodeTrainer->addFeatureVecs(train_fv, train_gt);
	nodeTrainer->train();

	// Main loop of parameters optimization
	for (int i = 1; ; i++) {
		// ================= Filling the Graph =====================
		Mat nodePotentials = nodeTrainer->getNodePotentials(test_fv);				// Classification: CV_32FC(nStates) <- CV_8UC(nFeatures)
		graphKit->getGraphExt().setGraph(nodePotentials);							// Filling-in the graph nodes
		graphKit->getGraphExt().addDefaultEdgesModel(vParams[0], vParams[2]);
		graphKit->getGraphExt().addDefaultEdgesModel(test_fv, vParams[1], vParams[3]);

		// ====================== Decoding =========================
		vec_byte_t optimalDecoding = graphKit->getInfer().decode(100);

		// ====================== Evaluation =======================
		Mat solution(imgSize, CV_8UC1, optimalDecoding.data());
		confMat.estimate(test_gt, solution);
		
		printf("Iteration: %d, parameters: { ", i);
		for (const float& param : vParams) printf("%.1f ", param);
		printf("}, accuracy: %.2f%%\n", confMat.getAccuracy());

		if (powell.isConverged()) break;
		vParams = powell.getParams(confMat.getAccuracy());
		graphKit->getGraph().reset();
		confMat.reset();
	}

	vParams = powell.getParams(1);
	printf("Resulting parameters: {");
	for (const float& param : vParams) printf("%.1f ", param);
	printf("}\n");

	return 0;
}
