// Example "Dense-CRF" 2D-case with model training
#include "DGM.h"
#include "VIS.h"
#include "DGM/timer.h"

using namespace DirectGraphicalModels;
using namespace DirectGraphicalModels::vis;

void print_help(char *argv0)
{
	printf("Usage: %s training_image_features training_groundtruth_image testing_image_features testing_groundtruth_image original_image output_image\n", argv0);
}

int main(int argc, char *argv[])
{
	const Size	imgSize		= Size(400, 400);
	const int	width		= imgSize.width;
	const int	height		= imgSize.height;
	const byte	nStates		= 6;				// {road, traffic island, grass, agriculture, tree, car} 	
	const word	nFeatures	= 3;
	const word	nParams		= 4;

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

	auto	nodeTrainer = CTrainNode::create(Bayes, nStates, nFeatures);
	auto	graphKit	= CGraphKit::create(GraphType::dense, nStates);
	CMarker	marker(DEF_PALETTE_6);
	CCMat	confMat(nStates);


	const float* pParam;
	const vec_float_t vInitParam  = { 10.0f, 1.0f, 10.0f, 1.0f };
	const vec_float_t vInitDeltas = { 2.0f, 0.2f, 2.0f, 0.2f };
	
	CPowell powell(nParams);
	powell.setInitParams(vInitParam);
	powell.setDeltas(vInitDeltas);

	pParam = vInitParam.data();

	// ========================= STAGE 2: Training =========================
	Timer::start("Training... ");
	nodeTrainer->addFeatureVecs(train_fv, train_gt);
	nodeTrainer->train();
	Timer::stop();

	while (!powell.isConverged()) {
		// ==================== STAGE 3: Filling the Graph =====================
		Timer::start("Filling the Graph... ");
		Mat nodePotentials = nodeTrainer->getNodePotentials(test_fv);		// Classification: CV_32FC(nStates) <- CV_8UC(nFeatures)
		graphKit->getGraphExt().setGraph(nodePotentials);							// Filling-in the graph nodes
		graphKit->getGraphExt().addDefaultEdgesModel(pParam[0], pParam[1]);
		graphKit->getGraphExt().addDefaultEdgesModel(test_fv, pParam[2], pParam[3]);
		Timer::stop();

		// ========================= STAGE 4: Decoding =========================
		Timer::start("Decoding... ");
		vec_byte_t optimalDecoding = graphKit->getInfer().decode(100);
		Timer::stop();

		// ====================== Evaluation =======================
		Mat solution(imgSize, CV_8UC1, optimalDecoding.data());
		confMat.estimate(test_gt, solution);
		pParam = powell.getParams(confMat.getAccuracy());
		graphKit->getGraph().reset();
		char str[255];
		sprintf(str, "Accuracy = %.2f%%", confMat.getAccuracy());
		printf("%s\n", str);

		confMat.reset();

		// ====================== Visualization =======================
		Mat test_img_temp = test_img.clone();
		marker.markClasses(test_img_temp, solution);
		rectangle(test_img_temp, Point(width - 160, height - 18), Point(width, height), CV_RGB(0, 0, 0), -1);
		putText(test_img_temp, str, Point(width - 155, height - 5), FONT_HERSHEY_SIMPLEX, 0.45, CV_RGB(225, 240, 255), 1, LineTypes::LINE_AA);
		//imwrite(argv[6], test_img_temp);
	
		imshow("Image", test_img_temp);
	
		waitKey(100);
	}

	return 0;
}
