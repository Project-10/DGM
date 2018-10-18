// Example "Training" 2D-case with model training
#include "DGM.h"
#include "VIS.h"
#include "DGM/timer.h"

using namespace DirectGraphicalModels;
using namespace DirectGraphicalModels::vis;
using namespace DirectGraphicalModels::FactoryTrain;

void print_help(char *argv0,
                const std::vector<std::pair<std::string, randomModelNode>> &vRandomModelsNode,
                const std::vector<std::pair<std::string, randomModelEdge>> &vRandomModelsEdge)
{
	printf("Usage: %s node_training_model edge_training_model training_image_features training_groundtruth_image testing_image_features testing_groundtruth_image original_image output_image\n", argv0);

	printf("\nNode training models:\n");
    for (size_t i = 0; i < vRandomModelsNode.size(); i++)
        printf("%zu: %s\n", i, vRandomModelsNode[i].first.c_str());
    
	printf("\nEdge training models:\n");
    for (size_t i = 0; i < vRandomModelsEdge.size(); i++)
        printf("%zu: %s\n", i, vRandomModelsEdge[i].first.c_str());
}

int main(int argc, char *argv[])
{
	const cv::Size		imgSize		= cv::Size(400, 400);
	const int			width		= imgSize.width;
	const int			height		= imgSize.height;
	const unsigned int	nStates		= 6;		// {road, traffic island, grass, agriculture, tree, car} 	
	const unsigned int	nFeatures	= 3;		
    const std::vector<std::pair<std::string, randomModelNode>> vRandomModelsNode = {
        std::make_pair("Bayes", randomModelNode::Bayes),
        std::make_pair("Gaussian Mixture Model", randomModelNode::GMM),
        std::make_pair("OpenCV Gaussian Mixture Model", randomModelNode::CvGMM),
        std::make_pair("Nearest Neighbor", randomModelNode::KNN),
        std::make_pair("OpenCV Nearest Neighbor", randomModelNode::CvKNN),
        std::make_pair("OpenCV Random Forest", randomModelNode::CvRF),
        std::make_pair("MicroSoft Random Forest", randomModelNode::MsRF),
        std::make_pair("OpenCV Artificial Neural Network", randomModelNode::CvANN),
        std::make_pair("OpenCV Support Vector Machines", randomModelNode::CvSVM)
    };
    const std::vector<std::pair<std::string, randomModelEdge>> vRandomModelsEdge = {
        std::make_pair("Without Edges", randomModelEdge::Potts),
        std::make_pair("Potts Model", randomModelEdge::Potts),
        std::make_pair("Contrast-Sensitive Potts Model", randomModelEdge::PottsCS),
        std::make_pair("Contrast-Sensitive Potts Model with Prior", randomModelEdge::Prior),
        std::make_pair("Concatenated Model", randomModelEdge::Concat),
    };
    
	if (argc != 9) {
		print_help(argv[0], vRandomModelsNode, vRandomModelsEdge);
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

    if (nodeModel < 0 || nodeModel >= vRandomModelsNode.size()) {
        printf("Unknown node training model is given: %d\n", nodeModel);
        print_help(argv[0], vRandomModelsNode, vRandomModelsEdge);
        return 0;
    }
    if (edgeModel < 0 || edgeModel >= vRandomModelsEdge.size()) {
        printf("Unknown edge training model is given: %d\n", edgeModel);
        print_help(argv[0], vRandomModelsNode, vRandomModelsEdge);
        return 0;
    }
    
    general_parameters params1;
    params1["Hello"] = "World";
    auto                  nodeTrainer	= createNodeTrainer(nStates, nFeatures, vRandomModelsNode[nodeModel].second, params1);
	auto			      edgeTrainer	= createEdgeTrainer(nStates, nFeatures, vRandomModelsEdge[edgeModel].second, randomModelNode::Bayes, params1);
    CFactoryGraphPairwise factory(nStates);
	CGraphPairwiseExt	& graphExt = factory.getGraphExt();
	CInfer			    & decoder  = factory.getInfer();
	CMarker				  marker(DEF_PALETTE_6);
	CCMat				  confMat(nStates);
	
    vec_float_t			  vParams = {100, 0.01f};
	switch(edgeModel) {
        case 0: vParams = {1};          break;	// Emulate "No edges"
		case 1:	vParams = {100};        break;
        case 2:	vParams = {100, 0.01f}; break;
        case 3:	vParams = {100, 0.01f}; break;
		case 4:	vParams = {100};        break;
	}

	// ==================== STAGE 1: Building the graph ====================
	Timer::start("Building the Graph... ");
	graphExt.addNodes(imgSize);
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
	graphExt.setNodes(nodePotentials);									// Filling-in the graph nodes
	graphExt.fillEdges(edgeTrainer.get(), test_fv, vParams);			// Filling-in the graph edges with pairwise potentials
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
	cv::waitKey(1000);

	return 0;
}
