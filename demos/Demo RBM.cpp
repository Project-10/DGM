#include "DNN.h"
#include "DGM.h"
#include "VIS.h"
#include "DGM/timer.h"
#include <fstream>

namespace dgm = DirectGraphicalModels;

/**
 * Reads the digits numerical value in a decimal notation
 *
 * @param file to read, and the number of digits to read
 * @return an array of digit labels
 */
std::vector<byte> readGroundTruth(const std::string& fileName)
{
	std::vector<byte> res;
	std::ifstream inFile;
	inFile.open(fileName.c_str());

	if (inFile.is_open()) {
		int val;
		while (!inFile.eof()) {
			inFile >> val;
			res.push_back(static_cast<byte>(val));
		}
		inFile.close();
	}
	return res;
}


float sigmoidFunction(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

float sigmoidFunction_derivative(float x)
{
	float s = sigmoidFunction(x);
	return s * (1 - s);
}

int main()
{
	const float learningRate             = 0.05f;
	const size_t numEpochs				 = 20;
	const size_t numTestSamples			 = 2000;
	const size_t numTrainSamples		 = 4000;
	
	//const byte	nStates					 = 10;
	const word nFeatures                 = 28 * 28;
	const size_t numNeuronsHiddenLayer   = 10;

#ifdef WIN32
	const std::string dataPath = "../../data/digits/";
#else
	const std::string dataPath = "../../../data/digits/";
#endif

	auto pLayerVisible = std::make_shared<dgm::dnn::CNeuronLayer>(nFeatures, 1, [](float x) { return x; }, [](float x) { return 1.0f; });
	auto pLayerHidden  = std::make_shared<dgm::dnn::CNeuronLayer>(numNeuronsHiddenLayer, nFeatures, &sigmoidFunction, &sigmoidFunction_derivative);

	pLayerVisible->generateRandomWeights();
	pLayerHidden->generateRandomWeights();
	
	dgm::dnn::CRBM rbm({ pLayerVisible, pLayerHidden });

	//rbm.debug();
	Mat fv;

	// ==================== TRAINING DIGITS ====================
	dgm::Timer::start("Training...\n");
	auto	trainGT = readGroundTruth(dataPath + "train_gt.txt");
	for (size_t e = 0; e < numEpochs; e++)
		for (int s = 0; s < numTrainSamples; s++) {
			std::stringstream ss;
			ss << dataPath << "train/digit_" << std::setfill('0') << std::setw(4) << s << ".png";
			std::string fileName = samples::findFile(ss.str());
			Mat img = imread(fileName, 0);
			img = img.reshape(1, img.cols * img.rows);
			img.convertTo(fv, CV_32FC1, 1.0 / 255);
			fv = Scalar(1.0f) - fv;

			rbm.contrastiveDivergence(fv, 0.5f);
		} // samples
	dgm::Timer::stop();

	// ==================== TESTING DIGITS ====================
	//dgm::CCMat confMat(nStates);
	dgm::Timer::start("Testing...");
	auto 	testGT = readGroundTruth(dataPath + "test_gt.txt");
	for (size_t s = 0; s < numTestSamples; s++) {
		std::stringstream ss;
		ss << dataPath << "test/digit_" << std::setfill('0') << std::setw(4) << s << ".png";
		std::string fileName = samples::findFile(ss.str());
		Mat img = imread(fileName, 0);
		img = img.reshape(1, img.cols * img.rows);
		img.convertTo(fv, CV_32FC1, 1.0 / 255);
		fv = Scalar(1.0f) - fv;

		Mat outputValues = rbm.reconstruct(fv);

		//Point maxclass;
		//minMaxLoc(outputValues, NULL, NULL, NULL, &maxclass);
		//int number = maxclass.y;

		//confMat.estimate(number, testGT[s]);
		//printf("prediction [%d] for digit %d with %.3f%s at position %zu \n", number, testDataDigit[z], maxAccuracy, "%", z);
	} // samples
	dgm::Timer::stop();
	//printf("Accuracy = %.2f%%\n", confMat.getAccuracy());

	// Confusion matrix
	//dgm::vis::CMarker marker;
	//Mat cMat = confMat.getConfusionMatrix();
	//Mat cMatImg = marker.drawConfusionMatrix(cMat, dgm::vis::MARK_BW);
	//imshow("Confusion Matrix", cMatImg);
	rbm.debug();

	waitKey();

	return 0;
}
