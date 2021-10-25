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

/**
 * Applies the Sigmoid Activation function
 *
 * @param the value at each node
 * @return a number between 0 and 1.
 */
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
	const byte		nStates					= 10;				// 10 digits (number of output nodes)
	const word		nFeatures				= 28 * 28;			// every pixel of 28x28 digit image is a distinct feature (number of input nodes)
    const size_t    numNeuronsHiddenLayer1	= 16;
	const size_t    numNeuronsHiddenLayer2	= 16;
    const size_t	numTrainSamples  		= 4000;
	const size_t 	numTestSamples    		= 2000;
	const size_t	numEpochs				= 20;

#ifdef WIN32
	const std::string dataPath = "../../data/digits/";
#else
	const std::string dataPath = "../../../data/digits/";
#endif

	auto pLayerInput  = std::make_shared<dgm::dnn::CNeuronLayer>(nFeatures, 0, [](float x) { return x; }, [](float x) { return 1.0f; });
    auto pLayerHidden1 = std::make_shared<dgm::dnn::CNeuronLayer>(numNeuronsHiddenLayer1, nFeatures, &sigmoidFunction, &sigmoidFunction_derivative);
	auto pLayerHidden2 = std::make_shared<dgm::dnn::CNeuronLayer>(numNeuronsHiddenLayer2, numNeuronsHiddenLayer1, &sigmoidFunction, &sigmoidFunction_derivative);
    auto pLayerOutput = std::make_shared<dgm::dnn::CNeuronLayer>(nStates, numNeuronsHiddenLayer2, &sigmoidFunction, &sigmoidFunction_derivative);
	pLayerHidden1->generateRandomWeights();
	pLayerHidden2->generateRandomWeights();
	pLayerOutput->generateRandomWeights();

	dgm::dnn::CPerceptron perceptron({ pLayerInput, pLayerHidden1, pLayerHidden2, pLayerOutput });

	Mat fv;

	// ==================== TRAINING DIGITS ====================
	dgm::Timer::start("Training...");
	auto	trainGT = readGroundTruth(dataPath + "train_gt.txt");
	for (size_t e = 0; e < numEpochs; e++)
		for(int s = 0; s < numTrainSamples; s++) {
			std::stringstream ss;
			ss << dataPath << "train/digit_" << std::setfill('0') << std::setw(4) << s << ".png";
			std::string fileName = samples::findFile(ss.str());
			Mat img = imread(fileName, 0);
			img = img.reshape(1, img.cols * img.rows);
			img.convertTo(fv, CV_32FC1, 1.0 / 255);
			fv = Scalar(1.0f) - fv;

			Mat outputValues = perceptron.getPrediction(fv);

			Mat outputGroundtruth(nStates, 1, CV_32FC1, Scalar(0));
			outputGroundtruth.at<float>(trainGT[s], 0) = 1.0f;

			perceptron.backPropagate(outputGroundtruth, 0.05f);
		} // samples
	dgm::Timer::stop();

	// ==================== TESTING DIGITS ====================
	dgm::CCMat confMat(nStates);
	dgm::Timer::start("Testing...");
	auto 	testGT = readGroundTruth(dataPath + "test_gt.txt");
	for(size_t s = 0; s < numTestSamples; s++) {
		std::stringstream ss;
		ss << dataPath << "test/digit_" << std::setfill('0') << std::setw(4) << s << ".png";
		std::string fileName = samples::findFile(ss.str());
		Mat img = imread(fileName, 0);
		img = img.reshape(1, img.cols * img.rows);
		img.convertTo(fv, CV_32FC1, 1.0 / 255);
		fv = Scalar(1.0f) - fv;

		Mat outputValues = perceptron.getPrediction(fv);
 
		Point maxclass;
		minMaxLoc(outputValues, NULL, NULL, NULL, &maxclass);
		int number = maxclass.y;
        
		confMat.estimate(number, testGT[s]);
        //printf("prediction [%d] for digit %d with %.3f%s at position %zu \n", number, testDataDigit[z], maxAccuracy, "%", z);
	} // samples
	dgm::Timer::stop();
	printf("Accuracy = %.2f%%\n", confMat.getAccuracy());
	
	// Confusion matrix
	dgm::vis::CMarker marker;
	Mat cMat    = confMat.getConfusionMatrix();
	Mat cMatImg = marker.drawConfusionMatrix(cMat, dgm::vis::MARK_BW);
	imshow("Confusion Matrix", cMatImg);
	
	waitKey();

	return 0;
}
