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

int main()
{
	const word 		nStates					= 10; 	// 10 digits
	const size_t    numNeuronsInputLayer   	= 784;
    const size_t    numNeuronsHiddenLayer	= 60;
    const size_t    numNeuronsOutputLayer  	= 10;
    const size_t	numTrainSamples  		= 4000;
	const size_t 	numTestSamples    		= 2000;

#ifdef WIN32
	const std::string dataPath = "../../data/digits/";
#else
	const std::string dataPath = "../../../data/digits/";
#endif

	dgm::dnn::CNeuronLayer layerInput(numNeuronsInputLayer, numNeuronsHiddenLayer);
	dgm::dnn::CNeuronLayer layerHidden(numNeuronsHiddenLayer, numNeuronsOutputLayer);
	dgm::dnn::CNeuronLayer layerOutput(numNeuronsOutputLayer, 0);

	layerInput.generateRandomWeights();
	layerHidden.generateRandomWeights();

	Mat fv;

	// ==================== TRAINING DIGITS ====================
	dgm::Timer::start("Training...");
	auto	trainGT = readGroundTruth(dataPath + "train_gt.txt");
	for(int s = 0; s < numTrainSamples; s++) {

		std::stringstream ss;
		ss << dataPath << "train/digit_" << std::setfill('0') << std::setw(4) << s << ".png";
		std::string fileName = samples::findFile(ss.str());
		Mat img = imread(fileName, 0);
		img = img.reshape(1, img.cols * img.rows);
		img.convertTo(fv, CV_32FC1, 1.0 / 255);
		fv = Scalar(1.0f) - fv;

		layerInput.setValues(fv);

		layerHidden.dotProd(layerInput);
		layerOutput.dotProd(layerHidden);

		Mat outputValues = layerOutput.getValues();

        std::vector<float> vResultErrorRate(numNeuronsOutputLayer);
		for(size_t i = 0; i < vResultErrorRate.size(); i++) {
			vResultErrorRate[i] = (trainGT[s] == i) ? 1 : 0;
			vResultErrorRate[i] -= outputValues.at<float>(static_cast<int>(i), 0);
		}

        dgm::dnn::CNeuronLayer::backPropagate(layerInput, layerHidden, layerOutput, vResultErrorRate, 0.1f);
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

		layerInput.setValues(fv);
		layerHidden.dotProd(layerInput);
		layerOutput.dotProd(layerHidden);

		Mat pot = layerOutput.getValues();			// potential vector

		// TODO: use minmaxloc here
		float maxPot = 0;
		byte   number;
		for(int i = 0 ; i < pot.rows; i++) {
			if(pot.at<float>(i, 0) > maxPot) {
				maxPot = pot.at<float>(i, 0);
				number = static_cast<byte>(i);
			}
		}
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


