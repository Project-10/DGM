#include "DNN.h"
#include "DGM/timer.h"

namespace dgm = DirectGraphicalModels;

void backPropagate(std::vector<dgm::dnn::ptr_neuron_t>& vpLayerA,
                   std::vector<dgm::dnn::ptr_neuron_t>& vpLayerB,
                   std::vector<dgm::dnn::ptr_neuron_t>& vpLayerC,
                   double resultErrorRate[]);

void dotProd(std::vector<dgm::dnn::ptr_neuron_t>& vpLayerA, std::vector<dgm::dnn::ptr_neuron_t>& vpLayerB);

/**
 * Applies the Sigmoid Activation function
 *
 * @param the value at each node
 * @return a number between 0 and 1.
 */
float applySigmoidFunction(float val)
{
	float sigmoid = 1 / (1 + exp(-val));
	return sigmoid;
}

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
 * Reads the image pixel value
 *
 * @param image to read, and the number of images to read
 * @return an array of pixel values for each image
 */
int **readImgData(std::string file, int dataSize)
{
	const int inputLayer = 784;
	static int **trainDataBin = new int*[dataSize];
	
	for(int m = 0; m < dataSize; m++)
	{
		trainDataBin[m] = new int[inputLayer];
		std::stringstream ss;
		ss << std::setfill('0') << std::setw(4);
		ss << m;
		std::string number = ss.str();
		std::string path = file + number + ".png";
		std::string image_path = samples::findFile(path);

		Mat img = imread(image_path, 0);
		
		int l=0;
		for(int i = 0; i < img.rows; i++) {
			for(int j = 0; j < img.cols; j++) {
				int value = abs((int)img.at<uchar>(i,j) - 255);
				trainDataBin[m][l] = value;
				l++;
			}
		}
	}
	return trainDataBin;
}

/**
 * Gives the correct firing output nodes for each digit so we can find the errorRate*
 * Example when comparing the output nodes in the end, the correct prediction for
 * number 2 lets say is : {0,0,1,0,0,0,0,0,0,0,0}
 *
 * @param the output digits value (10)
 * @return an array of numbers filled with 0's for input 'x' excepted with a single 1 in the index[x].
 */
int **resultPredictions(int outputLayer)
{
	int **result = new int*[outputLayer];
	
	for(int i = 0; i < outputLayer; i++) {
		result[i] = new int[outputLayer];
		for(int j = 0; j < outputLayer; j++) {
			result[i][j] = (i == j) ?  1 :  0;
		}
	}
	return result;
}

/**
 * Reads the digits pixel value in a decimal notation
 *
 * @param file to read, and the number of digits to read
 * @return an array of digits
 */
//int **readBinData(std::string file, int dataSize) {
//    const int inputLayer = 784;
//    int **trainDataBin = new int*[dataSize];
//
//    std::string fileBinData = file;
//    std::ifstream inFile;
//    inFile.open(fileBinData.c_str());
//
//    if (inFile.is_open()) {
//        for(int i = 0 ; i < dataSize; i++) {
//            trainDataBin[i] = new int[inputLayer];
//
//            for (int j = 0; j < inputLayer; j++) {
//                inFile >> trainDataBin[i][j];
//            }
//        }
//        inFile.close();
//    }
//    return trainDataBin;
//}

int main() {
    const size_t     numNeuronsInputLayer   = 784;
    const size_t     numNeuronsHiddenLayer  = 60;
    const size_t     numNeuronsOutputLayer  = 10;
    const size_t                  dataSize  = 4000;

    std::vector<dgm::dnn::ptr_neuron_t> vpInputLayer;
    std::vector<dgm::dnn::ptr_neuron_t> vpHiddenLayer;
    std::vector<dgm::dnn::ptr_neuron_t> vpOutputLayer;

    for (size_t i = 0; i < numNeuronsInputLayer; i++)
        vpInputLayer.push_back( std::make_shared<dgm::dnn::CNeuron>(numNeuronsHiddenLayer, 0) );

    for (size_t i = 0; i < numNeuronsHiddenLayer; i++)
        vpHiddenLayer.push_back( std::make_shared<dgm::dnn::CNeuron>(numNeuronsOutputLayer) );

    for (size_t i = 0; i < numNeuronsOutputLayer; i++)
        vpOutputLayer.push_back( std::make_shared<dgm::dnn::CNeuron>(0) );

    int **trainDataBin   = readImgData("../../../data/digits/train/digit_", dataSize);
    auto trainDataDigit  = readGroundTruth("../../../data/digits/train_gt.txt");
    int **resultsArray   = resultPredictions(numNeuronsOutputLayer);
	assert(trainDataDigit.size() == dataSize);
	
	
    for (size_t i = 0; i < vpHiddenLayer.size(); i++)
        vpHiddenLayer[i]->generateRandomWeights();

    for(size_t i = 0; i < vpInputLayer.size(); i++)
        vpInputLayer[i]->generateRandomWeights();

	dgm::Timer::start("Training...");
	for(int k = 0; k < dataSize; k++) {
            for(size_t i = 0; i < vpInputLayer.size(); i++) {
                float val = (float)trainDataBin[k][i]/255;
                vpInputLayer[i]->setNodeValue(val);
            }

            dotProd(vpInputLayer, vpHiddenLayer);
            dotProd(vpHiddenLayer, vpOutputLayer);
        
            double *resultErrorRate = new double[numNeuronsOutputLayer];
            for(size_t i=0 ; i < vpOutputLayer.size(); i++) {
                resultErrorRate[i] = resultsArray[trainDataDigit[k]][i] - vpOutputLayer[i]->getNodeValue();
            }

            backPropagate(vpInputLayer, vpHiddenLayer, vpOutputLayer, resultErrorRate);
    }
	dgm::Timer::stop();

//     ==================== TEST DIGITS ====================
    int testDataSize    = 2000;
    int correct         = 0;
    int uncorrect       = 0;
    int **testDataBin   = readImgData("../../../data/digits/test/digit_", testDataSize);
    auto testDataDigit  = readGroundTruth("../../../data/digits/test_gt.txt");

	dgm::Timer::start("Testing...");
	for(size_t z = 0; z < testDataSize; z++) {
		 for(size_t i = 0; i < vpInputLayer.size(); i++) {
			 float val = (float)testDataBin[z][i]/255;
			 vpInputLayer[i]->setNodeValue(val);
		 }

		 dotProd(vpInputLayer, vpHiddenLayer);
		 dotProd(vpHiddenLayer, vpOutputLayer);

		 double *allPredictionsforDigits = new double[numNeuronsOutputLayer];
		 for(size_t i=0 ; i < vpOutputLayer.size(); i++) {
			 allPredictionsforDigits[i] = vpOutputLayer[i]->getNodeValue();
		 }

		 float maxAccuracy = 0;
		 int   number;
		 for(size_t i=0 ; i < vpOutputLayer.size(); i++) {
			 if(allPredictionsforDigits[i] >= maxAccuracy) {
				 maxAccuracy = allPredictionsforDigits[i];
				 number = i;
			 }
		 }
//		 std::cout<<"prediction "<<"["<<number<<"] for digit " <<testDataDigit[z] <<" with "<<maxAccuracy<<"% at position: "<<z<<std::endl;
		 number == testDataDigit[z] ? correct++ : uncorrect++;
	}
	dgm::Timer::stop();

	std::cout << "poz: " << correct << std::endl << "neg: " << uncorrect << std::endl;
	std::cout << "average: " << (float)correct/(correct+uncorrect)*100 << "%" << std::endl;
	return 0;
}

void dotProd(std::vector<dgm::dnn::ptr_neuron_t>& vpLayerA, std::vector<dgm::dnn::ptr_neuron_t>& vpLayerB) {
	for(size_t i = 0 ; i < vpLayerB.size(); i++) {
        float value = 0;
        for(const auto& a : vpLayerA)
            value += a->getWeight(i) * a->getNodeValue();

        value = applySigmoidFunction(value);
        vpLayerB[i]->setNodeValue(value);
    }
}

void backPropagate(std::vector<dgm::dnn::ptr_neuron_t>& vpLayerA,
                   std::vector<dgm::dnn::ptr_neuron_t>& vpLayerB,
                   std::vector<dgm::dnn::ptr_neuron_t>& vpLayerC,
                   double resultErrorRate[])
{

    const int numNeuronsInputLayer = 784;
    const int numNeuronsHiddenLayer = 60;
    const int numNeuronsOutputLayer = 10;
    
    float (*DeltaWjk)[numNeuronsOutputLayer]  = new float[numNeuronsHiddenLayer][numNeuronsOutputLayer];
    float (*DeltaWik)[numNeuronsHiddenLayer]  = new float[numNeuronsInputLayer][numNeuronsHiddenLayer];
    float *DeltaJ                             = new float[numNeuronsHiddenLayer];
    float learningRate                        = 0.1;

    for(size_t i = 0; i < vpLayerB.size(); i++) {
        double nodeVal = 0;
        for(size_t j = 0; j < vpLayerC.size(); j++) {
            nodeVal += vpLayerB[i]->getWeight(j) * resultErrorRate[j];
            DeltaWjk[i][j] = learningRate * resultErrorRate[j]* vpLayerB[i]->getNodeValue();
        }
        float sigmoid = applySigmoidFunction(vpLayerB[i]->getNodeValue());
        DeltaJ[i] = nodeVal * sigmoid * (1-sigmoid);
    }

    for(size_t i = 0; i < vpLayerA.size(); i++) {
        for(size_t j = 0; j < vpLayerB.size(); j++) {
            DeltaWik[i][j] = learningRate * DeltaJ[j] * vpLayerA[i]->getNodeValue();
            float oldWeight = vpLayerA[i]->getWeight(j);
            vpLayerA[i]->setWeight(j, oldWeight + DeltaWik[i][j]);
        }
    }

    for(size_t i = 0; i < vpLayerB.size(); i++) {
        for(size_t j = 0; j < vpLayerC.size(); j++) {
            float oldWeight = vpLayerB[i]->getWeight(j);
            vpLayerB[i]->setWeight(j, oldWeight + DeltaWjk[i][j]);
        }
    }
}



//    ==== READ IMAGE DATA FROM PIXELS ====
//    int **trainDataBin = readImgData(200);
//    static int testDataBin[2000][784];
//
//    for(int m=0; m<25; m++) {
//        int num = m;
//        std::string number = std::to_string(num);
//        std::string path = "/Users/diond/Desktop/train_images_4000/digit_" + number + ".png";
//        std::string image_path = samples::findFile(path);
//
//        Mat img = imread(image_path, 0);
//
//        if(img.empty()) {
//            std::cout << "Could not read the image: " << image_path << std::endl;
//            return 1;
//        }
//
//        int l=0;
//        for(int i = 0; i < 28; i++)
//        {
//            for(int j = 0; j < 28; j++)
//            {
//                int value = abs((int)img.at<uchar>(i,j) - 255);
//                testDataBin[m][l] = value;
//                l++;
//            }
//        }
//    }
    
    
//    ==== CREATE IMAGES FROM DATA ====
//     int testDataSize    = 2000;
//     int *testDataDigit  = readDigitData("/Users/diond/Desktop/b_digit.txt", testDataSize);
//     int **testDataBin   = readBinData("/Users/diond/Desktop/b_data.txt", testDataSize);
//
//    for(int z=0; z<2000; z++){
//        Mat img(Size(28,28), CV_8UC3, Scalar(255,255,255));
//
//        //sprintf(filename, "digit_%04i.png", i);
//
//        std::stringstream ss;
//        ss << std::setfill('0') << std::setw(4);
//        ss << z;
//        std::string number = ss.str();
//
//        std::string path = "/Users/diond/Desktop/test_images/digit_" + number + ".png";
//
//            int arr2[28][28];
//            int l=0;
//            for(int i=0; i<28; i++){
//                for(int j=0; j<28; j++){
//                    arr2[i][j]=testDataBin[z][l];
//                    l++;
//                }
//            }
//            cv::Vec3b color = img.at<Vec3b>(Point(0,0));
//            for(int i = 0; i < 28; i++)
//            {
//                for(int j = 0; j < 28; j++)
//                {
//                    Vec3b bgrPixel = img.at<Vec3b>(j, i);
//                    //img.at<Vec3b>(i+100, j+20) = 255;
//                    if(arr2[i][j] == 0){
//                        color[0]=255;
//                        color[1]=255;
//                        color[2]=255;
//                    }
//                    else {
//                        color[0]= 255 - arr2[i][j];
//                        color[1]= 255 - arr2[i][j];
//                        color[2]= 255 - arr2[i][j];
//                    }
//                    img.at<Vec3b>(Point(j,i)) = color;
//                }
//            }
//            imwrite(path, img);
//        }

