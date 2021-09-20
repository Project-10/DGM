#include "DNN.h"
#include "DGM.h"
#include "VIS.h"
#include "DGM/timer.h"
#include <fstream>

namespace dgm = DirectGraphicalModels;

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
	const word nFeatures                 = 6;
	const word numNeuronsHiddenLayer     = 3;

	auto pLayerVisible = std::make_shared<dgm::dnn::CNeuronLayer>(nFeatures, 0, [](float x) { return x; }, [](float x) { return 1.0f; });
	auto pLayerHidden  = std::make_shared<dgm::dnn::CNeuronLayer>(numNeuronsHiddenLayer, nFeatures, &sigmoidFunction, &sigmoidFunction_derivative);

	pLayerHidden->generateRandomWeights();

}
