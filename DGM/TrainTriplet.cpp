#include "TrainTriplet.h"

namespace DirectGraphicalModels
{

Mat CTrainTriplet::getTripletPotentials(const Mat &featureVector1, const Mat &featureVector2, const Mat &featureVector3) const
{
	calculateTripletPotentials(featureVector1, featureVector2, featureVector3);
	return Mat();
}

void CTrainTriplet::calculateTripletPotentials(const Mat &featureVector1, const Mat &featureVector2, const Mat &featureVector3) const
{
}

}