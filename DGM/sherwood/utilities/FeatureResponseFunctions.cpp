#include "FeatureResponseFunctions.h"

#include <cmath>

#include <sstream>

#include "DataPointCollection.h"
#include "..\Random.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
LinearFeatureResponse::LinearFeatureResponse(unsigned char nFeatures, float *pDx) :m_nFeatures(nFeatures) 
{
	for (register unsigned char f = 0; f < m_nFeatures; f++) m_pDx[f] = pDx[f];
}

LinearFeatureResponse LinearFeatureResponse::CreateRandom(unsigned char nFeatures, Random& random)
{
	register unsigned char	  f;
	float					  R		= 0.0f;
	float					* pDx	= new float[nFeatures];
	for (f = 0; f < nFeatures; f++) {
		pDx[f] = static_cast<float>(2.0 * random.NextDouble() - 1.0);
		R += pDx[f] * pDx[f];
	} // f
	R = sqrt(R);
	for (f = 0; f < nFeatures; f++) pDx[f] /= R;
	LinearFeatureResponse res = LinearFeatureResponse(nFeatures, pDx);
	delete [] pDx;
	return res;

}

float LinearFeatureResponse::GetResponse(const IDataPointCollection& data, size_t index) const
{
	float res = 0.0f;
	const DataPointCollection& concreteData = (const DataPointCollection&) data;
	for (register unsigned char f = 0; f < m_nFeatures; f++)
		res += m_pDx[f] * concreteData.GetDataPoint((int) index)[f];

	return res;
}


} } }
