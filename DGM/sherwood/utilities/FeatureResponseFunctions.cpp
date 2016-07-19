#include "FeatureResponseFunctions.h"
#include "DataPointCollection.h"
#include "..\Random.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
	// Constructor
	LinearFeatureResponse::LinearFeatureResponse(void) : m_pDx(NULL), m_nFeatures(0) 
	{ }

	// Constuctor
	LinearFeatureResponse::LinearFeatureResponse(const LinearFeatureResponse &copy)
	{
		m_nFeatures = copy.m_nFeatures;
		m_pDx = new float[m_nFeatures];
		memcpy(m_pDx, copy.m_pDx, m_nFeatures * sizeof(float));
	}

	// Constructor
	LinearFeatureResponse::LinearFeatureResponse(unsigned short nFeatures, float *pDx) : m_nFeatures(nFeatures) 
	{
		m_pDx = new float[nFeatures];
		memcpy(m_pDx, pDx, nFeatures * sizeof(float));
	}

	LinearFeatureResponse & LinearFeatureResponse::operator=(const LinearFeatureResponse &rhs)
	{
		if (this == &rhs) return *this;
		this->m_nFeatures = rhs.m_nFeatures;
		this->m_pDx = new float[m_nFeatures];
		memcpy(this->m_pDx, rhs.m_pDx, m_nFeatures * sizeof(float));
		return *this;
	}

	// Destructor
	LinearFeatureResponse::~LinearFeatureResponse(void)
	{
		if (m_pDx) delete[] m_pDx;
		m_pDx = NULL;
	}

	LinearFeatureResponse LinearFeatureResponse::CreateRandom(unsigned short nFeatures, Random &random)
	{
		float	  R		= 0.0f;
		float	* pDx	= new float[nFeatures];
		for (unsigned short f = 0; f < nFeatures; f++) {
			pDx[f] = static_cast<float>(2.0 * random.NextDouble() - 1.0);
			R += pDx[f] * pDx[f];
		} // f
		R = sqrt(R);
		for (unsigned short f = 0; f < nFeatures; f++) 
			pDx[f] /= R;
		
		LinearFeatureResponse res = LinearFeatureResponse(nFeatures, pDx);
		delete [] pDx;
		return res;
	}

	float LinearFeatureResponse::GetResponse(const IDataPointCollection& data, size_t index) const
	{
		float res = 0.0f;
		const DataPointCollection &concreteData = (const DataPointCollection &) data;
		for (unsigned short f = 0; f < m_nFeatures; f++)
			res += m_pDx[f] * concreteData.GetDataPoint((int) index)[f];

		return res;
	}

} } }

