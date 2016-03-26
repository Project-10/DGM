#include "PDFGaussian.h"

namespace DirectGraphicalModels
{
void CPDFGaussian::reset(void)
{ 
	m_mu = 0;
	m_sigma2 = 0;
}

void CPDFGaussian::addPoint(float point)
{ 
	if (m_nPoints == 0) m_mu = point;
	else {
		long double a = static_cast<double>(m_nPoints) / (m_nPoints + 1.0);
		m_mu = static_cast<float>(a * m_mu + (1.0 - a) * point);

		float cr = (point - m_mu) * (point - m_mu);
		m_sigma2 = cr + static_cast<float>(a *  (m_sigma2 - cr));
	}
	m_nPoints++;
}

float CPDFGaussian::getDensity(float point) 
{ 
	float res = expf(- 0.5f * (point - m_mu) * (point - m_mu) / m_sigma2) /
		(sqrtf(2.0f * m_sigma2 * static_cast<float>(Pi)));
	
	return res;
}

void CPDFGaussian::saveFile(FILE *pFile) const
{ 
	fwrite(&m_mu,	  sizeof(float), 1, pFile);
	fwrite(&m_sigma2, sizeof(float), 1, pFile);	
}

void CPDFGaussian::loadFile(FILE *pFile)
{ 
	fread(&m_mu,     sizeof(float), 1, pFile);
	fread(&m_sigma2, sizeof(float), 1, pFile);
}
}