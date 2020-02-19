#include "PDFGaussian.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
	void CPDFGaussian::reset(void)
	{
		m_mu = 0;
		m_sigma2 = 0;
	}

	void CPDFGaussian::addPoint(Scalar point)
	{
		if (m_nPoints == 0) m_mu = point[0];
		else {
			long double a = static_cast<double>(m_nPoints) / (m_nPoints + 1.0);
			m_mu = a * m_mu + (1.0 - a) * point[0];

			double cr = (point[0] - m_mu) * (point[0] - m_mu);
			m_sigma2 = cr + a *  (m_sigma2 - cr);
		}
		m_nPoints++;
	}

	double CPDFGaussian::getDensity(Scalar point)
	{
		return exp(- 0.5f * (point[0] - m_mu) * (point[0] - m_mu) / m_sigma2) /
			(sqrt(2.0f * m_sigma2 * Pi));
	}

	void CPDFGaussian::smooth(unsigned int nIt)
	{
		DGM_WARNING("This function is not implemented");
	}

	void CPDFGaussian::saveFile(FILE *pFile) const
	{
		fwrite(&m_mu,	  sizeof(double), 1, pFile);
		fwrite(&m_sigma2, sizeof(double), 1, pFile);
	}

	void CPDFGaussian::loadFile(FILE *pFile)
	{
		fread(&m_mu,     sizeof(double), 1, pFile);
		fread(&m_sigma2, sizeof(double), 1, pFile);
	}
}
