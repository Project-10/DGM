#include "PDFHistogram.h"

namespace DirectGraphicalModels 
{
	// Constructor
	CPDFHistogram::CPDFHistogram(void) : IPDF()
	{
		memset(m_data, 0, 256 * sizeof(long));
	}

	void CPDFHistogram::reset(void)
	{
		memset(m_data, 0, 256 * sizeof(long));
		m_nPoints = 0;
	}

	void CPDFHistogram::addPoint(Scalar point)
	{
		byte i = static_cast<byte>(MIN(255, MAX(0, point[0])));
		m_data[i]++;
		m_nPoints++;
	}

	double CPDFHistogram::getDensity(Scalar point)
	{
		byte i = static_cast<byte>(MIN(255, MAX(0, point[0])));
		return m_nPoints ? static_cast<double>(m_data[i]) / m_nPoints : 0;
	}

	void CPDFHistogram::smooth(unsigned int nIt)
	{
		long tmp[256];
		for (unsigned int iter = 0; iter < nIt; iter++) {
			memcpy(tmp, m_data, 256 * sizeof(long));
			for (int i = 1; i < 255; i++)
				m_data[i] = static_cast<long>(0.25 * (tmp[i-1] + 2 * tmp[i] + tmp[i+1]));
		} // iterations
	}

	void CPDFHistogram::saveFile(FILE *pFile) const
	{
		fwrite(&m_data, sizeof(long), 256, pFile);
		fwrite(&m_nPoints, sizeof(long), 1,   pFile);
	}

	void CPDFHistogram::loadFile(FILE *pFile)
	{
		fread(&m_data, sizeof(long), 256, pFile);
		fread(&m_nPoints, sizeof(long), 1,   pFile);
	}
}
