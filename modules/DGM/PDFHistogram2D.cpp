#include "PDFHistogram2D.h"

namespace DirectGraphicalModels
{
// Constructor
CPDFHistogram2D::CPDFHistogram2D(void) : IPDF()
{ 
	memset(m_data, 0, 256 * 256 * sizeof(long)); 
}

// Destructor
CPDFHistogram2D::~CPDFHistogram2D(void)
{ }

void CPDFHistogram2D::reset(void)
{ 
	memset(m_data, 0, 256 * 256 * sizeof(long)); 
	m_nPoints = 0;
}

void CPDFHistogram2D::addPoint(Scalar point)
{
	byte x = static_cast<byte>(MIN(255, MAX(0, point[0])));
	byte y = static_cast<byte>(MIN(255, MAX(0, point[1])));
	m_data[x][y]++;
	m_nPoints++;
}

double CPDFHistogram2D::getDensity(Scalar point)
{
	byte x = static_cast<byte>(MIN(255, MAX(0, point[0])));
	byte y = static_cast<byte>(MIN(255, MAX(0, point[1])));
	return m_nPoints ? static_cast<double>(m_data[x][y]) / m_nPoints : 0;
}

void CPDFHistogram2D::smooth(int nIt)
{
	long tmp[256][256];
	for (int iter = 0; iter < nIt; iter++) {
		memcpy(tmp, m_data, 256 * 256 * sizeof(long));
		for (int x = 1; x < 255; x++)
			for (int y = 1; y < 255; y++)
				m_data[x][y] = static_cast<long>(0.125 * (tmp[x][y-1] + tmp[x-1][y] + 4 * tmp[x][y] + tmp[x+1][y] + tmp[x][y+1]));
	} // iterations
}

void CPDFHistogram2D::saveFile(FILE *pFile) const
{
	fwrite(&m_data, sizeof(long), 256 * 256, pFile);
	fwrite(&m_nPoints, sizeof(long), 1, pFile);
}

void CPDFHistogram2D::loadFile(FILE *pFile)
{
	fread(&m_data, sizeof(long), 256 * 256, pFile);
	fread(&m_nPoints, sizeof(long), 1, pFile);
}
}