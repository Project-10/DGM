#include "HOG.h"
#include "Gradient.h"
#include "macroses.h"

namespace DirectGraphicalModels { namespace fex
{
Mat CHOG::get(const Mat &img, int nBins, SqNeighbourhood nbhd)
{
	DGM_ASSERT_MSG(nBins < CV_CN_MAX, "Number of bins (%d) exceeds the maximum allowed number (%d)", nBins, CV_CN_MAX);
	
	register int	i;						// bins index
	register int	x, y;
	int				width	= img.cols;
	int				height	= img.rows;

	// Converting to one channel image
	Mat	I;
	if (img.channels() != 1) cvtColor(img, I, CV_RGB2GRAY);
	else img.copyTo(I);
	
	// Derivatives
	Mat Ix = CGradient::getDerivativeX(I);
	Mat Iy = CGradient::getDerivativeY(I);

	// Initializing bins and integrals
	Mat *pTemp	= new Mat[nBins];
	Mat *pBins	= new Mat[nBins];
	Mat *pInts	= new Mat[nBins];
	for (i = 0; i < nBins; i++) {
		pTemp[i].create(img.size(), CV_8UC1);
		pBins[i].create(img.size(), CV_32FC1);	pBins[i].setTo(0);
	}

	float	**ppBins	= new float *[nBins];
	double	**ppInts0	= new double*[nBins];
	double	**ppInts1	= new double*[nBins];
	byte	**ppTemp	= new byte  *[nBins];
	
	// Caclculating the bins
	for (y = 0; y < height; y++) {
		float *pIx = Ix.ptr<float>(y);
		float *pIy = Iy.ptr<float>(y);
		for (i = 0; i < nBins; i++) ppBins[i] = pBins[i].ptr<float>(y);
		for (x = 0; x < width; x++) {
			float ix = pIx[x];
			float iy = pIy[x];
			
			// gradient Magnitude
			float gMgn = sqrtf(ix*ix + iy*iy);

			// gradient Orientation
			if (fabs(ix) < FLT_EPSILON) ix = SIGN(ix) * FLT_EPSILON;
			float tg = iy / ix;
			float gOrt = (0.5f + atanf(tg) / (float)Pi) * 180.0f;			// [0°; 180°]

			// filling in the bins
			float gOrtStep = 180.0f / nBins;
			for (i = 0; i < nBins; i++)
				if (gOrt <= (i + 1) * gOrtStep) {
					ppBins[i][x] = gMgn;
					break;
				}
		}
	}

	// Calculating the integrals
	for (i = 0; i < nBins; i++) integral(pBins[i], pInts[i]);
	
	for (y = 0; y < height; y++) {	
		int y0 = MAX(0, y - nbhd.upperGap);		
		int y1 = MIN(y + nbhd.lowerGap, height - 1);
		for (i = 0; i < nBins; i++) ppInts0[i] = pInts[i].ptr<double>(y0);
		for (i = 0; i < nBins; i++) ppInts1[i] = pInts[i].ptr<double>(y1 + 1);
		for (i = 0; i < nBins; i++) ppTemp[i] = pTemp[i].ptr<byte>(y);
		for (x = 0; x < width; x++) {
			int x0 = MAX(0, x - nbhd.leftGap);
			int x1 = MIN(x + nbhd.rightGap, width - 1);

			Mat HOGcell(cvSize(nBins, 1), CV_64FC1);
			double *pHOGcell = HOGcell.ptr<double>(0);
			for (i = 0; i < nBins; i++) pHOGcell[i] = ppInts1[i][x1 + 1] - ppInts1[i][x0] - ppInts0[i][x1 + 1] + ppInts0[i][x0];
			normalize(HOGcell, HOGcell, 255, 0, CV_MINMAX);
			for (i = 0; i < nBins; i++) ppTemp[i][x] = static_cast<byte>(pHOGcell[i]);
			HOGcell.release();
		} // x
	} // y

	Mat res;
	merge(pTemp, nBins, res);

	// Releasing memory
	delete [] pBins;
	delete [] pInts;
	delete [] pTemp;

	return res;	
}
} }