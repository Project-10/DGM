#include "MarkerHistogram.h"

#include "TrainNodeNaiveBayes.h"
#include "PDF.h"

namespace DirectGraphicalModels
{
// Constants
const CvSize	CMarkerHistogram::margin		= cvSize(25, 16);
const byte		CMarkerHistogram::bkgIntencity	= 50;
const double	CMarkerHistogram::frgWeight		= 0.75;

Mat CMarkerHistogram::drawHistogram(Scalar color) const
{
	const byte		fMaxHeight	= 6;					// The maximal number of feature histograms in a column
	const byte		nFeatures	= m_pNodeTrainer->getNumFeatures();
	const int		activeState = getActiveState(color);

	CvSize			fSize;								// Size of the resulting image in feature histograms
	fSize.width = nFeatures / fMaxHeight;
	if (nFeatures % fMaxHeight != 0) fSize.width++;
	fSize.height = (nFeatures < fMaxHeight) ? nFeatures : fMaxHeight;

	CvSize	resSize;									// Size of the resulting image
	resSize.width  = margin.width  + fSize.width  * (256 + 2 * margin.width);
	resSize.height = margin.height + fSize.height * (100 + margin.height);

	// Legende
	Mat legende = drawLegend(resSize.height - margin.height, activeState);
	resSize.width += legende.cols;

	Mat		res =  Mat(resSize, CV_8UC3);				// Resulting Image
	res.setTo(bkgIntencity);
	
	for (byte f = 0; f < nFeatures; f++) {				// freatures
		int dx = f / fMaxHeight;	dx *= (256 + 2 * margin.width);
		int dy = f % fMaxHeight;	dy *= (100 + margin.height);
		
		Mat featureHistogram = drawFeatureHistogram(f, activeState);
		Rect roi(Point(dx, margin.height + dy), featureHistogram.size());
		featureHistogram.copyTo(res(roi));
		featureHistogram.release();
	} // f


	Rect roi(Point(res.cols - legende.cols, margin.height), legende.size());
	legende.copyTo(res(roi));
	legende.release();

	return res;
}

#ifdef DEBUG_MODE	// --- Debugging ---
Mat	CMarkerHistogram::TEST_drawHistogram(CTrainNode *pTrain) const
{
	CvSize	size;					// window size in pixels
	size.width	= 256 + 20;
	size.height	= 256 + 20;

	Mat res = Mat(size, CV_8UC3); res.setTo(bkgIntencity);
	vec_mat_t vChannels;
	split(res, vChannels);
	
	
	if (typeid(*pTrain) == typeid(CTrainNodeNaiveBayes)) {
		const float koeff = 100000.0f;
		for (int x = 0; x < 255; x++)
			for (int y = 0; y < 255; y++) {
				double data1 = MIN(255.0, koeff * static_cast<double>(dynamic_cast<CTrainNodeNaiveBayes *>(pTrain)->m_H2d[0].data[x][y]) / dynamic_cast<CTrainNodeNaiveBayes *>(pTrain)->m_H2d[0].n);
				vChannels.at(2).at<byte>(256 - y + 10, x + 10) = static_cast<byte>(data1);		// red channel - class1

				double data2 = MIN(255.0, koeff * static_cast<double>(dynamic_cast<CTrainNodeNaiveBayes *>(pTrain)->m_H2d[1].data[x][y]) / dynamic_cast<CTrainNodeNaiveBayes *>(pTrain)->m_H2d[1].n);
				vChannels.at(1).at<byte>(256 - y + 10, x + 10) = static_cast<byte>(data2);		// red channel - class1

				double data3 = MIN(255.0, koeff * static_cast<double>(dynamic_cast<CTrainNodeNaiveBayes *>(pTrain)->m_H2d[2].data[x][y]) / dynamic_cast<CTrainNodeNaiveBayes *>(pTrain)->m_H2d[2].n);
				vChannels.at(0).at<byte>(256 - y + 10, x + 10) = static_cast<byte>(data3);		// red channel - class1
			}
	} else {
		const float koeff = 255.0f; //765.0f;
		for (int x = 0; x < 256; x++)
			for (int y = 0; y < 256; y++) {
				Mat fv(2, 1, CV_8UC1);
				fv.ptr<byte>(0)[0] = static_cast<byte>(x);
				fv.ptr<byte>(1)[0] = static_cast<byte>(y);
				Mat pot = pTrain->getNodePotentials(fv);

				for (int s = 0; s < pot.rows; s++) {
					float data = MIN(255.0f, koeff * pot.at<float>(s,0));
					vChannels.at(2-s).at<byte>(255 - y + 10, x + 10) = static_cast<byte>(data);		
				}
				for (int s = pot.rows; s < 3; s++) {
					vChannels.at(2-s).at<byte>(255 - y + 10, x + 10) = 0;		
				}
			}
	}

	merge(vChannels, res);
	vChannels.clear();

	rectangle(res, Point(9, 9), Point(255 + 11, 255 + 11), CV_RGB(200, 200, 200));
	return res;
}
#endif			// --- --------- ---

// ======================================== Private ========================================

int CMarkerHistogram::getActiveState(Scalar color) const
{
	const size_t n = m_vPalette.size();
	for (byte s = 0; s < n; s++) {						// states
		int sum = 0;
		for (int i = 0; i < 3; i++)	sum += static_cast<int>(fabs(bkgIntencity + frgWeight * m_vPalette.at(s).first.val[i] - color.val[i]));
		if (sum == 0) return s;
	}
	return -1;
}

Mat CMarkerHistogram::drawFeatureHistogram(byte f, int activeState) const
{
	int				x, y;
	char			str[256];
	const byte		nStates	= m_pNodeTrainer->getNumStates();
	const int		koeff	= 1200;						// coefficient for histogram value enlargement
	const size_t	n		= m_vPalette.size();

	Mat		res(margin.height + 100, 2 * margin.width + 256, CV_8UC3);	res.setTo(bkgIntencity);
	Mat		tmp(res.size(), res.type());								tmp.setTo(0);
	
	// axis grid
	for (x = 0; x <= 255; x += 51) {
		y = ((x > 75) && (x < 180)) ? 25 : 0;
		line(tmp, Point(margin.width + x, y), Point(margin.width + x, 100), CV_RGB(50, 50, 50));
	}

	// histogram
	if (typeid(*m_pNodeTrainer) == typeid(CTrainNodeNaiveBayes))
		for (byte s = 0; s < nStates; s++) {				// states
				CPDF *pPDF = dynamic_cast<const CTrainNodeNaiveBayes *>(m_pNodeTrainer)->getPDF(s, f);
				for (x = 0; x < 256; x++) {
					int len	=  static_cast<int>(koeff * pPDF->getDensity(static_cast<float>(x)));
					y = 100 - MIN(99, len);
				
					if ((activeState == -1) || (activeState == s % n))
						line(tmp, Point(margin.width + x, 100), Point(margin.width + x, y), m_vPalette.at(s % n).first);
				} // x
			addWeighted(res, 1.0, tmp, frgWeight, 0.0, res);
			tmp.setTo(0);
		} // s
	tmp.release();

	// Feature Names
	if (m_ppFeatureNames == NULL) sprintf(str, "feature %d", f);
	else						  sprintf(str, "%s", m_ppFeatureNames[f]);
	CvSize textSize = getTextSize(str, CV_FONT_HERSHEY_SIMPLEX, 0.5, 1, NULL);
	putText(res, str, Point(margin.width + (MAX(256 - textSize.width, 108)) / 2, 16), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(125, 125, 125), 1, CV_AA);

	// Figure box
	rectangle(res, Point(margin.width - 1, 0), Point(margin.width + 256, 100), CV_RGB(255,255,255));
	for (x = 0; x <= 255; x += 51) {
		sprintf(str, "%d", x);
		putText(res, str, Point(margin.width + x - 5, 109), FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(255, 255, 255), 1, CV_AA);
	}
	
	return res;
}

Mat CMarkerHistogram::drawLegend(int maxHeight, int activeState) const
{
	char			str[256];
	const byte		sMaxHeight	= maxHeight / (2 * margin.height);					// The maximal number of states in a column
	const byte		nStates		= m_pNodeTrainer->getNumStates();
	const size_t	n			= m_vPalette.size();
	
	CvSize			sSize;								// Size of the resulting image in states
	sSize.width = (nStates + 1) / sMaxHeight;
	if ((nStates + 1) % sMaxHeight != 0) sSize.width++;
	sSize.height = ((nStates + 1) < sMaxHeight) ? (nStates + 1) : sMaxHeight;

	CvSize			resSize;							// Size of the resulting image
	resSize.width  = sSize.width  * 120;
	resSize.height = 2 * margin.height * sSize.height;

	Mat res(resSize,  CV_8UC3);	res.setTo(bkgIntencity);
	Mat tmp(res.size(), res.type());							tmp.setTo(0);

	for (byte s = 0; s < nStates; s++) {				// states
		int dx = s / sMaxHeight;	dx *= 120;
		int dy = s % sMaxHeight;	dy *= 2 * margin.height;

		// Color box
		rectangle(tmp, Point(dx, dy), Point(dx + margin.width, dy + margin.height), m_vPalette.at(s % n).first, -1);
		
		// Active triangle
		if (activeState == s)  {
			const Point triangle[3] = {Point(dx, dy), Point(dx + margin.height / 2, dy + margin.height / 2), Point(dx, dy + margin.height)}; 
			fillConvexPoly(tmp, triangle, 3, CV_RGB(0, 0, 0));
		}
		
		// Class name
		if (m_vPalette.at(s % n).second.empty()) sprintf(str, "c%d", s);
		else  sprintf(str, "%s", m_vPalette.at(s % n).second.c_str());
		putText(tmp, str, Point(dx + margin.width + 5, dy + margin.height - 3), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 255, 255), 1, CV_AA);		
	} // s
	
	// White "all" color box
	int dx = nStates / sMaxHeight;	dx *= 120;
	int dy = nStates % sMaxHeight;	dy *= 2 * margin.height;
	rectangle(tmp, Point(dx, dy), Point(dx + margin.width, dy + margin.height), CV_RGB(255, 255, 255), -1);
	if (activeState == -1)  {
		const Point triangle[3] = {Point(dx, dy), Point(dx + margin.height / 2, dy + margin.height / 2), Point(dx, dy + margin.height)}; 
		fillConvexPoly(tmp, triangle, 3, CV_RGB(0, 0, 0));
	}	
	putText(tmp, "all", Point(dx + margin.width + 5, dy + margin.height - 3), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 255, 255), 1, CV_AA);

	addWeighted(res, 1.0, tmp, frgWeight, 0.0, res);
	tmp.release();

	return res;
}

}