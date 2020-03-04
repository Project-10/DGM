#include "TestPDF.h"
#include "DGM/random.h"

using namespace DirectGraphicalModels;

void testPDF_1D(IPDF& pdf) {
	ASSERT_FALSE(pdf.isEstimated());
	
	const double mu 	= random::u<double>(1, 254);
	const double sigma2 = random::u<double>(1, 1000);
	
	CKDGauss gauss(1);
	gauss.setMu(Mat(1, 1, CV_64FC1, mu));
	gauss.setSigma(Mat(1, 1, CV_64FC1, sigma2));
	for (int i = 0; i < 100000; i++) {
		double sample = gauss.getSample().at<double>(0, 0);
		pdf.addPoint(sample);
	}
	ASSERT_TRUE(pdf.isEstimated());

	for (int i = 0; i < 100; i++) {
		double x = random::u<double>(1, 254);
		double pdf_density = pdf.getDensity(x);
		double gt_density = gauss.getAlpha() * gauss.getValue(Mat(1, 1, CV_64FC1, x));
		ASSERT_LE(abs(pdf_density - gt_density), 10e-3);
	}
}

TEST_F(CTestPDF, PDF_Histogram) {
	CPDFHistogram pdf;
	testPDF_1D(pdf);
}

TEST_F(CTestPDF, PDF_Gaussian) {
	CPDFGaussian pdf;
	testPDF_1D(pdf);
}
