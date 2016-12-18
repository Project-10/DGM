#include "Global.h"

namespace DirectGraphicalModels { namespace fex { namespace global
{
size_t getNumLines(const Mat &img, double threshold1, double threshold2)
{
	// Converting to one channel image
	Mat I;
	if (img.channels() != 1) cvtColor(img, I, CV_RGB2GRAY);
	else img.copyTo(I); 
		
	GaussianBlur(I, I, Size(5, 5), 0.75, 0.75);		// smooth it, otherwise a lot of false circles may be detected
	
	Mat canny8b;
	Canny(I, canny8b, threshold1 / 2, threshold1, 3);

#if 1
	std::vector<Vec2f> vLines;
	HoughLines(	  canny8b			// image
				, vLines			// lines
				, 1					// distance resolution
				, 5 * CV_PI / 180	// angle resolution
				, threshold2		// lower -> more lines
			);
#else
	std::vector<Vec4i> vLines;
	HoughLinesP(  canny8b			// image
				, vLines			// lines
				, 1					// distance resolution
				, 1 * CV_PI / 180	// angle resolution
				, threshold2		// lower -> more lines
				, 2					// min line length
				, canny8b.rows / 2	// max line length
			);
#endif
	

	if (false) {		// Visualization
		imshow("Canny", canny8b);
		
		Mat tmp;
		if (img.channels() == 1)  cvtColor(img, tmp, CV_GRAY2RGB);
		else img.copyTo(tmp);

#if 1
		for (Vec2f &l : vLines) {
			float rho	= l[0];
			float theta = l[1];
			double a = cos(theta);
			double b = sin(theta);
			double x0 = a * rho;
			double y0 = b * rho;
			Point pt1, pt2;
			pt1.x = cvRound(x0 - 1000 * b);
			pt1.y = cvRound(y0 + 1000 * a);
			pt2.x = cvRound(x0 + 1000 * b);
			pt2.y = cvRound(y0 - 1000 * a);
			line(tmp, pt1, pt2, CV_RGB(255, 0, 0), 1, CV_AA);
		}
#else
		for (Vec4i &l : vLines) 
			line(tmp, Point(l[0], l[1]), Point(l[2], l[3]), CV_RGB(255, 0, 0), 1, CV_AA);
#endif
		imshow("detected lines", tmp);
		waitKey();
	}
		
	return vLines.size();
}

size_t getNumCircles(const Mat &img, double threshold1, double threshold2)
{
	// Converting to one channel image
	Mat I;
	if (img.channels() != 1) cvtColor(img, I, CV_RGB2GRAY);
	else img.copyTo(I);

	GaussianBlur(I, I, Size(9, 9), 2, 2);		// smooth it, otherwise a lot of false circles may be detected

	std::vector<Vec3f> vCircles;
	HoughCircles(I				// image
		, vCircles				// circles
		, CV_HOUGH_GRADIENT		// method
		, 1						// dp 
		, 1						// min distance between centers
		, threshold1			// high treshold of the canny
		, threshold2			// lower -> more circles
	);	

	if (false) {		// Visualization
		Mat canny8b;
		Canny(I, canny8b, threshold1 / 2, threshold1, 3);
		imshow("Canny", canny8b);

		Mat tmp;
		if (img.channels() == 1)  cvtColor(img, tmp, CV_GRAY2RGB);
		else img.copyTo(tmp);

		for (Vec3f &c : vCircles) {
			Point center(cvRound(c[0]), cvRound(c[1]));
			int radius = cvRound(c[2]);
			circle(tmp, center, 3, CV_RGB(0, 255, 0), -1, 8, 0);		// draw the circle center
			circle(tmp, center, radius, CV_RGB(255, 0, 0), 1, 8, 0);		// draw the circle outline
		}
		imshow("circles", tmp);
		waitKey();
	}

	return vCircles.size();
}

} } }