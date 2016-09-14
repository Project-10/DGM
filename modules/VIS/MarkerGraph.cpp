#include "MarkerGraph.h"
#include "DGM\IGraph.h"

namespace DirectGraphicalModels { namespace vis
{
// Constants
const byte bkgIntencity = 50;

namespace {
	CvScalar hsv2rgb(CvScalar hsv)
	{
		double      hh, p, q, t, ff;
		long        i;
		CvScalar	out;

		if (hsv.val[1] <= 0.0) {       // < is bogus, just shuts up warnings
			out.val[0] = hsv.val[2];
			out.val[1] = hsv.val[2];
			out.val[2] = hsv.val[2];
			return out;
		}
		hh = hsv.val[0];
		if (hh >= 360.0) hh = 0.0;
		hh /= 60.0;
		i = (long)hh;
		ff = hh - i;
		p = hsv.val[2] * (1.0 - hsv.val[1] / 255.0);
		q = hsv.val[2] * (1.0 - (hsv.val[1] * ff) / 255.0);
		t = hsv.val[2] * (1.0 - (hsv.val[1] * (1.0 - ff)) / 255.0);

		switch (i) {
		case 0:
			out.val[0] = hsv.val[2];
			out.val[1] = t;
			out.val[2] = p;
			break;
		case 1:
			out.val[0] = q;
			out.val[1] = hsv.val[2];
			out.val[2] = p;
			break;
		case 2:
			out.val[0] = p;
			out.val[1] = hsv.val[2];
			out.val[2] = t;
			break;

		case 3:
			out.val[0] = p;
			out.val[1] = q;
			out.val[2] = hsv.val[2];
			break;
		case 4:
			out.val[0] = t;
			out.val[1] = p;
			out.val[2] = hsv.val[2];
			break;
		case 5:
		default:
			out.val[0] = hsv.val[2];
			out.val[1] = p;
			out.val[2] = q;
			break;
		}
		return out;
	}
}

	Mat drawGraph(IGraph * pGraph, CvPoint (*posFunc) (size_t nodeId, int size))
	{
		const int		size	= 1000;
		CvPoint			pt1, pt2;
		CvScalar		color	= CV_RGB(180, 180, 200);

		const size_t	nNodes = pGraph->getNumNodes();
		
		Mat res(size, size, CV_8UC3);
		Mat alpha(size, size, CV_8UC3);
		res.setTo(bkgIntencity);

		// Edges
		for (size_t n = 0; n < nNodes; n++) {
			vec_size_t childs;
			pGraph->getChildNodes(n, childs);
			pt1 = posFunc(n, size);
		
			color = hsv2rgb(DGM_HSV(360.0 * n / nNodes, 255.0, 64.0));
			for (size_t c = 0; c < childs.size(); c++) {
				pt2 = posFunc(childs[c], size);
				alpha.setTo(0);
				arrowedLine(alpha, pt1, pt2, color, 1, CV_AA, 0, 0.05);
			
			
				add(res, alpha, res);
			}
		}
		
		// Nodes
		for (size_t n = 0; n < nNodes; n++) {
			color = hsv2rgb(DGM_HSV(360.0 * n / nNodes, 255.0, 255.0));
			pt1 = posFunc(n, size);
			circle(res, pt1, 4, color, -1, CV_AA);
		} // n
		
		return res;
	}

} }