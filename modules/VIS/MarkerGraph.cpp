#include "MarkerGraph.h"
#include "DGM\IGraph.h"
#include "ColorSpaces.h"

namespace DirectGraphicalModels { namespace vis
{
	// Constants
	const byte bkgIntencity = 50;

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
		
			color = colorspaces::hsv2rgb(DGM_HSV(360.0 * n / nNodes, 255.0, 64.0));
			for (size_t c = 0; c < childs.size(); c++) {
				pt2 = posFunc(childs[c], size);
				alpha.setTo(0);
				arrowedLine(alpha, pt1, pt2, color, 1, CV_AA, 0, 0.05);
			
			
				add(res, alpha, res);
			}
		}
		
		// Nodes
		for (size_t n = 0; n < nNodes; n++) {
			color = colorspaces::hsv2rgb(DGM_HSV(360.0 * n / nNodes, 255.0, 255.0));
			pt1 = posFunc(n, size);
			circle(res, pt1, 4, color, -1, CV_AA);
		} // n
		
		return res;
	}

} }