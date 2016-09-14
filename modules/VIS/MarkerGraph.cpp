#include "MarkerGraph.h"
#include "DGM\IGraph.h"
#include "macroses.h"

#ifdef USE_OPENGL
#include <GLFW\glfw3.h>
#endif

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


#ifdef USE_OPENGL
	void render_loop()
	{
		glClearColor(.7, .1, .1, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glViewport(0, 0, 1024, 768);
		glMatrixMode(GL_PROJECTION);
		//gluPerspective( 65.0, (double)1024/(double)768, 1.0, 60.0 );
		glLoadIdentity();
		glOrtho(0, 1024, 768, 0, 100, -100);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		glPointSize(10);
		glBegin(GL_POINTS);
		glColor4f(1, 1, 1, 1);
		glVertex3f(512, 384, 0);
		glVertex3f(0, 384, 0);
		glEnd();
	}
	
	
	void drawGraph3D(IGraph *pGraph, CvPoint3D64f(*posFunc) (size_t nodeId, int size))
	{
		GLFWwindow * window;

		// Initialise GLFW
		DGM_ASSERT_MSG(glfwInit(), "Failed to initialize GLFW");

		// Create a windowed mode window and its OpenGL context 
		window = glfwCreateWindow(640, 480, "Graph Viewer", NULL, NULL);
		if (!window) {
			glfwTerminate();
			return;
		}

		// Make the window's context current 
		glfwMakeContextCurrent(window);

		glLoadIdentity();

		// Loop until the user closes the window 
		while (!glfwWindowShouldClose(window)) {
			// Render here 
			//glClear(GL_COLOR_BUFFER_BIT);

			
			render_loop();


			// Swap front and back buffers 
			glfwSwapBuffers(window);

			// Poll for and process events 
			glfwPollEvents();
		}

		glfwTerminate();
	}
#endif
} }