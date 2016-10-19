#include "MarkerGraph.h"
#include "DGM\IGraph.h"
#include "ColorSpaces.h"
#include "macroses.h"

#ifdef USE_OPENGL
#include "GL\glew.h"
#include "GLFW\glfw3.h"
#include "glm\glm.hpp"
#include "glm\gtc\matrix_transform.hpp"

#include "CameraControl.h"
#endif

namespace DirectGraphicalModels { namespace vis
{
	// Constants
	const byte bkgIntencity = 50;

	namespace {
		// Draws line with interpolated color
		void drawLine(Mat &img, Point pt1, Point pt2, Scalar color1, Scalar color2, int thikness = 1, int lineType = LINE_8, int shift = 0)
		{
			const int nSegments = 8;

			Point2f Pt1 = static_cast<Point2f>(pt1);
			Point2f inc = static_cast<Point2f>(pt2 - pt1) / nSegments;
			for (int i = 0; i < nSegments; i++) {
				Point2f Pt2 = Pt1 + inc;
				double a = static_cast<double>(i) / nSegments;
				Scalar color = (1 - a) * color1 + a * color2;
				line(img, Pt1, Pt2, color, thikness, lineType, shift);
				Pt1 = Pt2;
			}
		}

		// Draws filled triangle
		void drawTriangle(Mat &img, Point pt1, Point pt2, Point pt3, Scalar color, int lineType = LINE_8, int shift = 0)
		{
			Point triangle[1][3] = { pt1, pt2, pt3 };
			const Point * pts[1] = { triangle[0] };
			int npts = 3;
			fillPoly(img, pts, &npts, 1, color, lineType, shift);
		}

		// Draws an arrow
		void drawArrow(Mat &img, Point pt1, Point pt2, Scalar color, int lineType = LINE_8, int shift = 0, double tipLength = 15)
		{
			Point2f dir = static_cast<Point2f>(pt1 - pt2);
			float len = sqrtf(dir.dot(dir));
			dir *= tipLength / len;

			float alpha = 15 * static_cast<float>(Pi) / 180;
			float cs = cosf(alpha);
			float sn = sinf(alpha);
			
			Point2f left( dir.dot(Point2f(cs, -sn)) + 0.5f, dir.dot(Point2f( sn, cs)) + 0.5f);
			Point2f right(dir.dot(Point2f(cs,  sn)) + 0.5f, dir.dot(Point2f(-sn, cs)) + 0.5f);

			drawTriangle(img, pt2, pt2 + static_cast<Point>(left), pt2 + static_cast<Point>(right), color, lineType, shift);
		}

		void drawArrowedLine(Mat &img, Point pt1, Point pt2, Scalar color, int thikness = 1, int lineType = LINE_8, int shift = 0, double tipLength = 15)
		{
			line(img, pt1, pt2, color, thikness, lineType, shift);
			drawArrow(img, pt1, pt2, color, lineType, shift, tipLength);
		}

		void drawArrowedLine(Mat &img, Point pt1, Point pt2, Scalar color1, Scalar color2, int thikness = 1, int lineType = LINE_8, int shift = 0, double tipLength = 15)
		{
			drawLine(img, pt1, pt2, color1, color2, thikness, lineType, shift);
			drawArrow(img, pt1, pt2, color2, lineType, shift, tipLength);
		}
	}

	Mat drawGraph(int size, IGraph * pGraph, std::function<Point2f(size_t)> posFunc, std::function<CvScalar(size_t)> colorFunc, const vec_scalar_t &groupsColor)
	{
		Point2f	pt1, pt2;
		Scalar	color1, color2;

		const size_t	nNodes = pGraph->getNumNodes();
		
		Mat res(size, size, CV_8UC3);
		Mat alpha(size, size, CV_8UC3);
		res.setTo(bkgIntencity);

		// Edges
		for (size_t n = 0; n < nNodes; n++) {
			vec_size_t childs;
			pGraph->getChildNodes(n, childs);

			pt1 = posFunc(n);
			pt1.x = 0.5f * (1 + pt1.x) * size; 
			pt1.y = 0.5f * (1 + pt1.y) * size;
	
			color1 = colorFunc ? colorFunc(n) : colorspaces::hsv2rgb(DGM_HSV(360.0 * n / nNodes, 255.0, 192.0));

			for (size_t c : childs) {
				if (pGraph->isEdgeArc(n, c) && n < c) continue;			// draw only one edge in arc
			
				pt2 = posFunc(c);
				pt2.x = 0.5f * (1 + pt2.x) * size; 
				pt2.y = 0.5f * (1 + pt2.y) * size;
				color2 = colorFunc ? colorFunc(c) : colorspaces::hsv2rgb(DGM_HSV(360.0 * c / nNodes, 255.0, 192.0));

				// Group edge color
				if (groupsColor.size() > 0) 
					color1 = color2 = groupsColor[pGraph->getEdgeGroup(n, c) % groupsColor.size()];

				alpha.setTo(0);
				if (pGraph->isEdgeArc(n, c))	drawLine(alpha, pt1, pt2, color1, color2, 1, CV_AA);
				else							drawArrowedLine(alpha, pt1, pt2, color1, color2, 1, CV_AA);
				add(res, alpha, res);
			}
		}
		
		// Nodes
		for (size_t n = 0; n < nNodes; n++) {
			color1 = colorFunc ? colorFunc(n) : colorspaces::hsv2rgb(DGM_HSV(360.0 * n / nNodes, 255.0, 255.0));
			pt1 = posFunc(n);
			pt1.x = 0.5f * (1 + pt1.x) * size;
			pt1.y = 0.5f * (1 + pt1.y) * size;
			circle(res, pt1, 3, color1, -1, CV_AA);
		} // n
		
		return res;
	}

#ifdef USE_OPENGL
	// types
	using vec_vec3_t = std::vector<glm::vec3>;
	
	namespace {
		void LoadShaders(GLuint &NodeProgramID, GLuint &EdgeProgramID)
		{
			GLint	Result			= GL_FALSE;
			int		InfoLogLength	= 0;

			// -------------- Vertex Shader --------------
			GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
			{  
				const std::string sourceCode =
#include "VertexShader.glsl"
					;
#ifdef DEBUG_PRINT_INFO
				printf("Compiling vertex shader... ");
#endif
				char const * VertexSourcePointer = sourceCode.c_str();
				glShaderSource(VertexShaderID, 1, &VertexSourcePointer, NULL);
				glCompileShader(VertexShaderID);

#ifdef DEBUG_MODE	// Check Vertex Shader
				glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS,  &Result);
				glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
				if (InfoLogLength > 1) {
					std::vector<char> VertexShaderErrorMessage(InfoLogLength + 1);
					glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
					printf("\n%s\n", &VertexShaderErrorMessage[0]);
				}
#endif
#ifdef DEBUG_PRINT_INFO
				printf("Done\n");
#endif
			}

			// -------------- Node Fragment Shader --------------
			GLuint NodeFragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);
			{
				const std::string sourceCode =
#include "NodeFragmentShader.glsl"
					;
#ifdef DEBUG_PRINT_INFO
				printf("Compiling node fragment Shader... ");
#endif
				char const * FragmentSourcePointer = sourceCode.c_str();
				glShaderSource(NodeFragmentShaderID, 1, &FragmentSourcePointer, NULL);
				glCompileShader(NodeFragmentShaderID);

#ifdef DEBUG_MODE	// Check Fragment Shader
				glGetShaderiv(NodeFragmentShaderID, GL_COMPILE_STATUS, &Result);
				glGetShaderiv(NodeFragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
				if (InfoLogLength > 1) {
					std::vector<char> FragmentShaderErrorMessage(InfoLogLength + 1);
					glGetShaderInfoLog(NodeFragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
					printf("\n%s\n", &FragmentShaderErrorMessage[0]);
				}
#endif
#ifdef DEBUG_PRINT_INFO
				printf("Done\n");
#endif
			}

			// -------------- Edge Fragment Shader --------------
			GLuint EdgeFragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);
			{
				const std::string sourceCode =
#include "EdgeFragmentShader.glsl"
					;
#ifdef DEBUG_PRINT_INFO
				printf("Compiling edge fragment shader... ");
#endif
				char const * FragmentSourcePointer = sourceCode.c_str();
				glShaderSource(EdgeFragmentShaderID, 1, &FragmentSourcePointer, NULL);
				glCompileShader(EdgeFragmentShaderID);

#ifdef DEBUG_MODE	// Check Fragment Shader
				glGetShaderiv(EdgeFragmentShaderID, GL_COMPILE_STATUS, &Result);
				glGetShaderiv(EdgeFragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
				if (InfoLogLength > 1) {
					std::vector<char> FragmentShaderErrorMessage(InfoLogLength + 1);
					glGetShaderInfoLog(EdgeFragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
					printf("\n%s\n", &FragmentShaderErrorMessage[0]);
				}
#endif
#ifdef DEBUG_PRINT_INFO
				printf("Done\n");
#endif
			}


			// -------------- Node Program --------------
			{
#ifdef DEBUG_PRINT_INFO
				printf("Linking node program... ");
#endif
				NodeProgramID = glCreateProgram();
				glAttachShader(NodeProgramID, VertexShaderID);
				glAttachShader(NodeProgramID, NodeFragmentShaderID);
				glLinkProgram(NodeProgramID);

#ifdef DEBUG_MODE	// Check the program
				glGetProgramiv(NodeProgramID, GL_LINK_STATUS, &Result);
				glGetProgramiv(NodeProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
				if (InfoLogLength > 1) {
					std::vector<char> ProgramErrorMessage(InfoLogLength + 1);
					glGetProgramInfoLog(NodeProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
					printf("%s\n", &ProgramErrorMessage[0]);
				}
#endif
#ifdef DEBUG_PRINT_INFO
				printf("Done\n");
#endif
				glDetachShader(NodeProgramID, VertexShaderID);
				glDetachShader(NodeProgramID, NodeFragmentShaderID);
			}

			// -------------- Edge Program --------------
			{
#ifdef DEBUG_PRINT_INFO
				printf("Linking edge program... ");
#endif
				EdgeProgramID = glCreateProgram();
				glAttachShader(EdgeProgramID, VertexShaderID);
				glAttachShader(EdgeProgramID, EdgeFragmentShaderID);
				glLinkProgram(EdgeProgramID);

#ifdef DEBUG_MODE	// Check the program
				glGetProgramiv(NodeProgramID, GL_LINK_STATUS, &Result);
				glGetProgramiv(NodeProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
				if (InfoLogLength > 1) {
					std::vector<char> ProgramErrorMessage(InfoLogLength + 1);
					glGetProgramInfoLog(NodeProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
					printf("%s\n", &ProgramErrorMessage[0]);
				}
#endif
#ifdef DEBUG_PRINT_INFO
				printf("Done\n");
#endif
				glDetachShader(NodeProgramID, VertexShaderID);
				glDetachShader(NodeProgramID, NodeFragmentShaderID);

			}
			
			glDeleteShader(VertexShaderID);
			glDeleteShader(NodeFragmentShaderID);
			glDeleteShader(EdgeFragmentShaderID);
		}

		void addCone(vec_vec3_t &vVertices, vec_vec3_t &vColors, vec_word_t &vIndices, glm::vec3 pt1, glm::vec3 pt2, glm::vec3 color, float length)
		{
			const int	nSectors = 32;

			word v0 = static_cast<word>(vVertices.size());

			glm::vec3 dir	 = glm::normalize(pt1 - pt2);				// cone's axis
			glm::vec3 normal = glm::cross(dir, glm::vec3(0, 0, 1));		// normal to cone's axis
			if (glm::length(normal) < FLT_EPSILON) normal = glm::cross(dir, glm::vec3(0, 1, 0));
			normal = glm::normalize(normal);
			glm::vec3 top	 = pt2;										// cone's top point
			glm::vec3 mid	 = pt2 + dir * length;						// cone's middle point

			glm::mat4 rmat	 = glm::rotate(glm::mat4(1), glm::radians(360.0f / nSectors), dir);

			glm::vec3 point1 = normal * length * glm::tan(glm::radians(15.0f));
			for (unsigned short i = 0; i < nSectors; i++) {
				glm::vec3 point2 = rmat * glm::vec4(point1, 1.0f);
				vVertices.push_back(mid + point1);
				vColors.push_back(color);
				point1 = point2;
			}

			vVertices.push_back(top);
			vColors.push_back(color);

			vVertices.push_back(mid);
			vColors.push_back(color);

			for (word i = 1; i <= nSectors; i++) {
				// Side triangle
				vIndices.push_back(v0 + i - 1);
				vIndices.push_back(v0 + i % nSectors);
				vIndices.push_back(v0 + nSectors);

				// Base triangle
				vIndices.push_back(v0 + i % nSectors);
				vIndices.push_back(v0 + i - 1);
				vIndices.push_back(v0 + nSectors + 1);
			}
		}
	}

	void drawGraph3D(int size, IGraph *pGraph, std::function<Point3f(size_t)> posFunc, std::function<CvScalar(size_t)> colorFunc, const vec_scalar_t &groupsColor)
	{
		// Constants
		const size_t	nNodes = pGraph->getNumNodes();

		// Initialise GLFW
		DGM_ASSERT_MSG(glfwInit(), "Failed to initialize GLFW");

		// Window creation hints													
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);					// Tell GLFW to use OpenGL 3.3 
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);			// To make MacOS happy; should not be needed
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);	
		glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);						// Disable window resizing
		glfwWindowHint(GLFW_SAMPLES, 16);

		// Create a windowed mode window and its OpenGL context 
		GLFWwindow *window = glfwCreateWindow(size, size, "3D Graph Viewer", NULL, NULL);
		if (!window) {
			DGM_WARNING("Unable to create GLFW window");
			glfwTerminate();
			return;
		}
		
		glfwMakeContextCurrent(window);										// Make the window's context current 
		glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);				// Ensure we can capture the escape key being pressed below

		// Initialize GLEW
		DGM_ASSERT_MSG(glewInit() == GLEW_OK, "Failed to initialize GLEW");

#ifdef DEBUG_PRINT_INFO
		printf("OpenGL Ver: %s\n", glGetString(GL_VERSION));
#endif		

		// Options
		glShadeModel(GL_SMOOTH);											// Select flat or smooth shading
		//glEnable(GL_DEPTH_TEST);											// Ebable depth buffer
		//glDepthFunc(GL_LESS);												// Specify the value used for depth buffer comparisons
		glEnable(GL_PROGRAM_POINT_SIZE);
		glEnable(GL_BLEND);													// Enable blending
		glBlendFunc(GL_SRC_ALPHA, GL_ONE);
		glFrontFace(GL_CW);													// Define front- and back-facing polygons
		glEnable(GL_CULL_FACE);
		//glCullFace(GL_BACK);

		GLuint vertex_array_id;
		glGenVertexArrays(1, &vertex_array_id);
		glBindVertexArray(vertex_array_id);

		// Create and compile our GLSL program from the shaders
		GLuint NodeProgramID, EdgeProgramID;								// Different shaders for graph nodes and edges
		LoadShaders(NodeProgramID, EdgeProgramID);
		
		// Get a handle for our "MVP" uniform
		GLuint NodeMatrixID = glGetUniformLocation(NodeProgramID, "MVP");
		GLuint EdgeMatrixID = glGetUniformLocation(EdgeProgramID, "MVP");

		// Main containers
		vec_vec3_t vVertices;
		vec_vec3_t vColors;
		vec_vec3_t vGroupColors;
		vec_word_t vIndices, vConeIndices;

		// Filling in the containers
		// Nodes
		for (size_t n = 0; n < nNodes; n++) {
			Point3f pt = posFunc(n);
			vVertices.push_back(glm::vec3(pt.x, pt.y, pt.z));
			CvScalar color = colorFunc ? colorspaces::bgr2rgb(colorFunc(n)) : colorspaces::hsv2bgr(DGM_HSV(360.0 * n / nNodes, 255.0, 255.0));
			color = static_cast<Scalar>(color) / 255;
			vColors.push_back(glm::vec3(color.val[0], color.val[1], color.val[2]));
		}		

		// Edges
		for (size_t n = 0; n < nNodes; n++) {
			vec_size_t childs;
			pGraph->getChildNodes(n, childs);
			
			for (size_t c : childs) {
				if (pGraph->isEdgeArc(n, c) && n < c) continue;			// draw only one edge in arc
				vIndices.push_back(static_cast<word>(n));				// src
				vIndices.push_back(static_cast<word>(c));				// dst
				
				if (groupsColor.size() > 0) {
					byte group = pGraph->getEdgeGroup(n, c);
					CvScalar color = groupsColor[group % groupsColor.size()];
					color = static_cast<Scalar>(color) / 255;
					vGroupColors.push_back(glm::vec3(color.val[0], color.val[1], color.val[2]));
				}

				if (!pGraph->isEdgeArc(n, c))
					addCone(vVertices, vColors, vConeIndices, vVertices[n], vVertices[c], vColors[c], 30.0f / size);
			}
		}
		const size_t nEdgesIdx = vIndices.size();						// Number of edge indices in the container
		vIndices.insert(vIndices.end(), vConeIndices.begin(), vConeIndices.end());
		vConeIndices.clear();

		// Binding the containers: xxxBuffer to vXXX
		GLuint vertexBuffer;
		glGenBuffers(1, &vertexBuffer);																		// Create 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);														// Make this buffer current
		glBufferData(GL_ARRAY_BUFFER, vVertices.size() * sizeof(glm::vec3), &vVertices[0], GL_STATIC_DRAW);	// transmit data

		GLuint colorBuffer;
		glGenBuffers(1, &colorBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
		glBufferData(GL_ARRAY_BUFFER, vColors.size() * sizeof(glm::vec3), &vColors[0], GL_STATIC_DRAW);

		GLuint groupColorBuffer;
		glGenBuffers(1, &groupColorBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, groupColorBuffer);
		glBufferData(GL_ARRAY_BUFFER, vGroupColors.size() * sizeof(glm::vec3), &vGroupColors[0], GL_STATIC_DRAW);

		GLuint indexBuffer;
		glGenBuffers(1, &indexBuffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, vIndices.size() * sizeof(word), &vIndices[0], GL_STATIC_DRAW);

		CCameraControl camera(window);

		const float _bkgIntencity = static_cast<float>(bkgIntencity) / 255;
		glClearColor(_bkgIntencity, _bkgIntencity, _bkgIntencity, 0.0f);	// Set background color

		glm::mat4 ModelMatrix		= glm::mat4(1.0f);
		//glm::mat4 ViewMatrix		= glm::lookAt(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0., 0., 0.), glm::vec3(0., 1., 0.));
		glm::mat4 ProjectionMatrix	= glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 100.0f);
		
		// Main loop
		while (!glfwWindowShouldClose(window) && glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS ) {
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);				// Clear information from last draw
			
			// Compute the MVP matrix from keyboard and mouse input
			if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) camera.reset();
			glm::mat4 MVP = ProjectionMatrix * camera.getViewMatrix() * ModelMatrix;

			// 1-st attribute buffer : vertices
			glEnableVertexAttribArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);

			// 2-nd attribute buffer : colors
			glEnableVertexAttribArray(1);
			glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);

			if (true) { // Draw nodes
				glUseProgram(NodeProgramID);

				// Send our transformation to the currently bound shader, in the "MVP" uniform
				glUniformMatrix4fv(NodeMatrixID, 1, GL_FALSE, &MVP[0][0]);

				glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(nNodes));
			}
			
			if (true) {	// Draw edges
				glUseProgram(EdgeProgramID);
				
				glUniformMatrix4fv(EdgeMatrixID, 1, GL_FALSE, &MVP[0][0]);	// Send our transformation to the currently bound shader, in the "MVP" uniform

				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);			// Index buffer

				glDrawElements(GL_LINES, static_cast<GLsizei>(nEdgesIdx), GL_UNSIGNED_SHORT, (void *) 0);													// Edges
				glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(vIndices.size() - nEdgesIdx), GL_UNSIGNED_SHORT, (void *) (nEdgesIdx * sizeof(word)));	// Cones
			}

			glDisableVertexAttribArray(0);
			glDisableVertexAttribArray(1);

			glfwSwapBuffers(window);										// Swap front and back buffers 
			glfwPollEvents();												// Poll for and process events 
		}

		// Cleanup VBO and shader
		glDeleteBuffers(1, &vertexBuffer);
		glDeleteBuffers(1, &colorBuffer);
		glDeleteBuffers(1, &groupColorBuffer);
		glDeleteBuffers(1, &indexBuffer);
		
		glDeleteProgram(NodeProgramID);
		glDeleteProgram(EdgeProgramID);
		glDeleteVertexArrays(1, &vertex_array_id);

		// Close OpenGL window and terminate GLFW
		glfwTerminate();
	}
#endif
} }