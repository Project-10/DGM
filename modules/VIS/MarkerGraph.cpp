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

	Mat drawGraph(int size, IGraph * pGraph, CvPoint2D32f(*posFunc) (size_t nodeId))
	{
		CvPoint2D32f	pt1, pt2;
		CvScalar		color	= CV_RGB(180, 180, 200);

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
	
			color = colorspaces::hsv2rgb(DGM_HSV(360.0 * n / nNodes, 255.0, 64.0));

			for (size_t c = 0; c < childs.size(); c++) {
				pt2 = posFunc(childs[c]);
				pt2.x = 0.5f * (1 + pt2.x) * size; 
				pt2.y = 0.5f * (1 + pt2.y) * size;

				alpha.setTo(0);
				//arrowedLine(alpha, pt1, pt2, color, 1, CV_AA, 0, 0.05);
				line(alpha, pt1, pt2, color, 1, CV_AA, 0);
				add(res, alpha, res);
			}
		}
		
		// Nodes
		for (size_t n = 0; n < nNodes; n++) {
			color = colorspaces::hsv2rgb(DGM_HSV(360.0 * n / nNodes, 255.0, 255.0));
			pt1 = posFunc(n);
			pt1.x = 0.5f * (1 + pt1.x) * size;
			pt1.y = 0.5f * (1 + pt1.y) * size;
			circle(res, pt1, 1 + size / 333, color, -1, CV_AA);
		} // n
		
		return res;
	}


#ifdef USE_OPENGL
	void LoadShaders(GLuint &NodeProgramID, GLuint &EdgeProgramID)
	{
		GLint	Result = GL_FALSE;
		int		InfoLogLength;
		

		// -------------- Vertex Shader --------------
		GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
		
		{
			const std::string sourceCode =
			#include "VertexShader.glsl"
			;

			// Compile Vertex Shader
			printf("Compiling vertex shader\n");
			char const * VertexSourcePointer = sourceCode.c_str();
			glShaderSource(VertexShaderID, 1, &VertexSourcePointer, NULL);
			glCompileShader(VertexShaderID);

			// Check Vertex Shader
			glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
			glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
			if (InfoLogLength > 0) {
				std::vector<char> VertexShaderErrorMessage(InfoLogLength + 1);
				glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
				printf("%s\n", &VertexShaderErrorMessage[0]);
			}
		}

		// -------------- Node Fragment Shader --------------
		GLuint NodeFragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);
		
		{
			const std::string sourceCode =
			#include "NodeFragmentShader.glsl"
			;

			// Compile Fragment Shader
			printf("Compiling fragment shader\n");
			char const * FragmentSourcePointer = sourceCode.c_str();
			glShaderSource(NodeFragmentShaderID, 1, &FragmentSourcePointer, NULL);
			glCompileShader(NodeFragmentShaderID);

			// Check Fragment Shader
			glGetShaderiv(NodeFragmentShaderID, GL_COMPILE_STATUS, &Result);
			glGetShaderiv(NodeFragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
			if (InfoLogLength > 0) {
				std::vector<char> FragmentShaderErrorMessage(InfoLogLength + 1);
				glGetShaderInfoLog(NodeFragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
				printf("%s\n", &FragmentShaderErrorMessage[0]);
			}
		}

		// -------------- Edge Fragment Shader --------------
		GLuint EdgeFragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

		{
			const std::string sourceCode =
			#include "EdgeFragmentShader.glsl"
				;

			// Compile Fragment Shader
			printf("Compiling fragment shader\n");
			char const * FragmentSourcePointer = sourceCode.c_str();
			glShaderSource(EdgeFragmentShaderID, 1, &FragmentSourcePointer, NULL);
			glCompileShader(EdgeFragmentShaderID);

			// Check Fragment Shader
			glGetShaderiv(EdgeFragmentShaderID, GL_COMPILE_STATUS, &Result);
			glGetShaderiv(EdgeFragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
			if (InfoLogLength > 0) {
				std::vector<char> FragmentShaderErrorMessage(InfoLogLength + 1);
				glGetShaderInfoLog(EdgeFragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
				printf("%s\n", &FragmentShaderErrorMessage[0]);
			}
		}


		// -------------- Program --------------
		printf("Linking program\n");
		NodeProgramID = glCreateProgram();
		glAttachShader(NodeProgramID, VertexShaderID);
		glAttachShader(NodeProgramID, NodeFragmentShaderID);
		glLinkProgram(NodeProgramID);

		// Check the program
		glGetProgramiv(NodeProgramID, GL_LINK_STATUS, &Result);
		glGetProgramiv(NodeProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
		if (InfoLogLength > 0) {
			std::vector<char> ProgramErrorMessage(InfoLogLength + 1);
			glGetProgramInfoLog(NodeProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
			printf("%s\n", &ProgramErrorMessage[0]);
		}

		glDetachShader(NodeProgramID, VertexShaderID);
		glDetachShader(NodeProgramID, NodeFragmentShaderID);

		
		
		printf("Linking program\n");
		EdgeProgramID = glCreateProgram();
		glAttachShader(EdgeProgramID, VertexShaderID);
		glAttachShader(EdgeProgramID, EdgeFragmentShaderID);
		glLinkProgram(EdgeProgramID);


		
		glDeleteShader(VertexShaderID);
		glDeleteShader(NodeFragmentShaderID);
		glDeleteShader(EdgeFragmentShaderID);
	}
	
	void drawGraph3D(int size, IGraph *pGraph, CvPoint3D32f(*posFunc) (size_t nodeId))
	{
		// Constants
		const size_t	nNodes = pGraph->getNumNodes();

		// Initialise GLFW
		DGM_ASSERT_MSG(glfwInit(), "Failed to initialize GLFW");

		glfwWindowHint(GLFW_SAMPLES, 16);
		// Tell GLFW to use OpenGL 3.3 
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);			// To make MacOS happy; should not be needed
		// Window creation hints
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);						// Disable window resizing
		
		// Create a windowed mode window and its OpenGL context 
		GLFWwindow *window = glfwCreateWindow(size, size, "3D Graph Viewer", NULL, NULL);
		if (!window) {
			DGM_WARNING("Unable to create GLFW window");
			glfwTerminate();
			return;
		}
		// Make the window's context current 
		glfwMakeContextCurrent(window);
		glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);				// Ensure we can capture the escape key being pressed below
		// glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);		// Hide the mouse and enable unlimited mouvement

		// Set the mouse at the center of the screen
		glfwPollEvents();
		glfwSetCursorPos(window, size / 2, size / 2);

		// Initialize GLEW
		DGM_ASSERT_MSG(glewInit() == GLEW_OK, "Failed to initialize GLEW");

		printf("OpenGL Ver: %s\n", glGetString(GL_VERSION));
		
		// Arcball instance, sadly we put it here, so that it can be referenced in the callbacks 
		CCameraControl camera(window);
		
		const float _bkgIntencity = static_cast<float>(bkgIntencity) / 255;
		glClearColor(_bkgIntencity, _bkgIntencity, _bkgIntencity, 0.0f);	// Set background color

		glShadeModel(GL_SMOOTH);											// Select flat or smooth shading
		glEnable(GL_DEPTH_TEST);											// Ebable depth buffer
		glDepthFunc(GL_LESS);												// Specify the value used for depth buffer comparisons
		glEnable(GL_PROGRAM_POINT_SIZE);
		glEnable(GL_POINT_SMOOTH);
		// Enable blending
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_CONSTANT_ALPHA);

//		glFrontFace(GL_CCW);												// Define front- and back-facing polygons
		glEnable(GL_CULL_FACE);
//		glCullFace(GL_BACK);

		// Set the callback functions
		//glfwSetScrollCallback(window, scrollCallback);
		//glfwSetCursorPosCallback(window, cursorCallback);
		//glfwSetMouseButtonCallback(window, mouseButtonCallback);

		GLuint vertex_array_id;
		glGenVertexArrays(1, &vertex_array_id);
		glBindVertexArray(vertex_array_id);

		// Create and compile our GLSL program from the shaders
		GLuint NodeProgramID, EdgeProgramID;
		LoadShaders(NodeProgramID, EdgeProgramID);
		
		// Get a handle for our "MVP" uniform
		GLuint NodeMatrixID = glGetUniformLocation(NodeProgramID, "MVP");
		GLuint EdgeMatrixID = glGetUniformLocation(EdgeProgramID, "MVP");

		// Our vertices. Three consecutive floats give a 3D vertex; Three consecutive vertices give a triangle.
		// A cube has 6 faces with 2 triangles each, so this makes 6*2=12 triangles, and 12*3 vertices
		std::vector<unsigned short>	indices;
		std::vector<glm::vec3>		vertices;
		std::vector<glm::vec3>		colors;
		
		// Nodes
		for (size_t n = 0; n < nNodes; n++) {
			CvPoint3D32f pt = posFunc(n);
			vertices.push_back(glm::vec3(pt.x, pt.y, pt.z));
					
			CvScalar color = colorspaces::hsv2rgb(DGM_HSV(360.0 * n / nNodes, 255.0, 255.0));
			colors.push_back(glm::vec3(color.val[2] / 255, color.val[1] / 255, color.val[0] / 255));
		}		
		
		// Edges
		for (size_t n = 0; n < nNodes; n++) {
			vec_size_t childs;
			pGraph->getChildNodes(n, childs);
			
			for (size_t c : childs) {
				indices.push_back(static_cast<unsigned short>(n));
				indices.push_back(static_cast<unsigned short>(c));
			}
		}
		
		

		GLuint vertexbuffer;
		glGenBuffers(1, &vertexbuffer);																		// Create 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);														// Make this buffer current
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices[0], GL_STATIC_DRAW);	// transmit data

		GLuint colorbuffer;
		glGenBuffers(1, &colorbuffer);
		glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
		glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(glm::vec3), &colors[0], GL_STATIC_DRAW);

		GLuint elementbuffer;
		glGenBuffers(1, &elementbuffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned short), &indices[0], GL_STATIC_DRAW);


		glm::mat4 ModelMatrix		= glm::mat4(1.0f);
		//glm::mat4 ViewMatrix		= glm::lookAt(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0., 0., 0.), glm::vec3(0., 1., 0.));
		glm::mat4 ProjectionMatrix	= glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 100.0f);
		

		// Loop until the user closes the window 
		while (!glfwWindowShouldClose(window) && glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS ) {
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);					// Clear information from last draw
			

			glUseProgram(NodeProgramID);

			// Compute the MVP matrix from keyboard and mouse input
			//computeMatricesFromInputs(window, size);
			if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) camera.reset();
			glm::mat4 ViewMatrix = camera.getViewMatrix();
			glm::mat4 MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;

			// Send our transformation to the currently bound shader, in the "MVP" uniform
			glUniformMatrix4fv(NodeMatrixID, 1, GL_FALSE, &MVP[0][0]);

			// 1-st attribute buffer : vertices
			glEnableVertexAttribArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
			glVertexAttribPointer(
				0,                  // attribute. No particular reason for 1, but must match the layout in the shader.
				3,                  // size
				GL_FLOAT,           // type
				GL_FALSE,           // normalized ?
				0,                  // stride
				(void*)0            // array buffer offset
			);
			
			// 2-nd attribute buffer : colors
			glEnableVertexAttribArray(1);
			glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
			glVertexAttribPointer(
				1,                  // attribute. No particular reason for 1, but must match the layout in the shader.
				3,                  // size
				GL_FLOAT,           // type
				GL_FALSE,           // normalized ?
				0,                  // stride
				(void*)0            // array buffer offset
			);
			
			glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(vertices.size()));

			glDisableVertexAttribArray(0);
			glDisableVertexAttribArray(1);

			// --------------------------------------
			
			glUseProgram(EdgeProgramID);

			// Send our transformation to the currently bound shader, in the "MVP" uniform
			glUniformMatrix4fv(EdgeMatrixID, 1, GL_FALSE, &MVP[0][0]);

			// 1-st attribute buffer : vertices
			glEnableVertexAttribArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
			glVertexAttribPointer(
				0,                  // attribute. No particular reason for 1, but must match the layout in the shader.
				3,                  // size
				GL_FLOAT,           // type
				GL_FALSE,           // normalized ?
				0,                  // stride
				(void*)0            // array buffer offset
				);

			// 2-nd attribute buffer : colors
			glEnableVertexAttribArray(1);
			glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
			glVertexAttribPointer(
				1,                  // attribute. No particular reason for 1, but must match the layout in the shader.
				3,                  // size
				GL_FLOAT,           // type
				GL_FALSE,           // normalized ?
				0,                  // stride
				(void*)0            // array buffer offset
				);

			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);			// Index buffer

			glDrawElements(GL_LINES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_SHORT, (void *)0);
			
			glDisableVertexAttribArray(0);
			glDisableVertexAttribArray(1);

			glfwSwapBuffers(window);											// Swap front and back buffers 
			glfwPollEvents();													// Poll for and process events 
		}

		// Cleanup VBO and shader
		glDeleteBuffers(1, &vertexbuffer);
		glDeleteBuffers(1, &colorbuffer);
		glDeleteProgram(NodeProgramID);
		glDeleteProgram(EdgeProgramID);
		glDeleteVertexArrays(1, &vertex_array_id);

		// Close OpenGL window and terminate GLFW
		glfwTerminate();
	}
#endif
} }