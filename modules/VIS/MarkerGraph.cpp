#include "MarkerGraph.h"
#include "DGM\IGraph.h"
#include "macroses.h"

#ifdef USE_OPENGL
#include <fstream> // for LoadShaders
#include "GL\glew.h"
#include "GLFW\glfw3.h"
#include "glm\glm.hpp"
#include "glm\gtc\matrix_transform.hpp"
#include "arcball.h"
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
	GLuint LoadShaders(const char * vertex_file_path, const char * fragment_file_path) {

		// Create the shaders
		GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
		GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

		// Read the Vertex Shader code from the file
		std::string VertexShaderCode;
		std::ifstream VertexShaderStream(vertex_file_path, std::ios::in);
		if (VertexShaderStream.is_open()) {
			std::string Line = "";
			while (getline(VertexShaderStream, Line))
				VertexShaderCode += "\n" + Line;
			VertexShaderStream.close();
		}
		else {
			printf("Impossible to open %s. Are you in the right directory ? Don't forget to read the FAQ !\n", vertex_file_path);
			getchar();
			return 0;
		}

		// Read the Fragment Shader code from the file
		std::string FragmentShaderCode;
		std::ifstream FragmentShaderStream(fragment_file_path, std::ios::in);
		if (FragmentShaderStream.is_open()) {
			std::string Line = "";
			while (getline(FragmentShaderStream, Line))
				FragmentShaderCode += "\n" + Line;
			FragmentShaderStream.close();
		}

		GLint Result = GL_FALSE;
		int InfoLogLength;


		// Compile Vertex Shader
		printf("Compiling shader : %s\n", vertex_file_path);
		char const * VertexSourcePointer = VertexShaderCode.c_str();
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



		// Compile Fragment Shader
		printf("Compiling shader : %s\n", fragment_file_path);
		char const * FragmentSourcePointer = FragmentShaderCode.c_str();
		glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer, NULL);
		glCompileShader(FragmentShaderID);

		// Check Fragment Shader
		glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
		glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
		if (InfoLogLength > 0) {
			std::vector<char> FragmentShaderErrorMessage(InfoLogLength + 1);
			glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
			printf("%s\n", &FragmentShaderErrorMessage[0]);
		}



		// Link the program
		printf("Linking program\n");
		GLuint ProgramID = glCreateProgram();
		glAttachShader(ProgramID, VertexShaderID);
		glAttachShader(ProgramID, FragmentShaderID);
		glLinkProgram(ProgramID);

		// Check the program
		glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
		glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
		if (InfoLogLength > 0) {
			std::vector<char> ProgramErrorMessage(InfoLogLength + 1);
			glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
			printf("%s\n", &ProgramErrorMessage[0]);
		}


		glDetachShader(ProgramID, VertexShaderID);
		glDetachShader(ProgramID, FragmentShaderID);

		glDeleteShader(VertexShaderID);
		glDeleteShader(FragmentShaderID);

		return ProgramID;
	}
	
	// Arcball instance, sadly we put it here, so that it can be referenced in the callbacks 
	static Arcball arcball(0.3f, 0.3f, 5);

	void scrollCallback(GLFWwindow *window, double x, double y) 
	{
		arcball.scrollCallback(window, x, y);
	}

	void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods) 
	{
		arcball.mouseButtonCallback(window, button, action, mods);
	}

	void cursorCallback(GLFWwindow *window, double x, double y) 
	{
		arcball.cursorCallback(window, static_cast<float>(x), static_cast<float>(y));
	}

	
	void render_loop(size_t nNodes, CvPoint3D32f(*posFunc) (size_t nodeId))
	{
		CvPoint3D32f pt1;
		
		glMatrixMode(GL_MODELVIEW);										// Switch to the drawing perspective
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();													// Reset the drawing perspective

		glOrtho(0, 1000, 1000, 0, 1000, -1000);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		//gluLookAt(150, 150, 150, 0, 0, 0, 0, 100, 0);

		// Nodes
		glPointSize(5);
		glBegin(GL_POINTS);
		for (size_t n = 0; n < nNodes; n++) {
			glColor4f(0.3f, 1.0f, 0.3f, 1.0f);
			//glColor4dv(hsv2rgb(DGM_HSV(360.0 * n / nNodes, 255.0, 255.0)).val);
			pt1 = posFunc(n);
			glVertex3f(pt1.x * 1000, pt1.y * 1000, pt1.z * 1000);
		} // n		
		glEnd();
	}
	
	void drawGraph3D(IGraph *pGraph, CvPoint3D32f(*posFunc) (size_t nodeId))
	{
		// Constants
		const int		size = 800;
		const size_t	nNodes = pGraph->getNumNodes();

		// Initialise GLFW
		DGM_ASSERT_MSG(glfwInit(), "Failed to initialize GLFW");

		glfwWindowHint(GLFW_SAMPLES, 4);
		// Tell GLFW to use OpenGL 3.3 
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);			// To make MacOS happy; should not be needed
		// Window creation hints
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);						// Disable window resizing
		
		// Create a windowed mode window and its OpenGL context 
		GLFWwindow *window = glfwCreateWindow(size, size, "Graph Viewer", NULL, NULL);
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
		
		const float _bkgIntencity = static_cast<float>(bkgIntencity) / 255;
		glClearColor(_bkgIntencity, _bkgIntencity, _bkgIntencity, 0.0f);	// Set background color

//		glShadeModel(GL_SMOOTH);											// Select flat or smooth shading
		glEnable(GL_DEPTH_TEST);											// Ebable depth buffer
		glDepthFunc(GL_LESS);												// Specify the value used for depth buffer comparisons
		glEnable(GL_PROGRAM_POINT_SIZE);


//		glFrontFace(GL_CCW);												// Define front- and back-facing polygons
		glEnable(GL_CULL_FACE);
//		glCullFace(GL_BACK);

		// Set the callback functions
		glfwSetScrollCallback(window, scrollCallback);
		glfwSetCursorPosCallback(window, cursorCallback);
		glfwSetMouseButtonCallback(window, mouseButtonCallback);

		GLuint vertex_array_id;
		glGenVertexArrays(1, &vertex_array_id);
		glBindVertexArray(vertex_array_id);

		// Create and compile our GLSL program from the shaders
		GLuint programID = LoadShaders("D:\\Projects\\DGM\\etc\\shaders\\SimpleVertexShader.vertexshader", "D:\\Projects\\DGM\\etc\\shaders\\SimpleFragmentShader.fragmentshader");
		
		// Get a handle for our "MVP" uniform
		GLuint MatrixID = glGetUniformLocation(programID, "MVP");

		// Our vertices. Three consecutive floats give a 3D vertex; Three consecutive vertices give a triangle.
		// A cube has 6 faces with 2 triangles each, so this makes 6*2=12 triangles, and 12*3 vertices
		std::vector<glm::vec3> vertices;
		
		for (size_t n = 0; n < pGraph->getNumNodes(); n++) {
			CvPoint3D32f pt = posFunc(n);
			vertices.push_back(glm::vec3(pt.x, pt.y, pt.z));
		}

		
		static const GLfloat g_vertex_buffer_data[] = {
			-1.0f,-1.0f,-1.0f, // triangle 1 : begin
			-1.0f,-1.0f, 1.0f,
			-1.0f, 1.0f, 1.0f, // triangle 1 : end
			 1.0f, 1.0f,-1.0f, // triangle 2 : begin
			-1.0f,-1.0f,-1.0f,
			-1.0f, 1.0f,-1.0f, // triangle 2 : end
			 1.0f,-1.0f, 1.0f,
			-1.0f,-1.0f,-1.0f,
			 1.0f,-1.0f,-1.0f,
			 1.0f, 1.0f,-1.0f,
			 1.0f,-1.0f,-1.0f,
			-1.0f,-1.0f,-1.0f,
			-1.0f,-1.0f,-1.0f,
			-1.0f, 1.0f, 1.0f,
			-1.0f, 1.0f,-1.0f,
			 1.0f,-1.0f, 1.0f,
			-1.0f,-1.0f, 1.0f,
			-1.0f,-1.0f,-1.0f,
			-1.0f, 1.0f, 1.0f,
			-1.0f,-1.0f, 1.0f,
			 1.0f,-1.0f, 1.0f,
			 1.0f, 1.0f, 1.0f,
			 1.0f,-1.0f,-1.0f,
			 1.0f, 1.0f,-1.0f,
			 1.0f,-1.0f,-1.0f,
			 1.0f, 1.0f, 1.0f,
			 1.0f,-1.0f, 1.0f,
			 1.0f, 1.0f, 1.0f,
			 1.0f, 1.0f,-1.0f,
			-1.0f, 1.0f,-1.0f,
			 1.0f, 1.0f, 1.0f,
			-1.0f, 1.0f,-1.0f,
			-1.0f, 1.0f, 1.0f,
			 1.0f, 1.0f, 1.0f,
			-1.0f, 1.0f, 1.0f,
			 1.0f,-1.0f, 1.0f
		};

		// One color for each vertex. They were generated randomly.
		static const GLfloat g_color_buffer_data[] = {
			0.583f,  0.771f,  0.014f,
			0.609f,  0.115f,  0.436f,
			0.327f,  0.483f,  0.844f,
			0.822f,  0.569f,  0.201f,
			0.435f,  0.602f,  0.223f,
			0.310f,  0.747f,  0.185f,
			0.597f,  0.770f,  0.761f,
			0.559f,  0.436f,  0.730f,
			0.359f,  0.583f,  0.152f,
			0.483f,  0.596f,  0.789f,
			0.559f,  0.861f,  0.639f,
			0.195f,  0.548f,  0.859f,
			0.014f,  0.184f,  0.576f,
			0.771f,  0.328f,  0.970f,
			0.406f,  0.615f,  0.116f,
			0.676f,  0.977f,  0.133f,
			0.971f,  0.572f,  0.833f,
			0.140f,  0.616f,  0.489f,
			0.997f,  0.513f,  0.064f,
			0.945f,  0.719f,  0.592f,
			0.543f,  0.021f,  0.978f,
			0.279f,  0.317f,  0.505f,
			0.167f,  0.620f,  0.077f,
			0.347f,  0.857f,  0.137f,
			0.055f,  0.953f,  0.042f,
			0.714f,  0.505f,  0.345f,
			0.783f,  0.290f,  0.734f,
			0.722f,  0.645f,  0.174f,
			0.302f,  0.455f,  0.848f,
			0.225f,  0.587f,  0.040f,
			0.517f,  0.713f,  0.338f,
			0.053f,  0.959f,  0.120f,
			0.393f,  0.621f,  0.362f,
			0.673f,  0.211f,  0.457f,
			0.820f,  0.883f,  0.371f,
			0.982f,  0.099f,  0.879f
		};
		
		GLuint vertexbuffer;
		glGenBuffers(1, &vertexbuffer);											// Create 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);							// Make this buffer current
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices[0], GL_STATIC_DRAW);	// transmit data

		GLuint colorbuffer;
		glGenBuffers(1, &colorbuffer);
		glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(g_color_buffer_data), g_color_buffer_data, GL_STATIC_DRAW);


		glm::mat4 ModelMatrix		= glm::mat4(1.0f);
		//glm::mat4 ViewMatrix		= glm::lookAt(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0., 0., 0.), glm::vec3(0., 1., 0.));
		glm::mat4 ProjectionMatrix	= glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 100.0f);
		
	


		// Loop until the user closes the window 
		while (!glfwWindowShouldClose(window) && glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS ) {
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);					// Clear information from last draw
			
//			render_loop(nNodes, posFunc);
			
			// Use our shader
			glUseProgram(programID);

			// Compute the MVP matrix from keyboard and mouse input
			//computeMatricesFromInputs(window, size);
			glm::mat4 ViewMatrix = arcball.createViewRotationMatrix();
			glm::mat4 MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;

			// Send our transformation to the currently bound shader, in the "MVP" uniform
			glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);

	
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
//			glEnableVertexAttribArray(1);
//			glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
//			glVertexAttribPointer(
//				1,                  // attribute. No particular reason for 1, but must match the layout in the shader.
//				3,                  // size
//				GL_FLOAT,           // type
//				GL_FALSE,           // normalized ?
//				0,                  // stride
//				(void*)0            // array buffer offset
//			);
			

			// Вывести треугольник!
			glDrawArrays(GL_POINTS, 0, vertices.size()); // 12*3 indices starting at 0 -> 12 triangles -> 6 squares
			glDisableVertexAttribArray(0);
//			glDisableVertexAttribArray(1);

			glfwSwapBuffers(window);											// Swap front and back buffers 
			glfwPollEvents();													// Poll for and process events 
		}

		// Cleanup VBO and shader
		glDeleteBuffers(1, &vertexbuffer);
		glDeleteBuffers(1, &colorbuffer);
		glDeleteProgram(programID);
		glDeleteVertexArrays(1, &vertex_array_id);

		// Close OpenGL window and terminate GLFW
		glfwTerminate();
	}
#endif
} }