R"(
#version 330 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in vec3 vertexColor;

// Output data ; will be interpolated for each fragment.
out vec3 fragmentColor;

// Values that stay constant for the whole mesh.
uniform mat4 MVP;

void main()
{
	// Output position of the vertex, in clip space : MVP * position
	gl_Position =  MVP * vec4(vertexPosition, 1);
	gl_PointSize = max(5.0f, 25.0f / gl_Position.w);

	// The color of each vertex will be interpolated to produce the color of each fragment
	fragmentColor = vertexColor;  
}
)"
