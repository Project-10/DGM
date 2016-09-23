R"(
#version 330 core

// Interpolated values from the vertex shaders
in vec3 fragmentColor;

// Ouput data
out vec4 color;


void main()
{
	color = vec4(fragmentColor, 0.75f); 
	//color = vec3(1,0,0);
}
)"
