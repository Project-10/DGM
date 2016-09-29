# include"TrackballCamera.h"

glm::mat4 CTrackballCamera::getViewMatrix(void)
{ 
	if (m_viewNeedsUpdate) {
		updateViewMatrix();
		m_viewNeedsUpdate = false;
	}
	return m_view;
}

// ------------------------------------- Callbacks -------------------------------------
void CTrackballCamera::scrollCallback(GLFWwindow *window, double x, double y)
{
	zoom(static_cast<float>(y) * m_cameraScrollFactor);
}

void CTrackballCamera::mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{ 
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		if (action == GLFW_PRESS)	m_mouseEvent = 1;
		if (action == GLFW_RELEASE)	m_mouseEvent = 0;
	}
	if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
		if (action == GLFW_PRESS)	m_mouseEvent = 2;
		if (action == GLFW_RELEASE)	m_mouseEvent = 0;
	}
}

void CTrackballCamera::cursorCallback(GLFWwindow *window, float x, float y)
{
	if (m_mouseEvent == 0) return;

	if (m_mouseEvent == 11) {
		float dPhi	 = (m_mouseLastPos.y - y) / 300;
		float dTheta = (m_mouseLastPos.x - x) / 300;
		rotate(dTheta, dPhi);
	}

	if (m_mouseEvent == 12) {
		float dx = m_mouseLastPos.x - x;
		float dy = m_mouseLastPos.y - y;
		pan(-dx * m_cameraPanFactor, dy * m_cameraPanFactor);
	}

	m_mouseLastPos.x = x;
	m_mouseLastPos.y = y;

	if (m_mouseEvent < 10) m_mouseEvent += 10;
}


// ====================================== Potected ======================================
void CTrackballCamera::rotate(float dTheta, float dPhi)
{
	m_viewNeedsUpdate = true;
	
	// Update theta and phi
	if (m_up > 0.0f) m_theta += dTheta;
	else 			 m_theta -= dTheta;
	m_phi += dPhi;

	// Keep phi within -2Pi to +2Pi for easy 'up' comparison 
	if      (m_phi >  2 * glm::pi<float>())	m_phi -= 2 * glm::pi<float>();
	else if (m_phi < -2 * glm::pi<float>())	m_phi += 2 * glm::pi<float>();
	
	// If phi is between 0 to Pi or -Pi to -2Pi, make 'up' be positive Y, other wise make it negative Y 
	if ((m_phi > 0 && m_phi < glm::pi<float>()) || (m_phi < -glm::pi<float>() && m_phi > -2 * glm::pi<float>())) m_up =  1.0f;
	else																										 m_up = -1.0f;
}

void CTrackballCamera::zoom(float distance)
{ 
	m_viewNeedsUpdate = true;
	m_radius -= distance;
	if (m_radius <= 0.1f) m_radius = 0.1f;
}

void CTrackballCamera::pan(float dx, float dy)
{
	m_viewNeedsUpdate = true;
	
	glm::vec3 look		= glm::normalize(getCameraOrintation());		
	glm::vec3 worldUp	= glm::vec3(0.0f, m_up, 0.0f);
	glm::vec3 right		= glm::cross(look, worldUp);
	glm::vec3 up		= glm::cross(look, right);

	m_target += (right * dx) + (up * dy);
}

