#pragma once

#include <glm/glm.hpp>

class Arcball {
private:
	float		m_theta;
	float		m_phi;
	float		m_radius; 
	float		m_up;

	glm::vec3	m_target;

	glm::mat4	m_view;
	
	bool		m_viewNeedsUpdate;


	int			m_mouseEvent;
	glm::vec2	m_mouseLastPos;
	float		m_cameraScrollFactor;
	float		m_cameraPanFactor;
	

public:
	/**
	* @brief Constructor
	* @param roll_speed the speed of rotation.
	*/
	Arcball(float theta, float phi, float radius) 
		: m_theta(theta)
		, m_phi(phi)
		, m_radius(radius)
		, m_up(phi >= 0 ? 1 : -1)
		, m_target(0, 0, 0)
		, m_view(glm::mat4(1))
		, m_viewNeedsUpdate(true)
		, m_mouseEvent(0)
		, m_cameraScrollFactor(0.33f)
		, m_cameraPanFactor(0.01f)
	{ }


	inline void scrollCallback(GLFWwindow *window, double x, double y)
	{
		zoom(static_cast<float>(y) * m_cameraScrollFactor);
	}
	/**
	* Check whether we should start the mouse event
	* Event 0: when no tracking occured
	* Event 1: at the start of tracking, recording the first cursor pos
	* Event 11: tracking of subsequent cursor movement
	*/
	inline void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods) 
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
	/**
	*/
	inline void cursorCallback(GLFWwindow *window, float x, float y) 
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
	/**
	* Create rotation matrix within the camera coordinate,
	* multiply this matrix with view matrix to rotate the camera
	*/
	inline glm::mat4 createViewRotationMatrix() 
	{ 
		if (m_viewNeedsUpdate) {
			UpdateViewMatrix();
			m_viewNeedsUpdate = false;
		}
		return m_view;
	}


	protected:
		inline void rotate(float dTheta, float dPhi) 
		{
			m_viewNeedsUpdate = true;

			if (m_up > 0.0f) m_theta += dTheta;
			else 			 m_theta -= dTheta;
			m_phi += dPhi;

			// Keep phi within -2PI to +2PI for easy 'up' comparison 
			if (m_phi > 2 * glm::pi<float>())		m_phi -= 2 * glm::pi<float>();
			else if (m_phi < -2 * glm::pi<float>())	m_phi += 2 * glm::pi<float>();
			
			// If phi is between 0 to PI or -PI to -2PI, make 'up' be positive Y, other wise make it negative Y 
			if ((m_phi > 0 && m_phi < glm::pi<float>()) || (m_phi < -glm::pi<float>() && m_phi > -2 * glm::pi<float>())) m_up = 1.0f;
			else m_up = -1.0f;
		}
		
		inline void zoom(float distance) 
		{ 
			m_viewNeedsUpdate = true;
			m_radius -= distance;
			if (m_radius <= 0.1f) m_radius = 0.1f;
		}
		
		inline void pan(float dx, float dy) 
		{
			m_viewNeedsUpdate = true;
			
			glm::vec3 look		= glm::normalize(toCartesian());		
			glm::vec3 worldUp	= glm::vec3(0.0f, m_up, 0.0f);

			glm::vec3 right = glm::cross(look, worldUp);
			glm::vec3 up = glm::cross(look, right);

			m_target = m_target + (right * dx) + (up * dy);
		}
		
		inline glm::vec3 getCameraPosition(void) const 
		{
			return m_target + glm::vec3(toCartesian());
		}

		inline void UpdateViewMatrix(void) {
			m_view = glm::lookAt(getCameraPosition(), glm::vec3(m_target), glm::vec3(0, m_up, 0));
		}

		inline glm::vec4 toCartesian(void) const 
		{
			float x = m_radius * sinf(m_phi) * sinf(m_theta);
			float y = m_radius * cosf(m_phi);
			float z = m_radius * sinf(m_phi) * cosf(m_theta);
			return glm::vec4(x, y, z, 1);
		}
};

