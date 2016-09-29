#pragma once

#include "glm\glm.hpp"
#include "glm\gtc\matrix_transform.hpp"
#include "GLFW\glfw3.h"

class CTrackballCamera 
{
public:
	/**
	* @brief Constructor
	* @param theta
	* @param phi
	* @param radius
	*/
	CTrackballCamera(float theta, float phi, float radius)
		: m_theta(theta)
		, m_phi(phi)
		, m_radius(radius)
		, m_up(phi >= 0 ? 1.0f : -1.0f)
		, m_target(0, 0, 0)
		, m_view(glm::mat4(1))
		, m_viewNeedsUpdate(true)
		, m_mouseEvent(0)
		, m_cameraScrollFactor(0.33f)
		, m_cameraPanFactor(0.01f)
	{ }
	~CTrackballCamera(void) {}

	/**
	* Returns the view matrix within the camera coordinate,
	* multiply this matrix with view matrix to rotate the camera
	*/
	glm::mat4 getViewMatrix(void);

	/**
	*/
	void scrollCallback(GLFWwindow * window, double x, double y);
	/**
	* Check whether we should start the mouse event
	* Event 0: when no tracking occured
	* Event 1: at the start of tracking, recording the first cursor pos
	* Event 11: tracking of subsequent cursor movement
	*/
	void mouseButtonCallback(GLFWwindow * window, int button, int action, int mods);
	/**
	*/
	void cursorCallback(GLFWwindow * window, float x, float y);


protected:
	void		rotate(float dTheta, float dPhi);
	void		zoom(float distance);
	void		pan(float dx, float dy);

	glm::vec3	getCameraOrintation(void) const	{ return glm::vec3(sinf(m_phi) * sinf(m_theta), cosf(m_phi), sinf(m_phi) * cosf(m_theta)); }
	glm::vec3	getCameraPosition(void) const	{ return m_target + getCameraOrintation() * m_radius; }
	void		updateViewMatrix(void)			{ m_view = glm::lookAt(getCameraPosition(), m_target, glm::vec3(0, m_up, 0)); }



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

};

