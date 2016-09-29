// Trackball camera control class
// Written by Sergey Kosov in 2016 for Project X
#pragma once

#include "TrackballCamera.h"
#include "GLFW\glfw3.h"

namespace DirectGraphicalModels { namespace vis 
{
	// ================================ Camera Control Class ===============================
	/**
	* @brief Trackball camera control class
	* @details This class implements the control of the @ref CTrackballCamera "Trackball camera" by capturing GLFW mouse events from the target window
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CCameraControl : public CTrackballCamera
	{
	public:
		/**
		* @brief Constructor
		* @details The camera is defined in polar coordinate system with the parameters \b theta, \b phi and \b radius in point \f$P\f$
		* and looks at the origin \f$O\f$, which is initially situated at the zero point.
		* @param window The window that received the mouse evenents
		* @param theta The azimuth (or azimuthal angle) measured from the azimuth reference direction to the orthogonal projection of the line segment \f$OP\f$ on the reference plane
		* @param phi The inclination (or polar angle) between the zenith direction and the line segment \f$OP\f$
		* @param radius The radial distance of the camera to origin, \a i.e. the Euclidian distance between \f$O\f$ and \f$P\f$
		* @param turnSpeed The rotation speed
		* @param scrollSpeed The zoom speed
		* @param panSpeed The panning speed
		*/
		CCameraControl(GLFWwindow * window, float theta = 0.0f, float phi = -glm::pi<float>() / 2, float radius = 2.5f, float turnSpeed = 0.004f, float scrollSpeed = 0.33f, float panSpeed = 0.01f);
		virtual ~CCameraControl(void) {}

	
	private:
		void mouseButtonCallback(int button, int action, int mods);
		void cursorPosCallback(double x, double y);
		void scrollCallback(double x, double y);


	private:
		enum class mouseEvent {none, start_turn, turn, start_pan, pan};
		
		mouseEvent	m_mouseEvent;
		glm::vec2	m_mouseLastPos;
	
		float		m_turnSpeed;
		float		m_scrollSpeed;
		float		m_panSpeed;
	};
} }