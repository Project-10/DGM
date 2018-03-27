// Timer Class Interface
// Written by Sergey Kosov in 2015 for Project X
#pragma once

#include "types.h"

namespace DirectGraphicalModels
{
	// ================================ Timer Namespace ==============================
	/**
	* @brief %Timer
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	namespace Timer 
	{
		static int64 m_ticks = 0;

		/**
		* @brief Starts the timer
		* @param str Custom string to be printed when the timer has started
		*/
		void start(const std::string &str) 
		{
			printf("%s ", str.c_str());
			m_ticks = getTickCount();
		}

		/**
		* @brief Stops the timer
		* @details This function prints out the time in milliseconds passed between start() and stop()
		*/
		void stop(void) 
		{
			int64 ms = static_cast<int64>(1000 * (getTickCount() - m_ticks) / getTickFrequency());
			int64 sec = 0;
			int64 min = 0;
			int64 hrs = 0;

			if (ms >= 1000) {
				sec = ms / 1000;
				ms = ms % 1000;
			}
			if (sec >= 60) {
				min = sec / 60;
				sec = sec % 60;
			}
			if (min > 60) {
				hrs = min / 60;
				min = min % 60;
			}

			printf("Done! (");
			if (hrs) printf("%lld:", hrs);
			if (min) printf("%lld:", min);
			if (sec) printf("%lld'", sec);
			printf("%03lld ms)\n", ms);
		}
	}
}
