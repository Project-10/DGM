// OpenCV Mat Serialization Class Interface
// Written by Sergey Kosov in 2015 for Project X
#pragma once

#include "types.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
	// ================================ Serialize Namespace ==============================
	/**
	* @brief OpenCV Mat Serialization class
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	namespace Serialize
	{
		/**
		* @brief Saves matrix \b m into the file \b fileName
		* @param filename The full path to the destination file
		* @param m Matrix to be saved
		*/
		void to(const std::string &fileName, const Mat &m)
		{
			FILE *pFile = fopen(fileName.c_str(), "wb");
			// Header
			int height = m.rows;
			int width = m.cols;
			int depth = m.depth();
			int channels = m.channels();

			fwrite(&height, sizeof(int), 1, pFile);
			fwrite(&width, sizeof(int), 1, pFile);
			fwrite(&depth, sizeof(int), 1, pFile);
			fwrite(&channels, sizeof(int), 1, pFile);

			int elementSize;
			switch (depth) {
			case CV_8U:	 elementSize = 1; break;
			case CV_8S:  elementSize = 1; break;
			case CV_16U: elementSize = 2; break;
			case CV_16S: elementSize = 2; break;
			case CV_32S: elementSize = 4; break;
			case CV_32F: elementSize = 4; break;
			case CV_64F: elementSize = 8; break;
			case CV_USRTYPE1:
				DGM_WARNING("Custom matrix type is not supported");
				elementSize = 0;
				break;
			}

			fwrite(m.data, elementSize, height * width * channels, pFile);
			fclose(pFile);
		}

		/**
		* @brief Loads the matrix from file \b filename
		* @param filename The full path to the source file
		*/
		Mat from(const std::string &fileName)
		{
			FILE *pFile = fopen(fileName.c_str(), "rb");
			if (!pFile) return Mat();

			// Header
			int height, width, depth, channels;

			fread(&height, sizeof(int), 1, pFile);
			fread(&width, sizeof(int), 1, pFile);
			fread(&depth, sizeof(int), 1, pFile);
			fread(&channels, sizeof(int), 1, pFile);

			int elementSize;
			switch (depth) {
			case CV_8U:	 elementSize = 1; break;
			case CV_8S:  elementSize = 1; break;
			case CV_16U: elementSize = 2; break;
			case CV_16S: elementSize = 2; break;
			case CV_32S: elementSize = 4; break;
			case CV_32F: elementSize = 4; break;
			case CV_64F: elementSize = 8; break;
			case CV_USRTYPE1:
				DGM_WARNING("Custom matrix type is not supported");
				elementSize = 0;
				break;
			}

			Mat res(height, width, CV_MAKETYPE(depth, channels));
			fread(res.data, elementSize, height * width * channels, pFile);
			fclose(pFile);
			return res;
		}
	}
}