#pragma once

//#define PRINT_DEBUG_INFO	
//#define DEBUG_MODE			
#define USE_PPL
//#define USE_AMP
#define USE_SHERWOOD

#include <vector>
#include <memory>
#include <thread>
#include <math.h>
#ifdef USE_PPL
#include <ppl.h>
#endif
#ifdef USE_AMP
#include <amp.h>
#endif
#include "opencv.hpp"

using namespace cv;

using byte	= unsigned __int8;
using word	= unsigned __int16;
using dword	= unsigned __int32;
using qword	= unsigned __int64;

using vec_mat_t		= std::vector<Mat>;
using vec_bool_t	= std::vector<bool>;
using vec_byte_t	= std::vector<byte>;
using vec_int_t		= std::vector<int>;
using vec_float_t	= std::vector<float>;
using vec_size_t	= std::vector<size_t>;
using vec_string_t	= std::vector<std::string>;

using ptr_float_u	= std::unique_ptr<float[]>;

const double	Pi		= 3.1415926;			///< Pi number

#define isnan		_isnan
#define isinf		std::isinf
#define DllExport	__declspec(dllexport)

// DGM lib
namespace DirectGraphicalModels
{
	struct  Node;
	using	vec_node_t		= std::vector<Node>;
	struct  Edge;
	using	vec_edge_t		= std::vector<Edge>;
	struct	Triplet;
	using	vec_triplet_t	= std::vector<Triplet>;
	class	CNDGauss;
	using	vec_NDGauss_t	= std::vector<CNDGauss>;
	using   vec_nColor_t	= std::vector<std::pair<Scalar, std::string>>;
	const	size_t	STR_LEN	= 256; 
}
