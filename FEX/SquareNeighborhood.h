// Square Neighbourhood structure inferface
// Written by Sergey G. Kosov in 2015 for Project X
#pragma once

namespace DirectGraphicalModels { namespace fex
{
	/**
	* @brief Square neighborhood structure.
	* @details This structure defines rectangular neighborhood around its base point. 
	* The size of the neighborhoood is given via four gap values: \f$(leftGap+rightGap+1)\times(upperGap+lowerGap+1)\f$,
	* and its shape is defines as depicted at \b Figure \b 1.
	* > This definition of a neighbourhood extends the classical one, where the neighborhood is represented as a square with the base point in the center.
	* @image html square_neighborhood.gif "Fig. 1"
	*/
	typedef struct SqNeighbourhood
	{
		int leftGap;		///< Distance from the base point to the neighborhood's left boundary 
		int rightGap;		///< Distance from the base point to the neighborhood's right boundary
		int upperGap;		///< Distance from the base point to the neighborhood's upper boundary
		int lowerGap;		///< Distance from the base point to the neighborhood's lower boundary
	} SqNeighbourhood;

	/**
	* @brief Some special cases of the base point location inside the neighborhood.
	*/
	enum BasePointLocation {
		BP_CENTER,			///< The base point is located in the neighborhood's center 
		BP_LEFT,			///< The base point is located at the neighborhood's left boundry
		BP_RIGHT,			///< The base point is located in the neighborhood's right boundry 
		BP_TOP,				///< The base point is located in the neighborhood's upper boundry 
		BP_BOTTOM			///< The base point is located in the neighborhood's lower boundry 
	};

	/**
	* @brief Initializes the square neighborhood structure.
	* @param leftGap  Distance from the base point to the neighborhood's left boundary.
	* @param rightGap Distance from the base point to the neighborhood's right boundary.
	* @param upperGap Distance from the base point to the neighborhood's upper boundary.
	* @param lowerGap Distance from the base point to the neighborhood's lower boundary.
	* @returns Initialized square neighborhood of the size: \f$(leftGap+rightGap+1)\times(upperGap+lowerGap+1)\f$ (Ref. @ref SqNeighbourhood).
	*/
	inline SqNeighbourhood sqNeighbourhood(int leftGap, int rightGap, int upperGap, int lowerGap)
	{
		SqNeighbourhood nbhd;
		nbhd.leftGap	= leftGap;
		nbhd.rightGap	= rightGap;
		nbhd.upperGap	= upperGap;
		nbhd.lowerGap	= lowerGap;
		return nbhd;
	}
	/**
	* @brief Initializes the square neighborhood structure with all the same values (base point in the center)
	* @param R Distance from the base point to the neighborhood's boundaries (radius)
	* @returns Initialized square neighborhood of the size \f$(2R + 1)\times(2R + 1)\f$ (Ref. @ref SqNeighbourhood)
	*/
	inline SqNeighbourhood sqNeighbourhoodAll(int R) { return sqNeighbourhood(R, R, R, R); }
	/**
	* @brief Initializes the square neighborhood structure with a pre-define shape
	* @param R Distance from the base point to the neighborhood's boundaries
	* @param location Flag describing the location of the base point (Ref. @ref BasePointLocation and \b Figure \b 2)
	* @image html base_point_location.gif "Fig. 2"
	* @returns Initialized square neighborhood of the size \f$(2R + 1)\times(2R + 1)\f$ (Ref. @ref SqNeighbourhood)
	*/
	inline SqNeighbourhood sqNeighbourhood(int R, BasePointLocation location = BP_CENTER)
	{
		SqNeighbourhood nbhd = sqNeighbourhoodAll(R);
		switch (location) {
			case BP_LEFT:
				nbhd.leftGap	= 0;
				nbhd.rightGap	= 2 * R;
				break;
			case BP_RIGHT:
				nbhd.leftGap	= 2 * R;
				nbhd.rightGap	= 0;
				break;
			case BP_TOP:
				nbhd.upperGap	= 0;
				nbhd.lowerGap	= 2 * R;
				break;
			case BP_BOTTOM:
				nbhd.upperGap	= 2 * R;
				nbhd.lowerGap	= 0;
				break;
		}
		return nbhd;
	}
} }