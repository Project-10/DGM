#include "MAP.h"
#include "macroses.h"

namespace DirectGraphicalModels 
{
	float getMAP(const vec_byte_t &predictions, const vec_float_t &potentials, const vec_byte_t &gt, byte state)
	{
		// Assertions
		DGM_ASSERT(predictions.size() == potentials.size());
		DGM_ASSERT(predictions.size() == gt.size());
		
		std::vector<std::tuple<byte, float, byte>> container;			// prediction | pot | gt
		for (size_t i = 0; i < predictions.size(); i++)
			container.push_back(std::make_tuple(predictions[i], potentials[i], gt[i]));
		std::sort(container.begin(), container.end(), [](auto const &left, auto const &right) { return std::get<1>(left) > std::get<1>(right); });
	
		float	res				= 0.0f;
		int		nRelevants		= 0;					// number of elevant elements
		int		nCoincidences	= 0;					// number of correct predictions
		int		i				= 0;
		for (auto &c : container) {
			i++;
			if (std::get<2>(c) == state) {				// if (gt == state)
				nRelevants++;
				if (std::get<0>(c) == state) {			// if (prediction = state)
					nCoincidences++;
					res += static_cast<float>(nCoincidences) / i;
				}
			}
		}
		
		DGM_IF_WARNING(nRelevants == 0, "There are no states <state> in <gt>");

		if (nRelevants > 0) res /= nRelevants;

		return res;
	}

}