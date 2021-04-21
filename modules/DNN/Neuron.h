#pragma once
#include "types.h"

namespace DirectGraphicalModels {
    namespace dnn
    {
        class CNeuron
        {
        public:
            DllExport CNeuron(void) = delete;
            /**
             * @brief Constructor
			 * @param numConnections The number of incoming connections
			 * @note In feed-forward networks \b numConnections is usually equal to the number of neurons on the previouse layer
             */
            DllExport CNeuron(size_t numConnections);
            DllExport CNeuron(const CNeuron&) = delete;
            DllExport ~CNeuron(void) = default;
            
            DllExport bool      operator=(const CNeuron&) = delete;
            
            DllExport void		generateRandomWeights(void);
            
            // Accessors
            DllExport void      setValue(float value) { m_value = value; }
            DllExport float     getValue(void) const { return m_value; }
            DllExport void      setWeight(size_t index, float weight);
            DllExport float     getWeight(size_t index) const;
            DllExport size_t    getSize(void) const { return m_vWeights.size(); }
            
        private:
            float                 m_value;
            std::vector<float>    m_vWeights;
        };
    
        using ptr_neuron_t = std::shared_ptr<CNeuron>;
    }
}
