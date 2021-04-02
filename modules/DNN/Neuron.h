#pragma once
#include "types.h"
#include <random>

namespace DirectGraphicalModels {
    namespace dnn
    {
        class CNeuron
        {
        public:
            static const int SIZE = 60;
            
            void setNodeValue(double thisValue) {
                m_value = thisValue;
            }

            double getNodeValue() const {
              return m_value;
            }
            
            void generateWeights(){
                unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
                srand(seed);

                for (int i= 0; i < SIZE; i++) {
                    double f = (double)rand() / RAND_MAX;
                    double var = -0.5 + f * ((0.5) - (-0.5));
                    m_weight[i] = var;
                }
            }
            
            void setWeight(int index, double x){
                m_weight[index] = x;
            }

            double getWeight(int i) const {
                return m_weight[i];
            }
            
        private:
            double m_value;
            double m_weight[SIZE];
        };
    }
}
