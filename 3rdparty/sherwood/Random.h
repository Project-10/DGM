#pragma once

// This files defines the Random class, used throughout the forest training
// framework for random number generation.

#include <time.h>
#include <cstdlib>

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
  /// <summary>
  /// Encapsulates random number generation - so as to facilitate
  /// overriding of standard library behaviours.
  /// </summary>
  class Random
  {
  public:
    /// <summary>
    /// Creates a 'random number' generator using a seed derived from the system time.
    /// </summary>
    Random()
    {
      srand ( (unsigned int)(time(NULL)) );
    }

    /// <summary>
    /// Creates a deterministic 'random number' generator using the specified seed.
    /// May be useful for debugging.
    /// </summary>
    Random(unsigned int seed)
    {
      srand ( seed );
    }

    /// <summary>
    /// Generate a positive random number.
    int Next()
    {
      return rand();
    }

    /// <summary>
    /// Generate a random number in the range [0.0, 1.0).
    /// </summary>
    double NextDouble()
    {
      return (double)(rand())/RAND_MAX;
    }

    /// <summary>
    /// Generate a random integer within the sepcified range.
    /// </summary>
    /// <param name="minValue">Inclusive lower bound.</param>
    /// <param name="maxValue">Exclusive upper bound.</param>
    int Next(int minValue, int maxValue)
    {
      return minValue + rand()%(maxValue-minValue);
    }
  };
} } }
