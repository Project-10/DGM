#pragma once

// This file declares the ProgressStream class, used for progress reporting
// within the forest training framework.

#include <ostream>

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
  /// <summary>
  /// The verbosity associated with a ProgressStream (or a progress message
  /// reported to a ProgressStream).
  /// </summary>
  enum Verbosity { Silent=0, Error, Warning, Interest, Verbose };

  /// <summary>
  /// Encapsulates writing progress messages to a user-supplied std::ostream
  /// with user-defined verbosity.
  /// </summary>

  // Can be used in a std::ostream-like manner, e.g.
  // progressStream[Interest] << "Some progress has been made. Now at " << x << "%." << std::endl;

  class ProgressStream
  {
    Verbosity verbosity_;
    std::ostream& output_;

  public:
    /// <summary>
    /// Create a new ProgressStream.
    /// </summary>
    /// <param name="stream">The stream to which progress messages should be directed.</param>
	/// <param name="v">The maximimum verbosity of progress messages that will be directed to the output stream.</param>
    ProgressStream(std::ostream& stream, Verbosity v): verbosity_(v), output_(stream) { };

    ProgressStream operator[](Verbosity v)
    {
      return ProgressStream(output_, v>verbosity_? Silent : v);
    }

    template<typename T>
    ProgressStream& operator<<(const T& t)
    {
      if(verbosity_!=Silent)
        output_ << t;
      return *this;
    }

    ProgressStream& operator << (std::ostream & (*arg)(std::ostream &))
    {
      if(!verbosity_==Silent)
        output_ << arg;
      return *this;
    }
  };
} } }
