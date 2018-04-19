#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "DGM" for configuration "Debug"
set_property(TARGET DGM APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(DGM PROPERTIES
  IMPORTED_LOCATION_DEBUG "/Users/creator/Projects/DGM/build/install/lib/libdgm160d.1.6.0.dylib"
  IMPORTED_SONAME_DEBUG "@rpath/libdgm160d.1.6.0.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS DGM )
list(APPEND _IMPORT_CHECK_FILES_FOR_DGM "/Users/creator/Projects/DGM/build/install/lib/libdgm160d.1.6.0.dylib" )

# Import target "FEX" for configuration "Debug"
set_property(TARGET FEX APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(FEX PROPERTIES
  IMPORTED_LOCATION_DEBUG "/Users/creator/Projects/DGM/build/install/lib/libfex160d.1.6.0.dylib"
  IMPORTED_SONAME_DEBUG "@rpath/libfex160d.1.6.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS FEX )
list(APPEND _IMPORT_CHECK_FILES_FOR_FEX "/Users/creator/Projects/DGM/build/install/lib/libfex160d.1.6.0.dylib" )

# Import target "VIS" for configuration "Debug"
set_property(TARGET VIS APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(VIS PROPERTIES
  IMPORTED_LOCATION_DEBUG "/Users/creator/Projects/DGM/build/install/lib/libvis160d.1.6.0.dylib"
  IMPORTED_SONAME_DEBUG "@rpath/libvis160d.1.6.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS VIS )
list(APPEND _IMPORT_CHECK_FILES_FOR_VIS "/Users/creator/Projects/DGM/build/install/lib/libvis160d.1.6.0.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
