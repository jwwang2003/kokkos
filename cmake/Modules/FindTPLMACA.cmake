include(FindPackageHandleStandardArgs)

if(NOT MACA_PATH AND DEFINED ENV{MACA_PATH})
  set(MACA_PATH $ENV{MACA_PATH})
endif()
if(NOT MACA_PATH)
  set(MACA_PATH /opt/maca)
endif()

find_path(TPL_MACA_INCLUDE_DIRS
  NAMES mc/mc_runtime_api.h mcr/mc_runtime_api.h
  HINTS ${MACA_PATH}
  PATH_SUFFIXES include
)

find_library(TPL_MACA_RUNTIME_LIBRARIES
  NAMES mc_runtime mcruntime
  HINTS ${MACA_PATH}
  PATH_SUFFIXES lib lib64
)

set(_TPLMACA_MISSING_COMPONENTS "")
if(NOT TPL_MACA_INCLUDE_DIRS)
  list(APPEND _TPLMACA_MISSING_COMPONENTS
       "headers (mc/mc_runtime_api.h or mcr/mc_runtime_api.h)")
endif()
if(NOT TPL_MACA_RUNTIME_LIBRARIES)
  list(APPEND _TPLMACA_MISSING_COMPONENTS
       "runtime library (mc_runtime or mcruntime)")
endif()

if(_TPLMACA_MISSING_COMPONENTS)
  list(JOIN _TPLMACA_MISSING_COMPONENTS ", " _TPLMACA_MISSING_COMPONENTS)
  set(_TPLMACA_FAILURE_MESSAGE
      "Could not find a complete MACA SDK under MACA_PATH=${MACA_PATH}; missing ${_TPLMACA_MISSING_COMPONENTS}")
else()
  set(_TPLMACA_FAILURE_MESSAGE "")
endif()

if(_TPLMACA_FAILURE_MESSAGE)
  message(FATAL_ERROR "${_TPLMACA_FAILURE_MESSAGE}")
endif()

find_package_handle_standard_args(
  TPLMACA
  REQUIRED_VARS TPL_MACA_INCLUDE_DIRS TPL_MACA_RUNTIME_LIBRARIES
)

if(TPLMACA_FOUND)
  kokkos_create_imported_tpl(MACA INTERFACE
    LINK_LIBRARIES ${TPL_MACA_RUNTIME_LIBRARIES}
    INCLUDES ${TPL_MACA_INCLUDE_DIRS}
  )
endif()
