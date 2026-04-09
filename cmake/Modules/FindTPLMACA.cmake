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

if(TPL_MACA_RUNTIME_LIBRARIES)
  kokkos_create_imported_tpl(MACA INTERFACE
    LINK_LIBRARIES ${TPL_MACA_RUNTIME_LIBRARIES}
    INCLUDES ${TPL_MACA_INCLUDE_DIRS}
  )
else()
  kokkos_create_imported_tpl(MACA INTERFACE
    INCLUDES ${TPL_MACA_INCLUDE_DIRS}
  )
endif()
