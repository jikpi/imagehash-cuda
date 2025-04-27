find_path(FREEIMAGE_INCLUDE_DIR
        NAMES FreeImage.h
        PATHS
        /usr/include
        /usr/local/include
        /opt/local/include
        /sw/include
        ${FREEIMAGE_ROOT}/include
)

find_library(FREEIMAGE_LIBRARY
        NAMES freeimage FreeImage
        PATHS
        /usr/lib
        /usr/local/lib
        /opt/local/lib
        /sw/lib
        ${FREEIMAGE_ROOT}/lib
)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(FreeImage DEFAULT_MSG FREEIMAGE_LIBRARY FREEIMAGE_INCLUDE_DIR)

if(FREEIMAGE_FOUND)
    set(FREEIMAGE_LIBRARIES ${FREEIMAGE_LIBRARY})
    set(FREEIMAGE_INCLUDE_DIRS ${FREEIMAGE_INCLUDE_DIR})
endif()

mark_as_advanced(FREEIMAGE_INCLUDE_DIR FREEIMAGE_LIBRARY)