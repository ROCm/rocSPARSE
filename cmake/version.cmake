# ########################################################################
# Copyright (c) 2018 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

# TODO: move this function to https://github.com/RadeonOpenCompute/rocm-cmake/blob/master/share/rocm/cmake/ROCMSetupVersion.cmake

macro(rocm_set_parent VAR)
  set(${VAR} ${ARGN} PARENT_SCOPE)
  set(${VAR} ${ARGN})
endmacro()

function(rocm_get_git_commit_id OUTPUT_VERSION)
  set(options)
  set(oneValueArgs VERSION DIRECTORY)
  set(multiValueArgs)

  cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(_version ${PARSE_VERSION})

  set(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  if(PARSE_DIRECTORY)
    set(DIRECTORY ${PARSE_DIRECTORY})
  endif()

  find_program(GIT NAMES git)

  if(GIT)
    set(GIT_COMMAND ${GIT} describe --dirty --long --match [0-9]*)
    execute_process(COMMAND ${GIT_COMMAND}
      WORKING_DIRECTORY ${DIRECTORY}
      OUTPUT_VARIABLE GIT_TAG_VERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE RESULT
      ERROR_QUIET)
    if(${RESULT} EQUAL 0)
      set(_version ${GIT_TAG_VERSION})
    else()
      execute_process(COMMAND ${GIT_COMMAND} --always
	WORKING_DIRECTORY ${DIRECTORY}
	OUTPUT_VARIABLE GIT_TAG_VERSION
	OUTPUT_STRIP_TRAILING_WHITESPACE
	RESULT_VARIABLE RESULT
	ERROR_QUIET)
      if(${RESULT} EQUAL 0)
	set(_version ${GIT_TAG_VERSION})
      endif()
    endif()
  endif()
  rocm_set_parent(${OUTPUT_VERSION} ${_version})
endfunction()
