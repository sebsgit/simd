#pragma once

#if !defined(__ANDROID__)
#include "simdu16x8_sse.hpp"
#else
#include "neon/simdu16x8_neon.hpp"
#endif
