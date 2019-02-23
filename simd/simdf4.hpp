#pragma once

#if !defined(__ANDROID__)
#include "simdf4_sse.hpp"
#else
#include "simdf4_neon.hpp"
#endif
