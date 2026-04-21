/*================================================================
*  Copyright (C)2026 All rights reserved.
*  FileName : MACA_asm.hpp
*  Author   : jwwang2003
*  Email    : wjw_03@outlook.com
*  Date     : Tue 21 Apr 2026 12:27:32 PM CST
================================================================*/


#include <desul/atomics/Adapt_MACA.hpp>

#include<limits>
#include <type_traits>

namespace desul {
namespace Impl {
// Here we can choose the atomics to use depending on system architecture
// There might be differences between SDKs and device architectures
// NOTE: In the CUDA version, they have a bug patch what was handled here.
#include <desul/atomics/maca/maca_asm.inc>

}  // namespace Impl
}  // namespace desul
