/*================================================================
*  Copyright (C)2026 All rights reserved.
*  FileName : maca_asm.inc
*  Author   : jwwang2003
*  Email    : wjw_03@outlook.com
*  Date     : Tue 21 Apr 2026 04:32:49 PM CST
================================================================*/

// Legacy MACA asm layering mirrors the CUDA cc7 include structure: this file
// owns the scope expansion and delegates per-order instantiation to
// maca_asm_memorder.inc.

#define __DESUL_IMPL_MACA_MEMORY_SCOPE MemoryScopeDevice
#include "desul/atomics/maca/maca_asm_memorder.inc"
#undef __DESUL_IMPL_MACA_MEMORY_SCOPE

#define __DESUL_IMPL_MACA_MEMORY_SCOPE MemoryScopeNode
#include "desul/atomics/maca/maca_asm_memorder.inc"
#undef __DESUL_IMPL_MACA_MEMORY_SCOPE

#define __DESUL_IMPL_MACA_MEMORY_SCOPE MemoryScopeCore
#include "desul/atomics/maca/maca_asm_memorder.inc"
#undef __DESUL_IMPL_MACA_MEMORY_SCOPE
