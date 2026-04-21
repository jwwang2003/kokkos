/*================================================================
*  Copyright (C)2026 All rights reserved.
*  FileName : maca_asm_memorder.inc
*  Author   : jwwang2003
*  Email    : wjw_03@outlook.com
*  Date     : Tue 21 Apr 2026 01:31:22 PM CST
================================================================*/

#define __DESUL_IMPL_MACA_MEMORY_ORDER MemoryOrderRelaxed
#include "maca_asm_atomic_fetch_op.inc"
#undef __DESUL_IMPL_MACA_MEMORY_ORDER

#define __DESUL_IMPL_MACA_MEMORY_ORDER MemoryOrderRelease
#include "maca_asm_atomic_fetch_op.inc"
#undef __DESUL_IMPL_MACA_MEMORY_ORDER

#define __DESUL_IMPL_MACA_MEMORY_ORDER MemoryOrderAcquire
#include "maca_asm_atomic_fetch_op.inc"
#undef __DESUL_IMPL_MACA_MEMORY_ORDER

#define __DESUL_IMPL_MACA_MEMORY_ORDER MemoryOrderAcqRel
#include "maca_asm_atomic_fetch_op.inc"
#undef __DESUL_IMPL_MACA_MEMORY_ORDER

#define __DESUL_IMPL_MACA_MEMORY_ORDER MemoryOrderSeqCst
#include "maca_asm_atomic_fetch_op.inc"
#undef __DESUL_IMPL_MACA_MEMORY_ORDER
