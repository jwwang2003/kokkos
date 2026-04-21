/*================================================================
*  Copyright (C)2026 All rights reserved.
*  FileName : maca_asm_atomic_fetch_op.inc
*  Author   : jwwang2003
*  Email    : wjw_03@outlook.com
*  Date     : Tue 21 Apr 2026 12:27:32 PM CST
================================================================*/

#ifdef DESUL_MACA_ATOMICS_USE_PREDICATE_PATH
#include "maca_asm_atomic_fetch_op.inc_predicate"
#else
#include "maca_asm_atomic_fetch_op.inc_isglobal"
#endif
