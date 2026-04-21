/*================================================================
*  Copyright (C)2026 All rights reserved.
*  FileName : maca_asm_exchange_op.inc
*  Author   : jwwang2003
*  Email    : wjw_03@outlook.com
*  Date     : Tue 21 Apr 2026 12:27:32 PM CST
================================================================*/

// The respective CUDA PTX implementation is kept & commented out (atom.exch / atom.cas snippets)
// around the active __maca_atomic_* calls. Those comments describe the PTX
// instruction family the MACA API maps to. The first-class MACA backend calls
// the API equivalents directly and omits the inactive asm spellings.

template <class T, class MemoryScope>
inline __device__
    typename std::enable_if<sizeof(T) == 4 || sizeof(T) == 8, T>::type
    device_atomic_exchange(T* const dest, T value, MemoryOrderRelaxed,
                           MemoryScope) {
  if (__isLocal(dest)) {
    T old = *dest;
    *dest = value;
    return old;
  }
  /*
  Reference commented asm in maca_cc7_asm_exchange_op.inc:
    atom.exch ...   // .b32 or .b64
  */
  return __maca_atomic_exchange(
      dest, value, MACAMemoryOrder<MemoryOrderRelaxed>::value,
      MACAMemoryScope<MemoryScope>::value);
}

template <class T, class MemoryScope>
inline __device__
    typename std::enable_if<sizeof(T) == 4 || sizeof(T) == 8, T>::type
    device_atomic_compare_exchange(T* const dest, T compare, T value,
                                   MemoryOrderRelaxed, MemoryScope) {
  if (__isLocal(dest)) {
    T old = *dest;
    if (old == compare) *dest = value;
    return old;
  }
  /*
  Reference commented asm in maca_cc7_asm_exchange_op.inc:
    atom.cas ...   // .b32 or .b64
  */
  T expected = compare;
  __maca_atomic_compare_exchange_strong(
      dest, &expected, value, MACAMemoryOrder<MemoryOrderRelaxed>::value,
      MACAMemoryOrder<cmpexch_failure_memory_order<MemoryOrderRelaxed>>::value,
      MACAMemoryScope<MemoryScope>::value);
  return expected;
}
