// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_TEST_MACA_HPP
#define KOKKOS_TEST_MACA_HPP

#include <gtest/gtest.h>

#define TEST_CATEGORY maca
#define TEST_CATEGORY_NUMBER 10
#define TEST_CATEGORY_DEATH maca_DeathTest
#define TEST_EXECSPACE Kokkos::Maca
#define TEST_CATEGORY_FIXTURE(name) maca_##name

#endif
