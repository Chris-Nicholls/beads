/*
 * velvet_platform_beads.h — what velvet_reverb.c needs from the Beads board
 * (STM32H743, Cortex-M7 at 400 MHz) in place of the DLD's stm32f4xx.h.
 *
 * Every buffer is in internal SRAM, so there is no SDRAM to prefetch from:
 * the T2 gather and the pre-delay copies take the CPU path the host build
 * already uses (DMA2_DISABLED_FOR_DIAGNOSTIC).
 */
#pragma once

#include <stm32h7xx_hal.h>

#define DMA2_DISABLED_FOR_DIAGNOSTIC 1

/* The reverb puts two guarded statements on one line in a few places
 * ("if (v < 0) v = 0; if (v > 1) v = 1;"). Beads builds with -Werror. */
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
