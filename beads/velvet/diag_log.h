/*
 * diag_log.h — Beads stand-in for the DLD's FSK diagnostic log.
 *
 * Beads has no diagnostic link, so every event is dropped. The names match
 * the DLD header so velvet_reverb.c builds unchanged.
 */
#pragma once

#include <stdint.h>

typedef enum {
    DIAG_EVT_AUDIOISR_CH0,
    DIAG_EVT_AUDIOISR_CH1,
    DIAG_EVT_REVERB_BLOCK,
    DIAG_EVT_REVERB_DROP,
    DIAG_EVT_REVERB_T0,
    DIAG_EVT_REVERB_T2,
    DIAG_EVT_REVERB_T1,
    DIAG_EVT_REVERB_MORPH,
    DIAG_EVT_REVERB_FINALIZE,
    DIAG_EVT_REVERB_BG_EFF,
    DIAG_EVT_REVERB_PREDELAY,
    DIAG_EVT_REVERB_ISR_IN_BLOCK,
    DIAG_EVT_ISR_SDRAM_READ,
    DIAG_EVT_ISR_SDRAM_WRITE,
    DIAG_EVT_OUTPUT_MISS,
    DIAG_EVT_POLL_LATENCY,
    DIAG_EVT_COUNT
} diag_evt_t;

/* Codec-ISR cycles, used to subtract preemption from stage timings. Beads
 * does not count them, so stage timings are wall-clock. */
static volatile uint32_t diag_isr_cycles = 0;

static inline void diag_log(diag_evt_t evt, uint32_t cycles) { (void)evt; (void)cycles; }
#define DIAG_LOG_BLK(evt, cycles) do { (void)(evt); (void)(cycles); } while (0)
