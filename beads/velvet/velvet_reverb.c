/*
 * velvet_reverb.c — Velvet sparse-noise convolution reverb for DLD
 *
 * Three-stage cascade (T0 → T1 → T2) at REVERB_FS_HZ (24 or 48 kHz), processed
 * from the main loop in REVERB_BLOCK-sample blocks (= 32 codec samples, ~667 µs).
 *
 *   codec → push (avg 2:1 decimate at 24 kHz) → pre-delay sustain engine (2 modulated,
 *           damped feedback delay lines A/B read through allpass fractional
 *           delays, shared shelves, input duck, mix)
 *             ↓ (feedforward into the cascade)
 *           T0 ring (CCM)
 *           T0 sparse conv  (+ optional per-tap Lexicon LFO on the readout)
 *             ↓ saturate
 *           T1 ring (CCM)
 *           T1 sparse conv
 *             ↓ saturate
 *           T2 ring (SDRAM, DMA-prefetched)
 *           T2 sparse conv (stereo via independent L/R offsets, shared gain)
 *             ↓ saturate (+ 2× linear-interp upsample at 24 kHz)
 *           HPF + LPF biquads @ 48 kHz (at 24 kHz the LPF doubles as the
 *           reconstruction filter)
 *             ↓
 *           codec output
 *
 * Dynamic-parameter infrastructure:
 *
 *   - Every stage generates MAX taps once at init, at the DLD's density across
 *     the stage's longest span (sign + 1/sqrt(N)), and each tap gets a
 *     removal rank from its spacing. The Density macro sets the share of the
 *     taps kept, so taps per second; taps whose rank is past the (smoothed)
 *     count fade out one at a time, and the stage's loudness is held by an
 *     energy compensation. See compute_tap_ranks().
 *   - Window sizes are mutable globals smoothed per-block with a one-pole IIR
 *     (driven by the Size macro). The taps keep their places: Size sets how
 *     much of the span sounds, each tap fading in as the window reaches it.
 *   - Each block we recompute the *effective* per-tap gain (int16 Q15):
 *         base × density-fade × window-fade × [T2 exp envelope] × gain-comp
 *     and the convolution inner loops read from these effGains arrays, so
 *     window/envelope/loudness all change smoothly without regenerating taps.
 *   - The effGains recompute is round-robin across the three stages (one
 *     per block) to keep the per-block update cost flat.
 *   - The long tail / sustain comes from the pre-delay feedback lines in
 *     front of the cascade (the cascade itself is purely feedforward), which
 *     replaces the old per-stage + global recirculation.
 *
 * KEY INVARIANT: input_buf is ALWAYS written linearly 0..REVERB_BLOCK-1,
 * ensuring chronological order when committed to the T0 ring.
 */

#ifdef VELVET_REVERB_HOST
#include "host_shim.h"
#elif defined(VELVET_REVERB_BEADS)
#include "velvet_platform_beads.h"
#else
#include "stm32f4xx.h"
#endif
#include "velvet_reverb.h"
#include "wet_fx.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef VELVET_REVERB_HOST
#include "diag_log.h"
#endif
#if defined(DIAG_REVERB_PROFILE) && !defined(VELVET_REVERB_HOST)
#include "adc.h"                /* adc_overrun_count, for the profile report */
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define TWO_PI_F  6.28318530717958647692f
#define PI_F      3.14159265358979323846f
#define PI_2_F    1.57079632679489661923f

#ifdef VELVET_REVERB_HOST
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
__attribute__((always_inline)) static inline int32_t qadd_sat(int32_t a, int32_t b)
{
    int64_t r = (int64_t)a + (int64_t)b;
    if (r > INT32_MAX) return INT32_MAX;
    if (r < INT32_MIN) return INT32_MIN;
    return (int32_t)r;
}
__attribute__((always_inline)) static inline int32_t smulbb(uint32_t a, uint32_t b)
{
    return (int32_t)(int16_t)(a & 0xFFFFu) * (int32_t)(int16_t)(b & 0xFFFFu);
}
__attribute__((always_inline)) static inline int32_t smultb(uint32_t a, uint32_t b)
{
    return (int32_t)(int16_t)(a >> 16) * (int32_t)(int16_t)(b & 0xFFFFu);
}
/* Dual 16x16 multiply-accumulate: acc + a.lo*b.lo + a.hi*b.hi (no saturation). */
__attribute__((always_inline)) static inline int32_t smlad(uint32_t a, uint32_t b, int32_t acc)
{
    return acc + (int32_t)(int16_t)(a & 0xFFFFu) * (int32_t)(int16_t)(b & 0xFFFFu)
               + (int32_t)(int16_t)(a >> 16)     * (int32_t)(int16_t)(b >> 16);
}
/* Dual 16-bit add, each half wrapping on its own. */
__attribute__((always_inline)) static inline uint32_t sadd16(uint32_t a, uint32_t b)
{
    return ((a + b) & 0xFFFFu) | (((a >> 16) + (b >> 16)) << 16);
}
#else
__attribute__((always_inline)) static inline int32_t qadd_sat(int32_t a, int32_t b)
{
    int32_t r;
    __asm volatile ("qadd %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
__attribute__((always_inline)) static inline int32_t smulbb(uint32_t a, uint32_t b)
{
    int32_t r;
    __asm volatile ("smulbb %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
__attribute__((always_inline)) static inline int32_t smultb(uint32_t a, uint32_t b)
{
    int32_t r;
    __asm volatile ("smultb %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
/* Dual 16x16 multiply-accumulate: acc + a.lo*b.lo + a.hi*b.hi (single cycle). */
__attribute__((always_inline)) static inline int32_t smlad(uint32_t a, uint32_t b, int32_t acc)
{
    int32_t r;
    __asm volatile ("smlad %0, %1, %2, %3" : "=r"(r) : "r"(a), "r"(b), "r"(acc));
    return r;
}
/* Dual 16-bit add, each half wrapping on its own. */
__attribute__((always_inline)) static inline uint32_t sadd16(uint32_t a, uint32_t b)
{
    uint32_t r;
    __asm volatile ("sadd16 %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
#endif

/* ==== CCM placement ==== */
#if defined(VELVET_REVERB_HOST) || defined(VELVET_REVERB_BEADS)
/* Beads: no CCM; .bss is DTCM, which is just as fast. */
#define CCM_ATTR __attribute__((aligned(4)))
#else
#define CCM_ATTR __attribute__((aligned(4), section(".ccmdata")))
#endif

/* May-alias typedef for fused 32-bit reads/writes on int16_t buffers.
 * Without this, casting (uint32_t *)<int16_t *> violates strict aliasing —
 * GCC may reorder the read/write w.r.t. int16_t accesses to the same memory,
 * producing intermittent wrong-data bugs. With __attribute__((may_alias)),
 * GCC knows this type can alias any other and won't optimize across it. */
typedef uint32_t u32_alias __attribute__((may_alias));

/* ==== Rings ====
 *
 * Both cascade rings carry a REVERB_BLOCK-sample guard region past the nominal
 * end that mirrors samples [0, REVERB_BLOCK). Every tap reads a window of
 * REVERB_BLOCK (T1) or REVERB_BLOCK+1 (T0, which pair-reads for its 2-tap
 * interpolation) samples starting at an arbitrary masked position, so the last
 * such window begins at RING_SAMPLES-1 and needs indices up to
 * RING_SAMPLES+REVERB_BLOCK-1. With the mirror in place those reads are always
 * contiguous and always in bounds, which buys three things: the per-tap wrap
 * test and its segmented slow path disappear from the hot loop, the accumulator
 * loops become a single fixed-count run the compiler can unroll, and a latent
 * one-element overrun goes away — T1's pair loop advanced k by two up to an
 * `until_wrap` that can be odd, reading one int16 past the segment end.
 *
 * The mirror is refreshed unconditionally once per block (see ring_guard_sync),
 * which is ~16 halfword copies against the ~1000 tap reads it protects.
 *
 * At 48 kHz T1's ring is 64 KB, which is the whole of CCM, so it lives in main
 * SRAM instead. Both are zero-wait-state for the CPU. */
#if REVERB_FS_HZ > 24000
#define T1_RING_ATTR __attribute__((aligned(4)))
#else
#define T1_RING_ATTR CCM_ATTR
#endif
static int16_t t0_ring[T0_RING_SAMPLES + REVERB_BLOCK] CCM_ATTR;
static int16_t t1_ring[T1_RING_SAMPLES + REVERB_BLOCK] T1_RING_ATTR;
static int16_t * const t2_ring = (int16_t *)T2_RING_BASE;
/* The right channel's rings. Same taps as the left, so a mono input gives the
 * same result on both sides up to T2, whose L and R taps differ. On the Beads
 * T0's and T1's do not fit beside the left's in DTCM, so they sit in ITCM
 * (zero-wait for data loads too), behind a pad so neither starts at address 0.
 * The startup code does not fill ITCM; velvet_reverb_init clears them. */
#if defined(VELVET_REVERB_BEADS) && !defined(VELVET_REVERB_HOST)
static struct {
    uint32_t pad[8];
    int16_t t0[T0_RING_SAMPLES + REVERB_BLOCK];
    int16_t t1[T1_RING_SAMPLES + REVERB_BLOCK];
} itcm_rings __attribute__((section(".itcm"), aligned(4)));
#define t0_ring_r itcm_rings.t0
#define t1_ring_r itcm_rings.t1
#else
static int16_t t0_ring_r[T0_RING_SAMPLES + REVERB_BLOCK] __attribute__((aligned(4)));
static int16_t t1_ring_r[T1_RING_SAMPLES + REVERB_BLOCK] __attribute__((aligned(4)));
#endif
static int16_t * const t2_ring_r = (int16_t *)T2_RING_R_BASE;

/* Mirror a ring's first REVERB_BLOCK samples into its guard region. Must run
 * after the block write and before any tap reads that ring. */
__attribute__((always_inline))
static inline void ring_guard_sync(int16_t *ring, uint32_t nsamples)
{
    for (uint32_t k = 0; k < (uint32_t)REVERB_BLOCK; k++) ring[nsamples + k] = ring[k];
}

static uint32_t ring_write_idx = 0;
/* Wrap modulus for ring_write_idx/block_write_idx (see do_input_write_t0). 2^23 is
 * a multiple of every ring size (T0=2^13, T1=2^14, T2=2^18) so masked positions are
 * continuous across the wrap, and < 2^24 so the float cast in the recirc math is exact. */
#define REVERB_IDX_WRAP  0x800000UL   /* 2^23 = 8388608 */

/* ==== Tap arrays (generated once at MAX count + MAX window) ====
 * Density is fixed at MAX, so the per-tap removal-rank arrays are gone. */
uint32_t t0TapOffsets[MAX_T0_TAPS];
int16_t  t0TapGains [MAX_T0_TAPS];
int      t0TapCount = 0;

uint32_t t1TapOffsets[MAX_T1_TAPS];
int16_t  t1TapGains [MAX_T1_TAPS];
/* T1's onset weighting (see recompute_eff_gains_t1), set with the taps. */
#define T1_ONSET_GAIN   2.0f    /* extra gain at T1's first tap */
#define T1_ONSET_MS     25.0f   /* falling away with this time constant */
static float t1OnsetBoost[MAX_T1_TAPS];
int      t1TapCount = 0;

uint32_t t2TapOffsetsL[MAX_T2_TAPS];
uint32_t t2TapOffsetsR[MAX_T2_TAPS];
int16_t  t2TapGains   [MAX_T2_TAPS];
int      t2TapCount = 0;

/* ==== Per-tap position ladders ====
 * Room for up to MAX_LADDER_LEVELS read positions per tap, which
 * compute_tap_states walks as the window changes. Each tap now has a single
 * rung, its own position (see build_single_rung_ladders), so a tap sounds once
 * the window reaches it and is silent before. Offsets are uint32_t: T2's run
 * up to T2_DURATION_MAX_SAMPLES (78000). */
static uint32_t t0TapLadder [MAX_T0_TAPS * MAX_LADDER_LEVELS] CCM_ATTR;
static uint8_t  t0TapLadderCount[MAX_T0_TAPS]                 CCM_ATTR;
static uint32_t t1TapLadder [MAX_T1_TAPS * MAX_LADDER_LEVELS] CCM_ATTR;
static uint8_t  t1TapLadderCount[MAX_T1_TAPS]                 CCM_ATTR;
static uint32_t t2TapLadderL[MAX_T2_TAPS * MAX_LADDER_LEVELS] CCM_ATTR;
static uint32_t t2TapLadderR[MAX_T2_TAPS * MAX_LADDER_LEVELS] CCM_ATTR;
static uint8_t  t2TapLadderCount[MAX_T2_TAPS]                 CCM_ATTR;

/* ==== Per-block effective read offsets ====
 * Updated each block by compute_tap_states based on the current window —
 * each tap is at whichever ladder rung's band contains the window. The
 * convolution inner loops read from these instead of from tNTapOffsets. */
static uint32_t effOffT0 [MAX_T0_TAPS] CCM_ATTR;
static uint32_t effOffT1 [MAX_T1_TAPS] CCM_ATTR;
static uint32_t effOffT2L[MAX_T2_TAPS] CCM_ATTR;
static uint32_t effOffT2R[MAX_T2_TAPS] CCM_ATTR;

/* ==== Effective per-tap gains (rebuilt per block, round-robin across stages) ==== */
static int16_t effGains_t0[MAX_T0_TAPS] CCM_ATTR;
static int16_t effGains_t1[MAX_T1_TAPS] CCM_ATTR;
static int16_t effGains_t2[MAX_T2_TAPS] CCM_ATTR;

/* ==== Stereo-width prototype (host A/B; 0 = the shipped sound) ====
 * T2 is the only stereo stage, and by default its L and R taps share one gain
 * and sign each. Tap 0 carries the most envelope gain; R's sits a few ms after
 * L's (REVERB_T2_R_TAP0_MS) so the first ~100 ms of a tail isn't mono.
 *   REVERB_T2_R_SIGNS 1: R's taps 1.. get their own signs (tap 0 keeps the
 *     shared one, so the earliest reflection stays mono-safe). Drawn from a
 *     separate generator, so the tap layout is unchanged. */
#ifndef REVERB_T2_R_SIGNS
#define REVERB_T2_R_SIGNS        0
#endif
/* R's tap 0 sits this far (ms) after L's, so the first reflection after a hit
 * arrives at a different time on each side and the onset is already wide
 * rather than ~100 ms of mono. Short enough to stay one early reflection. */
#ifndef REVERB_T2_R_TAP0_MS
#define REVERB_T2_R_TAP0_MS      3.0f
#endif
/* R's own gains: below the full Density set, R keeps its own share of the
 * taps (see recompute_eff_gains_t2). */
static int16_t effGains_t2R[MAX_T2_TAPS] CCM_ATTR;
#define T2_GAINS_R effGains_t2R
#if REVERB_T2_R_SIGNS
static uint8_t t2FlipR[MAX_T2_TAPS];
#endif
#ifdef VELVET_REVERB_HOST
static float host_effwin_t2[MAX_T2_TAPS];  /* relocation/window gain only (no decay env) */
#endif

/* The old precomputed 2D energy LUT and per-tap env LUT have been replaced
 * by per-block ladder-based recompute (see compute_tap_states + recompute
 * functions below). The ladder makes density approximately constant across
 * the window range, so the per-stage energy is dominated by the per-tap
 * effective window-gain — fast to sum on the fly, no 2D table required. */

/* ==== Mutable globals — driven by the macro system (see below). Initial
 *      values are what the macros produce at their default (D=0, Dec=0,
 *      Tone=0.83) positions; will be overwritten on the first macro call. */
float reverb_send      = 0.7f;   /* updated each params iter from right MIX pot */
float reverb_dry_gain  = 1.0f;   /* fades to 0 over top 10% of right MIX pot */
/* Defaults below match the user's velvet_buffers slider snapshot. Macro-driven
 * params (windows, feedback, mix, lpf/hpf, high-shelf Hz) are recomputed from
 * the Decay/Tone pots at init. */
float reverb_lpf_hz    = 8600.0f;   /* Tone → MP_LPF */
float reverb_hpf_hz    = 170.0f;    /* Tone → MP_HPF */

/* ==== Output tilt ====
 *   REVERB_OUTPUT_TILT 1: while reverb_tilt_on, the output's TONE high-pass
 *     is replaced by a first-order tilt round reverb_tilt_hz: reverb_tilt_db
 *     is the lift of the highs over the lows (half up, half down), so 0 is
 *     flat. It takes the high-pass's biquad slot, so the output loop is
 *     unchanged; with reverb_tilt_on 0 the output has TONE's high-pass. The
 *     defaults are the settings chosen on the A/B page. */
#ifndef REVERB_OUTPUT_TILT
#define REVERB_OUTPUT_TILT 1
#endif
#if REVERB_OUTPUT_TILT
int   reverb_tilt_on = 1;
float reverb_tilt_db = 3.0f;
float reverb_tilt_hz = 930.0f;
#endif

/* ==== Decay-matched loop shelves ====
 *   REVERB_LOOP_T60_SHELVES 1: while reverb_t60_shelves_on, the loop's shelf
 *     gains come from decay-time ratios instead of fixed dB per trip. A line
 *     loses 20*log10(feedback) dB per trip in the mids; for its highs to ring
 *     r times as long they must lose 1/r as much, so the high shelf is
 *     20*log10(feedback) * (1/r - 1). Both lines share the feedback, so they
 *     share the shelf too. r for the highs runs from reverb_t60_hi_ratio_min at
 *     Tone 0 to reverb_t60_hi_ratio_max at Tone 1 (exponential); the lows use
 *     reverb_t60_lo_ratio. Corners are unchanged (low fixed, high from Tone);
 *     r above 1 is a boost, still stable since the loop gain stays below 1.
 *     The shelf gains are worked out with the macros (velvet_reverb_recompute_
 *     macros), not per block. While on, reverb_fb_low/high_shelf_db are unused. */
#ifndef REVERB_LOOP_T60_SHELVES
#define REVERB_LOOP_T60_SHELVES 1
#endif
#if REVERB_LOOP_T60_SHELVES
int   reverb_t60_shelves_on   = 1;
float reverb_t60_hi_ratio_min = 0.10f;
float reverb_t60_hi_ratio_max = 0.53f;
float reverb_t60_lo_ratio     = 0.49f;
static float t60_hs_db, t60_ls_db;   /* the shelf gains they give, dB per trip */
#endif

/* ==== Loop decay curve ====
 *   REVERB_LOOP_FB_CURVE 1: while reverb_loop_curve_on, the sustain loop's
 *     feedback is capped so the loop's tail is short where DECAY first mixes
 *     it in, instead of a long quiet tail behind the cascade's ~1 s one. The
 *     cap gives line B (the longer line, which sets the tail) a T60 of
 *     reverb_loop_t60_lo seconds at DECAY reverb_loop_curve_lo and below,
 *     rising exponentially to meet the shipped feedback at
 *     reverb_loop_curve_join; from there up the feedback is the shipped one.
 *     The shelves add their own loss on top, so the tail is a little shorter
 *     than the cap's T60. The DECAY points are macro values (the pot scaled by
 *     1/DECAY_MACRO_TOP_SAT): the join, 0.75, is the pot at 0.72.
 *     On Beads, TIME is the loop's feedback alone, so the curve starts at the
 *     bottom of the knob: from a 0.25 s loop T60 at TIME 0 up to the join. */
#ifndef REVERB_LOOP_FB_CURVE
#define REVERB_LOOP_FB_CURVE 1
#endif
#if REVERB_LOOP_FB_CURVE
int   reverb_loop_curve_on   = 1;
float reverb_loop_t60_lo     = 0.25f;
float reverb_loop_curve_lo   = 0.0f;
float reverb_loop_curve_join = 0.75f;
#endif

/* T2's fade length (Size): T2's taps fall 80 dB over this length, and those
 * past it are silent, so at small sizes the tail fades before the window
 * ends. From Size's midpoint up it equals the T2 window. */
float reverb_t2_fade_s_target = 3.217f;
float reverb_t0_window_ms_target  = 69.0f;     /* driven by Size  */
float reverb_t1_window_ms_target  = 536.0f;    /* driven by Size  */
float reverb_t2_duration_s_target = 3.217f;    /* driven by Size  */

/* ==== Pre-delay sustain engine params ==== Macro-driven: feedback, delay_mix,
 * high-shelf Hz. Fixed (no macro maps them): delay times, low-shelf, high-shelf
 * dB, fb-mod, duck. All values from the user's slider snapshot. */
float reverb_feedback         = 0.969f;  /* driven by Decay */
/* The two loops' lengths are in the golden ratio, the most irrational of
 * ratios, so their echoes never fall back into step. (The DLD's 0.185 / 0.3 s
 * is 60/37, within 0.2% of it, but rational: the two lined up again every
 * 11.1 s, 37 passes of one against 60 of the other.) B is 299.34 ms, not a
 * whole number of samples, so even the sample grid never repeats them. */
#define LOOP_GOLDEN_RATIO  1.6180340f
float reverb_predelay_a_s      = 0.185f;
float reverb_predelay_b_s      = 0.185f * LOOP_GOLDEN_RATIO;
float reverb_delay_mix         = 0.68f;   /* fixed: the DLD's at the default Decay */
float reverb_fb_low_shelf_hz   = 124.0f;  /* fixed */
float reverb_fb_low_shelf_db   = -0.6f;   /* fixed; unused with decay-matched shelves */
float reverb_fb_high_shelf_hz  = 2000.0f; /* driven by Tone */
float reverb_fb_high_shelf_db  = -0.25f;  /* fixed; unused with decay-matched shelves */
float reverb_fb_mod_rate       = 0.35f;   /* fixed */
float reverb_duck_amount       = 1.0f;    /* fixed */
/* Duck release, expressed in LOOP TRAVERSALS rather than seconds.
 *
 * The duck attenuates whatever passes the pre-delay write node, and buffer
 * content passes that node once per loop period — so the loop length, not an
 * absolute time, is the natural unit. At 1.0 everything already circulating is
 * attenuated exactly once and the duck is then out of the way, which is what
 * makes an arriving sound overwrite the buffer rather than pile onto it.
 * Beyond ~1.5 the duck is still engaged while the NEW sound is recirculating,
 * so it starts erasing the very thing it just wrote: measured at -6 dBFS,
 * going 1x -> 2x buys 1.4 dB more erasure of the old sound but costs 1.5 s of
 * the new sound's tail, and by 3x both are worse.
 *
 * Deriving this from the line lengths (rather than the old fixed 2.0 s) also
 * means it keeps tracking if the pre-delay times are ever changed. */
float reverb_duck_release_loops = 1.0f;   /* fixed */

/* T0 per-tap Lexicon LFO (early-reflection smearer; replaces recirc mod). */
float reverb_t0_tap_mod_rate   = 0.59f;

/* Modulation amount (velvet_reverb_set_modulation), 0..1: how far the
 * reverb's delays move, T0's taps and the sustain loops' reads together.
 * Any delay movement, passed through the long velvet stages, turns a steady
 * tone into level swings (a sine wobbled 10-18 dB at the depths that used to
 * be fixed), so less is steadier; but with the loops' reads still, each pass
 * round loop B came back the same and a full, long tail pulsed at its period
 * (3.3 Hz). The depths go with the square of a share (mod_depth_share) that
 * runs from 0.2 at 0, a minimum set by ear, to 0.5 at MOD_AMOUNT_DEFAULT, the
 * middle (about the old fixed depths: T0's swing a quarter of its full ~5 ms,
 * the loops' 0.04 of PRE_MOD_MAX_SAMPLES), and on to 1 (T0's full swing, the
 * loops' MOD_FB_DEPTH_MAX). It glides
 * (MOD_AMOUNT_ALPHA per block, ~0.25 s) so the read heads never jump, landing
 * exactly, and holds while frozen so a frozen loop keeps its exact copy. */
#define MOD_FB_DEPTH_MAX    0.16f
#define MOD_AMOUNT_ALPHA    0.002f
static float mod_amount_target = MOD_AMOUNT_DEFAULT, mod_amount_morph = MOD_AMOUNT_DEFAULT;

/* The share of the full depths the amount gives, before squaring. */
static inline float mod_depth_share(float amount)
{
    return amount < 0.5f ? 0.2f + 0.6f * amount : amount;
}

void velvet_reverb_set_modulation(float amount)
{
    if (amount < 0.0f) amount = 0.0f; else if (amount > 1.0f) amount = 1.0f;
    mod_amount_target = amount;
}

#define TAP_COMP_CLAMP        3.0f       /* max gain-comp boost (avoid int16 overflow) */
#define T2_COMP_AT_MID        2.0f       /* T2's comp at Size's midpoint: the old clamp it settled at */

/* ~12 Hz DC-block one-pole in each pre-delay feedback loop. */
#define PRE_HP_FREQ_HZ        12.0f
/* Fixed ~2 ms duck attack. */
#define DUCK_ATK_SEC          0.002f
/* Minimum pre-delay read time (5 ms) so reads stay in the past. */
#define PRE_MIN_TIME_SAMPLES  (0.005f * (float)REVERB_FS_HZ)
/* depth = 1 → ~5 ms T0 per-tap swing. */
#define T0_TAP_MOD_MAX_SAMPLES  (0.005f * (float)REVERB_FS_HZ)
#define GOLDEN_ANGLE_RAD      2.39996322972865332f

/* Mid-side stereo width applied to the final T2 output in do_finalize.
 * 1.0 = neutral, <1 narrows toward mono, >1 widens by boosting the
 * decorrelated (L-R) component. Only T2 is stereo (T0/T1 are mono), so this
 * directly scales that decorrelation. With R's tap 0 offset (REVERB_T2_R_TAP0_MS)
 * L and R are already uncorrelated, so 1.0 is the widest that stays in phase. */
#ifndef REVERB_STEREO_WIDTH            /* overridable so the harness can sweep it */
#define REVERB_STEREO_WIDTH   1.0f
#endif

/* Internal headroom. The whole int16 cascade runs at REVERB_HEADROOM × level
 * (cooler), and the output is scaled back up by REVERB_OUT_MAKEUP = 1/headroom.
 * Inter-stage values that would otherwise push past int16 full-scale (conv-sum
 * peaks, recirc feedback, etc.) now sit below it, so the soft knee (below)
 * rarely engages and the path stays transparent until a true over. Cost: the internal computation runs ~6 dB closer to the int16 LSB at
 * 0.5, so the quietest tail loses ~1 bit of SNR. Raise toward 1.0 if the decay
 * sounds grainy; lower toward 0.5 if clipping persists. */
/* On the Beads the reverb takes the codec input raw, with no input gain
 * control, so a held, bright modular signal at full send overran the stage
 * bridges' soft knees (thousands of hits a second at 0.5 of full scale with
 * the sustain loop full), heard as grit and pops that grew with Size. At
 * 0.25 a full-scale sawtooth at full Size passes every stage below the knee;
 * the output level is unchanged (the makeup follows). */
#ifndef REVERB_HEADROOM                /* overridable so the harness can sweep it */
#define REVERB_HEADROOM    0.25f
#endif

/* Soft-knee threshold (full-scale units) for the inter-stage / output saturators.
 * |x| <= threshold passes through linearly (transparent); above it a
 * C1 rational knee rounds off toward ±1.0 instead of a hard int16 clamp. With the
 * headroom above, signals normally stay under the threshold; the knee only shapes
 * genuine overs. Declared here rather than further down because the host
 * instrumentation needs it to count knee entry. Overridable so the harness can A/B
 * the knee out of the path. */
#ifndef SOFT_KNEE_T
#define SOFT_KNEE_T  0.80f
#endif
#define REVERB_OUT_MAKEUP  (1.0f / REVERB_HEADROOM)

#ifdef VELVET_REVERB_HOST
/* Host-only drive instrumentation, read by the host harness. Not compiled into
 * the firmware. */
float    host_dbg_t0_in_peak   = 0.0f;
uint32_t host_dbg_t0_sat_count = 0;

/* Every other clipping point in the cascade, so a test can say WHICH stage is
 * being driven into its knee rather than only the first one. Peaks are expressed
 * as a fraction of full scale, so 1.0 is exactly at the limit. This matters
 * because a soft knee is silent about how hard it is working, and clipping high
 * frequencies at 24 kHz throws harmonics past Nyquist that fold back as
 * broadband hash — which sounds like noise rather than like distortion. */
float    host_dbg_t01_peak = 0.0f;  uint32_t host_dbg_t01_sat = 0;  /* accT0 -> t1_ring */
float    host_dbg_t12_peak = 0.0f;  uint32_t host_dbg_t12_sat = 0;  /* accT1 -> t2_ring */
float    host_dbg_out_peak = 0.0f;  uint32_t host_dbg_out_sat = 0;  /* accL/R -> output  */

/* The final int16 conversion, measured as a DISTRIBUTION rather than a peak. A
 * peak says only that the ceiling was touched once; what decides the fix is how
 * much of the time it is being touched. A handful of samples per second can be
 * dealt with by a limiter at no cost in loudness, whereas a few percent means the
 * wet signal is simply louder than the output can carry. */
float    host_dbg_final_peak  = 0.0f;
uint64_t host_dbg_final_n     = 0;   /* samples converted        */
uint64_t host_dbg_final_knee  = 0;   /* ... above the knee       */
uint64_t host_dbg_final_over  = 0;   /* ... at or past full scale */
double   host_dbg_final_sumsq = 0.0;

/* Q30 full scale, matching soft_saturate_q15. */
#define HOST_Q30_FS 1073741824.0f
/* Counts entry into the KNEE, not overs. soft_clip_unit is transparent only up to
 * SOFT_KNEE_T (0.80), so a peak of 0.98 is already being shaped even though it
 * never reaches full scale — counting against 1.0 reports zero while the stage is
 * distorting steadily. */
static inline void host_note_sat(int32_t x, float *peak, uint32_t *cnt)
{
    float m = (float)x * (1.0f / HOST_Q30_FS);
    if (m < 0.0f) m = -m;
    if (m > *peak) *peak = m;
    if (m > SOFT_KNEE_T) (*cnt)++;
}

/* The ±4 hard clamp on the pre-delay feedback node. A hard clamp is a level
 * dependent nonlinearity that engages only on peaks, which is the shape of the
 * remaining artifact: sporadic, worse with louder input, and present in a
 * sustained tail. Counted so a test can say whether it fires at all. */
float    host_dbg_node_peak    = 0.0f;
uint32_t host_dbg_node_clamps  = 0;

/* effGains recompute accounting. The count alone says little — what clicks is an
 * ISOLATED recompute, one that lands after a long gap and so has to snap the tap
 * gains across all the morph drift accumulated in that gap. A recompute that
 * follows hard on the heels of another only moves the gains a little. */
uint32_t host_dbg_bg_blocks    = 0;   /* blocks background_eff_gains_update saw  */
uint32_t host_dbg_eff_calls    = 0;   /* recomputes performed                    */
uint32_t host_dbg_eff_isolated = 0;   /* recomputes >10 blocks after the last    */
uint32_t host_dbg_eff_maxgap   = 0;   /* largest gap before a recompute (blocks) */
uint32_t host_dbg_eff_lastblk  = 0;   /* exported so a test can reset the run     */

/* T2 gather: blocks served by the rows fetched at the end of the previous block
 * (early), by the second half of rows fetched a block before that (reuse), or by
 * a fetch at T2 time (late: startup, or an offset that changed in between). A
 * test asserting bit-exactness must also see early and reuse dominate, or it is
 * only testing the fallback. */
uint32_t host_dbg_t2_gather_early = 0;
uint32_t host_dbg_t2_gather_reuse = 0;
uint32_t host_dbg_t2_gather_late  = 0;
#endif


/* ==== Macro system ====
 * Four macros (Decay, Tone, Density, Size), each 0..1. Decay is the sustain
 * loop's feedback alone (with the loop curve below, from a 0.25 s loop T60). Size is the
 * space: the stage windows, which set both the tap spacing and the T1/T2 tap
 * fades, and the sustain loops' lengths. Every mapped param uses the full macro
 * range — macro value u lerps bound_min..bound_max (exp where both > 0). */
/* Density: tap count per stage, from DENSITY_Tx_MIN to the full MAX_Tx_TAPS.
 * Turning down from the top, the stages thin one after another: T0 over the
 * top third of the knob, then T1 over the middle third, then T2 over the
 * bottom third, each exponentially in its own third. So the early taps go
 * first, then the middle stage's, and the long last stage, which carries the
 * tail, thins last. The cascade is serial, so a stage needs one tap or the
 * reverb goes silent; at one tap in every stage the cascade is a single
 * path, and what is left is the sustain engine's loops (echoes, not a
 * wash). Counts are fractional: the tap at the boundary is part-faded. */
/* The fewest taps Density keeps, as a share of each stage's whole span: the
 * DLD's one tap in its 40 (32 for T2) at the midpoint window, so the sparse
 * end keeps the DLD's taps per second too. T2's span is 3.25 s rather than
 * the 8 s that rate was set over, so its floor is scaled to match: the sparse
 * end is unchanged and only the dense end rises (to ~2.5x). */
#define DENSITY_T0_MIN  ((float)MAX_T0_TAPS / 40.0f)
#define DENSITY_T1_MIN  ((float)MAX_T1_TAPS / 40.0f)
#define DENSITY_T2_MIN  ((float)MAX_T2_TAPS / 32.0f * T2_SPAN_MAX_S / 8.0f)
#define DENSITY_THIN_U  0.25f     /* the bottom of the knob that thins the loop (see loop_thin_target) */
float reverb_t0_count_target = (float)MAX_T0_TAPS;   /* driven by Density */
float reverb_t1_count_target = (float)MAX_T1_TAPS;
float reverb_t2_count_target = (float)MAX_T2_TAPS;

typedef enum {
    MP_T0_WINDOW,
    MP_T1_WINDOW,
    MP_T2_DURATION,
    MP_FEEDBACK,
    MP_LPF,
    MP_HPF,
    MP_FB_HIGH_SHELF_HZ,
    MP_T0_COUNT,
    MP_T1_COUNT,
    MP_T2_COUNT,
    MP_T2_FADE,
    MP_COUNT
} macro_param_id_t;

typedef struct {
    float *target;
    float bound_min;
    float bound_max;
    uint8_t use_exp;       /* 1 = exp lerp when bounds both > 0; else linear */
    float bound_mid;       /* > 0: exp through min, mid (at SIZE_MID_U) and max */
} macro_param_t;

typedef struct {
    float value;
    int   num_params;
    const macro_param_id_t *params;
} macro_t;

/* Size's curves pass through the DLD's spacing (the defaults above, and loop
 * scale 1) at SIZE_MID_U, a little past noon, so the default sound is the
 * DLD's and the knob reaches far either side of it: down to a small, tight
 * space (short loops, short T2 fade; the windows only go down to their old
 * minimums, see reverb_t2_fade_s_target), up to the longest windows the
 * rings hold. */
#define SIZE_MID_U  0.6f

/* Unmapped globals (predelay times, shelf dB/freq-low, duck, fb-mod) keep their
 * fixed defaults. */
static macro_param_t reverb_macro_params[MP_COUNT] = {
    [MP_T0_WINDOW]        = { &reverb_t0_window_ms_target,   37.0f,   200.0f,  1,  69.0f  },
    [MP_T1_WINDOW]        = { &reverb_t1_window_ms_target,  230.0f,   660.0f,  1, 536.0f  },
    [MP_T2_DURATION]      = { &reverb_t2_duration_s_target,   1.6f, T2_SPAN_MAX_S, 1, 3.217f },
    [MP_T2_FADE]          = { &reverb_t2_fade_s_target,       0.9f, T2_SPAN_MAX_S, 1, 3.217f },
    [MP_FEEDBACK]         = { &reverb_feedback,               0.8f,     1.0f,  1 },
    [MP_LPF]              = { &reverb_lpf_hz,               500.0f, 20000.0f,  1 },
    [MP_HPF]              = { &reverb_hpf_hz,               100.0f,   200.0f,  1 },
    [MP_FB_HIGH_SHELF_HZ] = { &reverb_fb_high_shelf_hz,    1000.0f,  2500.0f,  1 },
    [MP_T0_COUNT]         = { &reverb_t0_count_target, DENSITY_T0_MIN, (float)MAX_T0_TAPS, 1 },
    [MP_T1_COUNT]         = { &reverb_t1_count_target, DENSITY_T1_MIN, (float)MAX_T1_TAPS, 1 },
    [MP_T2_COUNT]         = { &reverb_t2_count_target, DENSITY_T2_MIN, (float)MAX_T2_TAPS, 1 },
};

static const macro_param_id_t decay_params[] = {
    MP_FEEDBACK,
};
static const macro_param_id_t size_params[] = {
    MP_T2_DURATION, MP_T0_WINDOW, MP_T1_WINDOW, MP_T2_FADE,
};
static const macro_param_id_t tone_params[] = {
    MP_LPF, MP_HPF, MP_FB_HIGH_SHELF_HZ,
};
static const macro_param_id_t density_params[] = {
    MP_T0_COUNT, MP_T1_COUNT, MP_T2_COUNT,
};

#define M_COUNT(arr) (int)(sizeof(arr) / sizeof((arr)[0]))
#define MACRO_DECAY  0
#define MACRO_TONE   1
#define MACRO_DENSITY 2
#define MACRO_SIZE   3
#define NUM_MACROS   4
static macro_t reverb_macros[NUM_MACROS] = {
    { 0.86f, M_COUNT(decay_params),   decay_params   },
    { 0.77f, M_COUNT(tone_params),    tone_params    },
    { 1.00f, M_COUNT(density_params), density_params },   /* full density: the DLD's sound */
    { SIZE_MID_U, M_COUNT(size_params), size_params  },   /* the DLD's spacing */
};

/* Macro recompute is ~14 powf calls (~17 µs). Called from params.c at
 * main-loop rate (~1 kHz), but the morph IIRs downstream have a 30 ms tau,
 * so anything faster than ~100 Hz is wasted work. apply_*_macro now just
 * stores the value + sets a dirty flag; the actual recompute happens at
 * audio-block rate, throttled. */
static volatile uint8_t macros_dirty = 1;   /* force initial recompute */

void velvet_reverb_recompute_macros(void)
{
    for (int m = 0; m < NUM_MACROS; m++) {
        float u = reverb_macros[m].value;
        if (u < 0.0f) u = 0.0f; if (u > 1.0f) u = 1.0f;
        const macro_param_id_t *pids = reverb_macros[m].params;
        int n = reverb_macros[m].num_params;
        for (int i = 0; i < n; i++) {
            const macro_param_t *mp = &reverb_macro_params[pids[i]];
            float val;
            if (mp->bound_mid > 0.0f) {
                val = (u < SIZE_MID_U)
                    ? mp->bound_min * powf(mp->bound_mid / mp->bound_min, u * (1.0f / SIZE_MID_U))
                    : mp->bound_mid * powf(mp->bound_max / mp->bound_mid, (u - SIZE_MID_U) * (1.0f / (1.0f - SIZE_MID_U)));
            } else if (mp->use_exp && mp->bound_min > 0.0f && mp->bound_max > 0.0f &&
                mp->bound_min != mp->bound_max) {
                val = mp->bound_min * powf(mp->bound_max / mp->bound_min, u);
            } else {
                val = mp->bound_min + u * (mp->bound_max - mp->bound_min);
            }
            *mp->target = val;
        }
    }
    /* Density's stages, one after another (see DENSITY_T0_MIN). */
    {
        float v = reverb_macros[MACRO_DENSITY].value;
        if (v < 0.0f) v = 0.0f; if (v > 1.0f) v = 1.0f;
        static const macro_param_id_t ids[3] = { MP_T2_COUNT, MP_T1_COUNT, MP_T0_COUNT };
        for (int k = 0; k < 3; k++) {
            /* Stage ids[k] thins over the k-th third from the bottom. */
            float x = v * 3.0f - (float)k;
            if (x < 0.0f) x = 0.0f; else if (x > 1.0f) x = 1.0f;
            const macro_param_t *mp = &reverb_macro_params[ids[k]];
            *mp->target = mp->bound_min * powf(mp->bound_max / mp->bound_min, x);
        }
    }
#if REVERB_LOOP_FB_CURVE
    float d = reverb_macros[MACRO_DECAY].value;
    if (reverb_loop_curve_on && d < reverb_loop_curve_join) {
        const macro_param_t *fp = &reverb_macro_params[MP_FEEDBACK];
        float lo = reverb_loop_curve_lo, join = reverb_loop_curve_join;
        if (join > 0.98f) join = 0.98f;        /* feedback reaches 1.0 at the top */
        if (lo > join - 0.01f) lo = join - 0.01f;
        /* line B's T60 under the shipped feedback at the join, in seconds */
        float fb_join = fp->bound_min * powf(fp->bound_max / fp->bound_min, join);
        float t60_join = -3.0f * reverb_predelay_b_s / log10f(fb_join);
        float x = (d - lo) / (join - lo); if (x < 0.0f) x = 0.0f;
        float t60 = reverb_loop_t60_lo * powf(t60_join / reverb_loop_t60_lo, x);
        float fb_cap = powf(10.0f, -3.0f * reverb_predelay_b_s / t60);
        if (fb_cap < reverb_feedback) reverb_feedback = fb_cap;
    }
#endif
#if REVERB_LOOP_T60_SHELVES
    /* After the feedback is final: do_predelay clamps it to 0.999 the same way. */
    float fb = reverb_feedback;
    if (fb > 0.999f) fb = 0.999f;
    if (fb > 0.0f) {
        float mid_db = 20.0f * log10f(fb);
        float t = reverb_macros[MACRO_TONE].value;
        float hi_r = reverb_t60_hi_ratio_min * powf(reverb_t60_hi_ratio_max / reverb_t60_hi_ratio_min, t);
        t60_hs_db = mid_db * (1.0f / hi_r - 1.0f);
        t60_ls_db = mid_db * (1.0f / reverb_t60_lo_ratio - 1.0f);
    }
#endif
}

/* Dead-band of ~1/4096 — comfortably above 12-bit ADC LSB noise but below
 * any audible step (cosumes a single ADC count of movement). */
#define MACRO_VALUE_DEADBAND  (1.0f / 4096.0f)

#if defined(DIAG_REVERB_PROFILE) && !defined(VELVET_REVERB_HOST)
/* Set by a profile build's variant cycler (prof_step), which then drives the
 * macros and the octave itself: calls from the panel are ignored so the
 * capture runs under a known, repeatable load. */
static uint8_t prof_override = 0;
#define PROF_PANEL_IGNORED()  do { if (prof_override) return; } while (0)
#else
#define PROF_PANEL_IGNORED()  do { } while (0)
#endif

static inline void set_macro_value(int idx, float v) {
    if (v < 0.0f) v = 0.0f; if (v > 1.0f) v = 1.0f;
    if (fabsf(v - reverb_macros[idx].value) < MACRO_VALUE_DEADBAND) return;
    reverb_macros[idx].value = v;
    macros_dirty = 1;
}

/* The Decay→feedback curve (bounds {0.8, 1.0}, exp) is extremely sensitive at
 * the top: the last few % of travel cover fb 0.994→0.999, which is roughly a
 * 6x tail-length change and a ~9 dB drop in the sustain plateau. Hardware pots
 * rarely reach full ADC scale, so the pre-delay tail would never reach the
 * full length the design allows at macro = 100. Saturate the top of the Decay
 * macro so the last DECAY_MACRO_TOP_SAT of travel maps to full — this leaves the
 * true-full behaviour unchanged, it only makes it reachable. */
#define DECAY_MACRO_TOP_SAT  0.96f
void velvet_reverb_apply_decay_macro  (float v) {
    PROF_PANEL_IGNORED();
    v *= (1.0f / DECAY_MACRO_TOP_SAT);          /* set_macro_value clamps to [0,1] */
    set_macro_value(MACRO_DECAY, v);
}
void velvet_reverb_apply_tone_macro   (float v) { PROF_PANEL_IGNORED(); set_macro_value(MACRO_TONE, v); }
/* Below one tap a stage, Density thins the sustain loop's echoes: over the
 * bottom DENSITY_THIN_U of its travel, line A (the shorter) fades out of the
 * cascade feed and line B takes its level, so two interleaved echo trains
 * become one, and the loops' blur turns down to none, so that train repeats
 * only itself instead of multiplying. Line A keeps recirculating, so turning
 * back up brings its tail straight back. */
static float loop_thin_target = 0.0f, loop_thin_morph = 0.0f;   /* 0 = both lines, 1 = B only */

/* ==== Input delay (the panel's pre-delay) ====
 * The input goes into a ring as it arrives and is read back a delay later.
 * The delay asked for is smoothed (~20 ms), then followed two ways:
 *   - small moves glide: the read moves at most ID_SLEW samples per sample
 *     (1%, ~17 cents while it moves), reading between samples, so pot noise
 *     and an attenurandomizer's drift pass without a trace;
 *   - a jump of more than ID_JUMP (a knob turned) crossfades from the old
 *     read to the new over ID_XFADE instead, as gliding that far would bend
 *     the pitch for too long.
 * It used to crossfade on every move of more than 1 ms. On the module, pot
 * noise on TIME (or its attenurandomizer just off noon) moved the delay that
 * much every second or so at mid settings, and each crossfade blended the
 * input with itself a moment apart: heard in the echoes as repeated
 * fragments. */
#define ID_XFADE  (480 * REVERB_RATE_MULT)                 /* 20 ms */
#define ID_JUMP   (0.02f * (float)REVERB_FS_HZ)            /* 20 ms */
#define ID_SLEW   0.01f
#define ID_SMOOTH 0.002f                                   /* ~20 ms at 24 kHz */
static int16_t * const input_delay = (int16_t *)INPUT_DELAY_BASE;
static int16_t * const input_delay_r = (int16_t *)INPUT_DELAY_R_BASE;
static uint32_t id_w;                       /* write index */
static float id_target;                     /* delay asked for, samples */
static float id_smooth;                     /* the same, smoothed */
static float id_pos;                        /* delay read at, samples (fractional) */
static int32_t id_to, id_left;              /* delay fading to, fade samples left */
static float id_out[REVERB_BLOCK];          /* this block's delayed input, int16 scale */
static float id_out_r[REVERB_BLOCK];

void velvet_reverb_set_input_delay(float v)
{
    if (v < 0.0f) v = 0.0f; else if (v > 1.0f) v = 1.0f;
    id_target = v * v * INPUT_DELAY_MAX_S * (float)REVERB_FS_HZ;
}

void velvet_reverb_apply_density_macro(float v)
{
    PROF_PANEL_IGNORED();
    if (v < 0.0f) v = 0.0f; if (v > 1.0f) v = 1.0f;
    set_macro_value(MACRO_DENSITY, v);
    float x = 1.0f - v * (1.0f / DENSITY_THIN_U);
    if (x < 0.0f) x = 0.0f;
    loop_thin_target = x * x * (3.0f - 2.0f * x);
}
/* No top saturation here: the windows' top is not the sensitive end. */
void velvet_reverb_apply_size_macro   (float v) { PROF_PANEL_IGNORED(); set_macro_value(MACRO_SIZE, v); }

/* ==== Morph alphas (per block, derived from per-tau / blocks-per-sec) ====
 * Block rate = fs_reverb / REVERB_BLOCK = 1500 Hz at either rate.
 *   tau ≈ 1/(α × 1500). 0.0073 → ~90 ms (window), 0.00664 → ~100 ms (decay).
 * The windows, tap counts and level comps follow Size and Density at ~90 ms:
 * at the old ~30 ms a quick turn moved the gains far enough between
 * recomputes to zip. */
#define ALPHA_WINDOW_MORPH   0.0073f
/* ==== Morph state ====
 * Windows morph (driven by the Size macro), tap counts morph (Density), and
 * the gain-comp morph tracks the resulting energy. */
static float t0_count_morph;        /* taps; fractional */
static float t1_count_morph;
static float t2_count_morph;
/* Removal rank per tap (0 = kept longest), set at IR-gen. */
static float t0TapRanks[MAX_T0_TAPS];
static float t1TapRanks[MAX_T1_TAPS];
static float t2TapRanks[MAX_T2_TAPS];
static float t0_window_morph;       /* samples at fs_reverb */
static float t1_window_morph;
static float t2_window_morph;
static float t2_fade_morph;         /* samples (Size) */
static float t0_tap_comp_morph;
static float t1_tap_comp_morph;
static float t2_tap_comp_morph;
static float t0_full_energy;        /* sum(baseGain²) — set at IR-gen */
static float t1_full_energy;
static float t2_full_energy;
/* The level comps' latest targets: the effGains keep recomputing until the
 * comps reach them (see background_eff_gains_update). */
static float t0_comp_goal = 1.0f, t1_comp_goal = 1.0f, t2_comp_goal = 1.0f;

/* Pre-delay delay times, smoothed; feedback/mix/shelf changes are gain-like
 * and used directly. Samples at fs_reverb. */
static float predelay_a_morph;
static float predelay_b_morph;

/* ==== Output biquads ==== */
typedef struct {
    float b0, b1, b2, a1, a2;
    float z1, z2;
} Biquad;

static Biquad hpf_L CCM_ATTR;
static Biquad hpf_R CCM_ATTR;
static Biquad lpf_L CCM_ATTR;
static Biquad lpf_R CCM_ATTR;
/* Upper limit on the Tone-driven output LPF. At 24 kHz the LPF is also the
 * reconstruction filter for the 2x linear-interp upsample, so opening it past
 * ~12 kHz lets the interpolation images through. At 48 kHz there are no images,
 * so it can follow Tone to the top of its 20 kHz range. */
#if REVERB_DECIM == 2
#define REVERB_LPF_MAX_HZ  11900.0f
#else
#define REVERB_LPF_MAX_HZ  20000.0f
#endif
static float last_lpf_hz = -1.0f;
static float last_hpf_hz = -1.0f;
#if REVERB_OUTPUT_TILT
static float last_tilt_db = -999.0f, last_tilt_hz = -1.0f;
#endif
/* Smoothed Tone-macro cutoffs. The macro writes reverb_lpf_hz/reverb_hpf_hz/
 * reverb_fb_high_shelf_hz directly (in ADC-sized steps), so feeding them to
 * the biquads raw produced a staircase of coefficient jumps = zipper on a
 * Tone sweep. These IIR-track the targets (per reverb block, ~30 ms tau, same
 * as the window morphs) so the biquads see a continuous cutoff. */
static float lpf_hz_morph   = 8600.0f;
static float hpf_hz_morph   = 170.0f;
static float fb_hs_hz_morph = 2000.0f;

/* Pitch shift on the reverb feed, set by velvet_reverb_set_pitch (semitones,
 * clamped to ±12) and velvet_reverb_set_pitch_mix (0 = none, 1 = only the
 * shifted feed). Declared up here because the HPF morph below follows them as
 * well as the pitch feed heads. */
#define PITCH_SEMIS_MAX  12.0f
static volatile float pitch_semis_target = 0.0f;
static volatile float pitch_mix_target   = 0.0f;

/* ==== Pre-delay sustain engine state ====
 * Two float feedback delay lines in SDRAM (see velvet_reverb.h). Shared loop
 * gain + shelves (tone/damping), per-line modulation + DC block, input duck,
 * and a dry/wet crossfade into the cascade. The line storage is float so the
 * high-feedback loop doesn't
 * accumulate int16 quantisation noise. */
static float * const predelay_a = (float *)PRE_DELAY_A_BASE;
static float * const predelay_b = (float *)PRE_DELAY_B_BASE;
/* The right channel's lines. Every per-line state below has an _r twin, run
 * the same way, so a mono input loops identically on both sides. */
static float * const predelay_a_r = (float *)PRE_DELAY_A_R_BASE;
static float * const predelay_b_r = (float *)PRE_DELAY_B_R_BASE;
static uint32_t predelay_wh = 0;            /* shared free-running write head */
static float pre_mod_phase_a = 0.0f;
static float pre_mod_phase_b = PI_F;        /* decorrelated phase */
static float pre_hp_a = 0.0f, pre_hp_b = 0.0f;   /* per-line ~12 Hz DC block state */
static float pre_hp_a_r = 0.0f, pre_hp_b_r = 0.0f;
/* Per-line fractional-read state (see PREDELAY_READ): the allpass's last input
 * and output at 24 kHz, the cubic's four-sample window at 48 kHz. */
#if REVERB_DECIM == 2
typedef struct {
    float prev_in, prev_out;   /* the allpass: last input and output */
    int32_t i;                 /* its sample index at the last read (INT32_MIN: none yet) */
    float b_in, b_out;         /* the second allpass, during a crossing */
    int32_t bi;
    int32_t xf;                /* crossing fade samples left, 0 = none */
} PdReadState;
#else
typedef struct { int32_t i; float y0, y1, y2, y3; } PdReadState;
#endif
static PdReadState pre_rd_a, pre_rd_b, pre_rd_a_r, pre_rd_b_r;
/* A loop's length changes in steps: the line keeps its read at the old length
 * (so nothing it holds changes pitch) and crossfades to a read at the new one,
 * the way a tape loop is cut shorter or spliced longer. A step is taken only
 * when the length Size asks for is more than PD_LEN_STEP away, so a pot
 * creeping by a hair never moves it, and a new step waits for the fade in
 * progress to finish, so a sweep moves the loop in a run of steps. */
#define PD_LEN_XFADE  (480 * REVERB_RATE_MULT)   /* 20 ms */
#define PD_LEN_STEP   0.01f                      /* of the current length */
typedef struct {
    float time;            /* the length the loop reads at */
    float to;              /* the length it is fading to */
    int32_t left;          /* fade samples left, 0 = none */
    PdReadState rs;        /* the new read's interpolator state */
} PdLen;
static PdLen pre_len_a, pre_len_b, pre_len_a_r, pre_len_b_r;
static float duck_env = 0.0f;
static float fb_morph = -1.0f;     /* the loop feedback, gliding (see do_predelay); < 0 = unset */
static Biquad pre_low_shelf_a CCM_ATTR, pre_high_shelf_a CCM_ATTR;
static Biquad pre_low_shelf_b CCM_ATTR, pre_high_shelf_b CCM_ATTR;
static Biquad pre_low_shelf_a_r CCM_ATTR, pre_high_shelf_a_r CCM_ATTR;
static Biquad pre_low_shelf_b_r CCM_ATTR, pre_high_shelf_b_r CCM_ATTR;
/* Cached shelf coeff inputs — recompute only on change. */
static float last_fb_ls_hz = -1.0f, last_fb_ls_db = -999.0f;
static float last_fb_hs_hz = -1.0f, last_fb_hs_db = -999.0f;
static float pre_hp_coeff = 0.0f;
static float duck_atk_coeff = 0.0f;

/* Per-block float scratch holding the (pre-delayed) cascade input, normalised
 * to ±1 (may exceed during sustain build-up; clamped on the int16 ring write
 * in do_input_write_t0). */
static float predelay_out[REVERB_BLOCK] CCM_ATTR;
static float predelay_out_r[REVERB_BLOCK] CCM_ATTR;

#ifdef VELVET_REVERB_HOST
const float *host_dbg_predelay_out(void) { return predelay_out; }
#endif

/* ==== T0 per-tap Lexicon LFO state ====
 * Each early-reflection tap is displaced by a per-tap (golden-angle phase)
 * LFO so the early cluster shimmers diffusely. cos/sin tables seeded at
 * regen; the shared phase advances per block. */
static float t0_tap_mod_phase = 0.0f;
/* The depth the previous block ended at (-1: none), so a change of depth
 * also glides. */
static float t0_mod_depth_last = -1.0f;
static float t0TapModCos[MAX_T0_TAPS] CCM_ATTR;
static float t0TapModSin[MAX_T0_TAPS] CCM_ATTR;

#ifndef VELVET_REVERB_HOST
static inline void dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;
}

/* ==== Stage timing (pure compute, ISR time removed) ====
 *
 * A stage's cost is wall time minus the codec-ISR time that preempted it. Taking
 * that as two independent reads is racy: if an ISR lands BETWEEN reading CYCCNT
 * and reading diag_isr_cycles, one delta contains that ISR and the other does
 * not. Depending on which end it happens at, the result is either inflated or
 * driven negative — and being unsigned, negative wraps to ~4e9 and saturates at
 * DIAG_CYCLES_MASK, i.e. a fake 6241.5 us. That is not a subtle skew: it
 * manufactures the largest possible outlier out of nothing, in exactly the p99
 * column used to hunt overruns, which is worse than having no measurement.
 *
 * So sample the pair atomically, and clamp instead of wrapping if the two still
 * disagree. Interrupts are masked for two register reads only. */
static inline uint32_t diag_stage_start(uint32_t *isr_out)
{
    uint32_t primask = __get_PRIMASK();
    __disable_irq();
    uint32_t c = DWT->CYCCNT;
    *isr_out = diag_isr_cycles;
    if (!primask) __enable_irq();
    return c;
}

static inline uint32_t diag_stage_cycles(uint32_t cyc0, uint32_t isr0)
{
    uint32_t primask = __get_PRIMASK();
    __disable_irq();
    uint32_t c = DWT->CYCCNT;
    uint32_t i = diag_isr_cycles;
    if (!primask) __enable_irq();
    uint32_t wall = c - cyc0;
    uint32_t isr  = i - isr0;
    return (wall > isr) ? (wall - isr) : 0u;
}
#endif

/* ==== Input queue ====
 *
 * The codec ISR fills a block and hands it to poll, which runs in the main loop.
 * This used to be a two-buffer swap guarded by a single `block_ready` flag, and
 * that gave the main loop exactly ONE block period — 666 us — to collect a block
 * before the next one had nowhere to go and was thrown away. A discarded block
 * splices REVERB_BLOCK samples out of the reverb's input, and the cascade
 * diffuses that step into an audible whoosh. Since the main loop only has to
 * overrun once for this to fire, it presented as an occasional whoosh rather
 * than anything obviously correlated with load.
 *
 * A queue of REVERB_IN_NBUF blocks turns that hard 666 us deadline into
 * (REVERB_IN_NBUF - 1) block periods of slack. Note what this does NOT cost:
 * poll still consumes a block as soon as one is available, so steady-state
 * latency is unchanged at one block. The depth is only drawn upon when the main
 * loop runs late, and it drains again as soon as it catches up — latency is
 * transient, proportional to actual lateness, and bounded by the depth. That is
 * the opposite of the output ring, where the cushion is a standing cost.
 *
 * Lock-free single-producer/single-consumer via monotonic sequence counters:
 * the ISR owns in_wr_seq, poll owns in_rd_seq, and neither writes the other's.
 * Depth in flight is (in_wr_seq - in_rd_seq), which is correct across counter
 * wrap because both are unsigned. The producer publishes by incrementing
 * in_wr_seq only after the samples are written, with a barrier between, so poll
 * can never observe a sequence number for a block that is not yet complete.
 *
 * Sized against measured lateness: poll latency on hardware peaked at 2.2 ms,
 * and the harness confirms the depth is the binding limit (stalls below it are
 * absorbed, stalls above it still drop). 16 blocks is 512 bytes for 10 ms of
 * tolerance — roughly 4x the observed worst case, which is the right trade when
 * a single overrun is audible and the memory is this cheap. REVERB_IN_NBUF is in
 * the header, and overridable so the harness can A/B against the old one-block
 * behaviour by building with 2. */

/* Power of two keeps the modulo in the ISR an AND. */
typedef char in_nbuf_pow2_check[((REVERB_IN_NBUF & (REVERB_IN_NBUF - 1)) == 0) ? 1 : -1];

static int16_t in_ring[REVERB_IN_NBUF][REVERB_BLOCK];
static int16_t in_ring_r[REVERB_IN_NBUF][REVERB_BLOCK];
static volatile uint32_t in_wr_seq = 0;   /* written by ISR only */
static volatile uint32_t in_rd_seq = 0;   /* written by poll only */
static int16_t *input_ready = in_ring[0]; /* block poll is currently reading */
static int16_t *input_ready_r = in_ring_r[0];
static uint8_t  input_fill_idx = 0;
static int32_t push_acc = 0, push_acc_r = 0;
static uint8_t push_phase = 0;
static uint32_t block_write_idx = 0;
/* High-water mark of queue occupancy: how much of the slack the main loop
 * actually uses, so the depth can be sized from measurement rather than guessed
 * at. Not a diag event because the on-wire event field is 4 bits and all 16
 * slots are taken; read it from the debugger or the host harness. */
volatile uint8_t input_queue_max = 0;

/* ==== Output-underrun (bitcrush) instrumentation ====
 * blocks_produced is the monotonic count of fully-written output blocks; the
 * output ring readers use it as the publish sequence (see below). It also
 * doubles as the ground-truth defect counter: output_miss_count is bumped only
 * when a reader genuinely underruns the ring (poll produced nothing for two
 * consecutive wraps), independent of the FSK diagnostic load.
 * block_ready_cyc timestamps when an input block goes ready so poll can report
 * how long it sat before the main loop picked it up. */
static volatile uint32_t blocks_produced   = 0;
static volatile uint32_t output_miss_count = 0;
static volatile uint32_t output_miss_count_r = 0;   /* the right reader's */
/* Input blocks discarded because poll had not consumed the previous one. See
 * the drop path in velvet_reverb_push_sample. */
volatile uint32_t input_drop_count = 0;
#ifndef VELVET_REVERB_HOST
/* Per-slot, not a single global: poll consumes the oldest queued block while the
 * ISR publishes the newest, so one shared timestamp would report the age of a
 * block other than the one being processed and hide exactly the lateness this
 * queue exists to absorb. */
static volatile uint32_t in_ready_cyc[REVERB_IN_NBUF];
#endif

static int32_t accT0[REVERB_BLOCK] CCM_ATTR;
static int32_t accT1[REVERB_BLOCK] CCM_ATTR;
static int32_t accT0_r[REVERB_BLOCK] CCM_ATTR;
static int32_t accT1_r[REVERB_BLOCK] CCM_ATTR;
static int32_t accL [REVERB_BLOCK] CCM_ATTR;
static int32_t accR [REVERB_BLOCK] CCM_ATTR;

/* Accumulator blocking.
 *
 * The natural way to write these convolutions puts the tap loop outermost and
 * the sample loop inside, which means the whole accumulator array is loaded and
 * stored once per tap. GCC cannot keep it in registers across the tap boundary
 * because these accumulators are statics whose address escapes, so it emits a
 * real load/qadd/store per sample per tap — eleven instructions for two MACs,
 * four of them pure accumulator traffic.
 *
 * Interchanging the loops fixes it: hold REVERB_ACC_CHUNK accumulators in
 * registers, sweep every tap into them, then store once. At 40 taps that turns
 * roughly 1280 accumulator accesses per stage into 32.
 *
 * The interchange is exactly bit-preserving, which is what makes it safe to
 * apply to a saturating accumulation. qadd_sat is not associative, so reordering
 * the additions would be a genuine risk — but this does not reorder them. For any
 * given output sample the taps are still summed in ascending tap order; all that
 * changes is the order in which different output samples are visited. The IR
 * harness confirms this byte for byte against the pre-optimisation baseline.
 *
 * Chunk of 4 is chosen to fit: four accumulators plus gain, base, source pointer
 * and two sample pairs stay inside the register file without spilling, which is
 * the whole point of the exercise. */
#define REVERB_ACC_CHUNK 4

/* The blocked convolutions are kept out of line. Under LTO, poll is inlined into
 * main, which uses r7 as a frame pointer; with one register fewer, two of the four
 * accumulators spilled to the stack and each tap paid four extra memory accesses,
 * undoing much of the blocking. As separate functions they get the whole register
 * file, and the call is once per block. */
#define REVERB_CONV_FN  __attribute__((noinline))

/* ==== Output ring (jitter buffer) ====
 * poll() produces one output block per input block (avg rate matches the
 * codec consume rate — reverb_drop is 0). The defect was timing jitter: poll
 * runs asynchronously in the main loop and occasionally finishes a block just
 * *after* the codec reader has already wrapped, so the reader replayed the
 * previous block (audible bitcrush). A 2-deep double buffer had zero slack to
 * absorb that.
 *
 * This ring decouples production from consumption. The writer (poll) fills
 * slot (blocks_produced % NBUF) then publishes by incrementing blocks_produced.
 * Each codec reader advances its own monotonic read sequence by one block per
 * wrap and holds a steady lag of ~1 completed block behind the newest, so a
 * single late/slow poll no longer starves it — only two consecutive misses do.
 * The reader holds a steady TARGET_LAG-block cushion behind the newest
 * published block: it accumulates slack during fast polls and spends it during
 * slow ones, so a run of late/slow polls no longer starves it. Only when the
 * cushion is fully drained (TARGET_LAG consecutive misses) does a real bitcrush
 * occur. NBUF > TARGET_LAG keeps a guard slot so the writer can never overwrite
 * a block being read. Costs TARGET_LAG blocks of latency (~5.3 ms, inaudible
 * on a reverb tail). */
#define REVERB_OUT_NBUF        16
#define REVERB_OUT_TARGET_LAG  8   /* 5.3 ms: on the Beads the main loop can be a few ms late */
static int16_t outL[REVERB_OUT_NBUF][REVERB_OUT_BLOCK] CCM_ATTR;
static int16_t outR[REVERB_OUT_NBUF][REVERB_OUT_BLOCK] CCM_ATTR;
/* Per-channel read sequence + cached slot. The two codec ISRs wrap
 * independently (their DMAs can drift), so each tracks the writer on its own;
 * neither depends on the other firing in lockstep. */
static uint32_t rd_seq_L = 0;
static uint32_t rd_seq_R = 0;
static uint8_t  reader_buf_L = 0;
static uint8_t  reader_started_L = 0, reader_started_R = 0;   /* see out_next_block */
static uint8_t  reader_buf_R = 0;
static uint8_t  out_idx_L = 0;
static uint8_t  out_idx_R = 0;

static float prev_out_L = 0.0f;   /* width-processed reverb-rate output (for upsample interp) */
static float prev_out_R = 0.0f;

/* Output limiter state. Two phases: one block being computed, one being emitted,
 * which is what provides the look-ahead (see the limiter notes further down). */
static float   lim_buf_L[2][REVERB_OUT_BLOCK] CCM_ATTR;
static float   lim_buf_R[2][REVERB_OUT_BLOCK] CCM_ATTR;
static uint8_t lim_phase = 0;
static float   lim_gain = 1.0f;         /* gain at the last block boundary */
static float   lim_gain_target = 1.0f;
/* Each block's wanted gain over the last LIM_HOLD_BLOCKS, for the hold (see
 * LIM_HOLD_BLOCKS). */
#define LIM_HOLD_BLOCKS 24
static float   lim_hold[LIM_HOLD_BLOCKS];
static uint8_t lim_hold_idx = 0;

/* ==== PRNG ==== */
static uint32_t prng_state = 0xDEADBEEF;
static uint32_t xorshift32(void)
{
    uint32_t x = prng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    prng_state = x;
    return x;
}

/* ==== Smoothstep helper for fade curves ==== */
static inline float smoothstep01(float x)
{
    if (x <= 0.0f) return 0.0f;
    if (x >= 1.0f) return 1.0f;
    return x * x * (3.0f - 2.0f * x);
}
/* Per-tap T2 exponential decay envelope (frac ∈ [0,1] = effOff / window).
 * exp falls to ~1e-4 (-80 dB) across the window. T0 has taper baked in;
 * T1 has no decay envelope. Uses exp_env_lut (filled in velvet_reverb_init). */
#define ENV_FRAC_BINS  32
static float exp_env_lut[ENV_FRAC_BINS] CCM_ATTR;

static inline float exp_decay_from_frac(float frac)
{
    if (frac < 0.0f) frac = 0.0f;
    if (frac > 1.0f) frac = 1.0f;
    float bin_f = frac * (float)(ENV_FRAC_BINS - 1);
    int i = (int)bin_f;
    if (i > ENV_FRAC_BINS - 2) i = ENV_FRAC_BINS - 2;
    float t = bin_f - (float)i;
    return exp_env_lut[i] + t * (exp_env_lut[i + 1] - exp_env_lut[i]);
}

/* ==== Biquad cookbook coefficients ==== */
/* TONE's output low-pass. REVERB_TONE_LPF_6DB 1 (a host A/B prototype; 0 is
 * the shipped sound) swaps the 12 dB/oct biquad for a 6 dB/oct first-order
 * low-pass at the same corner, for more air above it. */
#ifndef REVERB_TONE_LPF_6DB
#define REVERB_TONE_LPF_6DB 0
#endif
#if REVERB_TONE_LPF_6DB
static void biquad_lpf1(Biquad *bq, float freq, float fs)
{
    float k = tanf(PI_F * freq / fs);
    bq->b0 = k / (1.0f + k);
    bq->b1 = bq->b0;
    bq->b2 = 0.0f;
    bq->a1 = (k - 1.0f) / (k + 1.0f);
    bq->a2 = 0.0f;
}
#define TONE_LPF biquad_lpf1
#else
#define TONE_LPF biquad_lpf
#endif

static void biquad_lpf(Biquad *bq, float freq, float fs)
{
    float w0 = TWO_PI_F * freq / fs;
    float alpha = sinf(w0) / (2.0f * 0.707f);
    float cosw = cosf(w0);
    float a0 = 1.0f + alpha;
    bq->b0 = ((1.0f - cosw) * 0.5f) / a0;
    bq->b1 = (1.0f - cosw) / a0;
    bq->b2 = ((1.0f - cosw) * 0.5f) / a0;
    bq->a1 = (-2.0f * cosw) / a0;
    bq->a2 = (1.0f - alpha) / a0;
}
#if REVERB_OUTPUT_TILT
/* One-pole split, y = lo·lp + hi·(x − lp), as a biquad with b2 = a2 = 0. */
static void biquad_tilt(Biquad *bq, float freq, float db, float fs)
{
    float c  = 1.0f - expf(-TWO_PI_F * freq / fs);
    float p  = 1.0f - c;
    float hi = powf(10.0f,  db / 40.0f);
    float lo = powf(10.0f, -db / 40.0f);
    bq->b0 = hi + (lo - hi) * c;
    bq->b1 = -hi * p;
    bq->b2 = 0.0f;
    bq->a1 = -p;
    bq->a2 = 0.0f;
}
#endif
static void biquad_hpf(Biquad *bq, float freq, float fs)
{
    float w0 = TWO_PI_F * freq / fs;
    float alpha = sinf(w0) / (2.0f * 0.707f);
    float cosw = cosf(w0);
    float a0 = 1.0f + alpha;
    bq->b0 = ((1.0f + cosw) * 0.5f) / a0;
    bq->b1 = -(1.0f + cosw) / a0;
    bq->b2 = ((1.0f + cosw) * 0.5f) / a0;
    bq->a1 = (-2.0f * cosw) / a0;
    bq->a2 = (1.0f - alpha) / a0;
}
/* RBJ shelving filters (S = 1 slope). Used inside the pre-delay feedback loop
 * at the reverb rate. dB > 0 boosts, < 0 cuts. */
static void biquad_lowshelf(Biquad *bq, float freq, float dB, float fs)
{
    float A = powf(10.0f, dB / 40.0f);
    float w0 = TWO_PI_F * freq / fs;
    float cosw = cosf(w0);
    float alpha = sinf(w0) * 0.5f * 1.41421356f;     /* sqrt(2) */
    float tsa = 2.0f * sqrtf(A) * alpha;
    float a0 = (A + 1.0f) + (A - 1.0f) * cosw + tsa;
    bq->b0 =  A * ((A + 1.0f) - (A - 1.0f) * cosw + tsa) / a0;
    bq->b1 =  2.0f * A * ((A - 1.0f) - (A + 1.0f) * cosw) / a0;
    bq->b2 =  A * ((A + 1.0f) - (A - 1.0f) * cosw - tsa) / a0;
    bq->a1 = -2.0f * ((A - 1.0f) + (A + 1.0f) * cosw) / a0;
    bq->a2 =  ((A + 1.0f) + (A - 1.0f) * cosw - tsa) / a0;
}
static void biquad_highshelf(Biquad *bq, float freq, float dB, float fs)
{
    float A = powf(10.0f, dB / 40.0f);
    float w0 = TWO_PI_F * freq / fs;
    float cosw = cosf(w0);
    float alpha = sinf(w0) * 0.5f * 1.41421356f;
    float tsa = 2.0f * sqrtf(A) * alpha;
    float a0 = (A + 1.0f) - (A - 1.0f) * cosw + tsa;
    bq->b0 =  A * ((A + 1.0f) + (A - 1.0f) * cosw + tsa) / a0;
    bq->b1 = -2.0f * A * ((A - 1.0f) + (A + 1.0f) * cosw) / a0;
    bq->b2 =  A * ((A + 1.0f) + (A - 1.0f) * cosw - tsa) / a0;
    bq->a1 =  2.0f * ((A - 1.0f) - (A + 1.0f) * cosw) / a0;
    bq->a2 =  ((A + 1.0f) - (A - 1.0f) * cosw - tsa) / a0;
}
/* Each filter here comes as a matched pair (output L/R, pre-delay lines A/B) with
 * identical coefficients and separate state. Design once and copy the
 * coefficients: while Tone moves, every block redesigns these, and each design
 * is several transcendental calls. The state (z1, z2) is not touched. */
__attribute__((always_inline)) static inline void biquad_copy_coeffs(Biquad *dst, const Biquad *src)
{
    dst->b0 = src->b0; dst->b1 = src->b1; dst->b2 = src->b2;
    dst->a1 = src->a1; dst->a2 = src->a2;
}

__attribute__((always_inline)) static inline float biquad_process(Biquad *bq, float x)
{
    float y = bq->b0 * x + bq->z1;
    bq->z1 = bq->b1 * x - bq->a1 * y + bq->z2;
    bq->z2 = bq->b2 * x - bq->a2 * y;
    return y;
}

/* Octave shift feed heads and pre-delay line I/O: see do_predelay(). */
static void oct_init(void);
static void pd_init(void);

/* ==== DMA2 (T2 gather, pre-delay line I/O) ==== */
/* `make NODMA=1` replaces every reverb DMA transfer with a synchronous CPU copy.
 * T2's DMA shares the upper SDRAM half with channel 1's audio reads, so this is
 * the A/B for bus contention there. */
#ifndef DMA2_DISABLED_FOR_DIAGNOSTIC
#define DMA2_DISABLED_FOR_DIAGNOSTIC  0
#endif

/* Stream 1's configuration, so the handler can restart it with a plain write
 * rather than a read-modify-write of CR. Memory to memory, 32-bit transfers
 * (see velvet_reverb_init for why that is safe). Medium priority: below the
 * pre-delay streams, which the CPU waits on, and the ADCs (see adc.c). */
#define T2_DMA_CR        (DMA_SxCR_DIR_1 | DMA_SxCR_MINC | DMA_SxCR_PINC \
                          | (2u << 11) | (2u << 13) | (1u << 16))
#define T2_DMA_CR_CHAIN  (T2_DMA_CR | DMA_SxCR_TCIE | DMA_SxCR_TEIE)

/* ==== T2 gather: windows fetched ahead, then a blocked convolution ====
 *
 * Every window a block's T2 reads is in t2_gather before the block starts, and T2
 * runs the same register-blocked loop as T1 over data already in SRAM. (Fetching
 * each window as T2 reached it forced the accumulators through memory per tap.)
 *
 * A row holds two blocks' windows, which are contiguous in the ring: the next
 * block's and the one after's. So only every other block starts a fetch, which
 * halves the number of transfers, and the interrupts that drive them are what
 * the fetch costs the CPU. Fetching ahead is valid because:
 *   - the data is final: every T2 tap reads at least TAP_MIN_OFFSET samples back,
 *     more than two blocks, so the newest sample either window needs was written
 *     (and pre-T2-saturated) before the fetch. A row whose offset is shorter than
 *     two blocks would make the second block's windows stale, so the rows then
 *     serve one block only;
 *   - the offsets are final: effOffT2 only changes in the effGains recompute,
 *     which runs at the end of poll, before the fetch starts.
 * Rows are reused for the second block only if every offset still matches, and T2
 * checks the block index and every offset before using them, fetching
 * synchronously if anything differs, so the output is bit-identical to fetching
 * at T2 time in every case.
 *
 * On the device the fetch is a chain of DMA transfers, one per row, each kicked
 * by the previous one's transfer-complete interrupt, so it runs in the background
 * while the next block's pre-delay, T0 and T1 execute. The buffer is in SRAM
 * because the DMA cannot reach CCM. */
#define T2_GATHER_ROWS  (2 * MAX_T2_TAPS)            /* L windows, then R */
#define T2_GATHER_SPAN  (2 * REVERB_BLOCK)           /* samples per row: two blocks */
static int16_t  t2_gather[T2_GATHER_ROWS][T2_GATHER_SPAN] __attribute__((aligned(4)));
static uint32_t t2_gather_off[T2_GATHER_ROWS];       /* offset each row was fetched at */
static uint32_t t2_gather_bw;                        /* block_write_idx of the rows' first block */
static uint8_t  t2_gather_both;                      /* 1 if the second block's windows are final too */
#define T2G_INVALID  0
#define T2G_BUSY     1
#define T2G_READY    2
static volatile uint8_t t2_gather_state = T2G_INVALID;

static inline uint32_t t2_row_off(int r)
{
    return (r < MAX_T2_TAPS) ? effOffT2L[r] : effOffT2R[r - MAX_T2_TAPS];
}
/* The L rows read the left T2 ring, the R rows the right. */
static inline const int16_t *t2_row_ring(int r)
{
    return (r < MAX_T2_TAPS) ? t2_ring : t2_ring_r;
}

/* Where block bw's windows start within the rows (0, or REVERB_BLOCK for the
 * second block), or -1 if the rows do not hold them: fetched for another block,
 * still in flight, or fetched at offsets that have since changed. */
static int t2_gather_find(uint32_t bw)
{
    if (t2_gather_state != T2G_READY) return -1;
    const uint32_t d = (bw - t2_gather_bw) & (REVERB_IDX_WRAP - 1u);
    if (!(d == 0u || (d == (uint32_t)REVERB_BLOCK && t2_gather_both))) return -1;
    const int count = t2TapCount;
    for (int t = 0; t < count; t++) {
        if (t2_gather_off[t] != effOffT2L[t]) return -1;
        if (t2_gather_off[MAX_T2_TAPS + t] != effOffT2R[t]) return -1;
    }
    return (int)d;
}

/* Fetch every row in use for block index bw, synchronously, by CPU. */
static void t2_gather_sync(uint32_t bw)
{
    const int count = t2TapCount;
    uint8_t both = 1;
    for (int r = 0; r < T2_GATHER_ROWS; r++) {
        int t = (r < MAX_T2_TAPS) ? r : r - MAX_T2_TAPS;
        if (t >= count) continue;
        uint32_t off = t2_row_off(r);
        uint32_t idx = (bw - off) & T2_RING_MASK;
        t2_gather_off[r] = off;
        if (off < (uint32_t)T2_GATHER_SPAN) both = 0;
        const int16_t *ring = t2_row_ring(r);
        if (idx + T2_GATHER_SPAN <= T2_RING_SAMPLES) {
            memcpy(t2_gather[r], &ring[idx], T2_GATHER_SPAN * sizeof(int16_t));
        } else {
            for (int i = 0; i < T2_GATHER_SPAN; i++)
                t2_gather[r][i] = ring[(idx + (uint32_t)i) & T2_RING_MASK];
        }
    }
    t2_gather_bw = bw;
    t2_gather_both = both;
    t2_gather_state = T2G_READY;
}

#if defined(VELVET_REVERB_HOST) || DMA2_DISABLED_FOR_DIAGNOSTIC
/* No DMA: fetch at the same point, synchronously. On the host this is what makes
 * the harness test the early-fetch reasoning above. */
static void t2_gather_start(void)
{
    if (t2_gather_find(ring_write_idx) >= 0) return;    /* the rows already hold it */
    t2_gather_sync(ring_write_idx);
}
static inline void t2_gather_wait(void) { }
#else
/* The chain, as (source, destination) pairs walked by pointer: the interrupt
 * handler is the one piece of this that runs once per row, so it is kept to one
 * paired load, the flag clear, and four register writes. */
typedef struct { const int16_t *src; int16_t *dst; } T2ChainEntry;
static T2ChainEntry t2_chain[T2_GATHER_ROWS];
static const T2ChainEntry *t2_chain_next;   /* entry the next interrupt starts */
static const T2ChainEntry *t2_chain_end;

/* A chain that has not finished long after it should have is broken (a DMA fault
 * with no interrupt, say). Abandon it and let T2 fetch synchronously rather than
 * spin forever. 300 µs is several times the whole chain's duration. */
#define T2G_TIMEOUT_CYC  (180u * 300u)

static void t2_chain_abort(void)
{
    DMA2_Stream1->CR = T2_DMA_CR;               /* EN and the interrupts off */
    while (DMA2_Stream1->CR & DMA_SxCR_EN) {}
    DMA2->LIFCR = 0x00000F40;
    NVIC_ClearPendingIRQ(DMA2_Stream1_IRQn);
    t2_gather_state = T2G_INVALID;
}

static void t2_gather_start(void)
{
    if (t2_gather_state == T2G_BUSY) t2_chain_abort();   /* never expected */
    /* ring_write_idx is the block_write_idx the next block will use. */
    const uint32_t bw = ring_write_idx;
    if (t2_gather_find(bw) >= 0) return;                 /* the rows already hold it */
    const int count = t2TapCount;
    uint8_t both = 1;
    T2ChainEntry *e = t2_chain;
    for (int r = 0; r < T2_GATHER_ROWS; r++) {
        int t = (r < MAX_T2_TAPS) ? r : r - MAX_T2_TAPS;
        if (t >= count) continue;
        uint32_t off = t2_row_off(r);
        uint32_t idx = (bw - off) & T2_RING_MASK;
        t2_gather_off[r] = off;
        if (off < (uint32_t)T2_GATHER_SPAN) both = 0;
        if (idx + T2_GATHER_SPAN <= T2_RING_SAMPLES) {
            e->src = &t2_row_ring(r)[idx];
            e->dst = t2_gather[r];
            e++;
        } else {
            /* Straddles the ring end: rare, so copy it here rather than split it. */
            for (int i = 0; i < T2_GATHER_SPAN; i++)
                t2_gather[r][i] = t2_row_ring(r)[(idx + (uint32_t)i) & T2_RING_MASK];
        }
    }
    t2_gather_bw = bw;
    t2_gather_both = both;
    if (e == t2_chain) {
        t2_gather_state = T2G_READY;
    } else {
        t2_chain_next = &t2_chain[1];
        t2_chain_end  = e;
        t2_gather_state = T2G_BUSY;
        /* This block's ring writes, and the chain table, must be in memory before
         * the DMA and its interrupt read them. */
        __DSB();
        DMA2_Stream1->CR = T2_DMA_CR;
        while (DMA2_Stream1->CR & DMA_SxCR_EN) {}
        DMA2->LIFCR = 0x00000F40;
        DMA2_Stream1->PAR  = (uint32_t)t2_chain[0].src;
        DMA2_Stream1->M0AR = (uint32_t)t2_chain[0].dst;
        DMA2_Stream1->NDTR = T2_GATHER_SPAN / 2;
        DMA2_Stream1->CR   = T2_DMA_CR_CHAIN;
        DMA2_Stream1->CR   = T2_DMA_CR_CHAIN | DMA_SxCR_EN;
    }
}

/* Each transfer's completion kicks the next. Preemption priority 2: below the
 * codec ISRs, above the main loop the chain overlaps. After a completion EN is
 * already clear and CR otherwise holds T2_DMA_CR_CHAIN, so setting EN with a
 * plain write changes nothing else. */
void DMA2_Stream1_IRQHandler(void)
{
    DIAG_IRQ_ENTER();
    const uint32_t lisr = DMA2->LISR;
    DMA2->LIFCR = 0x00000F40;
    const T2ChainEntry *e = t2_chain_next;
    if (!(lisr & DMA_LISR_TEIF1) && t2_gather_state == T2G_BUSY && e != t2_chain_end) {
        t2_chain_next = e + 1;
        DMA2_Stream1->PAR  = (uint32_t)e->src;
        DMA2_Stream1->M0AR = (uint32_t)e->dst;
        DMA2_Stream1->NDTR = T2_GATHER_SPAN / 2;
        DMA2_Stream1->CR   = T2_DMA_CR_CHAIN | DMA_SxCR_EN;
    } else {
        DMA2_Stream1->CR = T2_DMA_CR;           /* chain over: interrupts off */
        __DMB();
        if (lisr & DMA_LISR_TEIF1)              t2_gather_state = T2G_INVALID;
        else if (t2_gather_state == T2G_BUSY)   t2_gather_state = T2G_READY;
    }
    DIAG_IRQ_EXIT(DIAG_CPU_T2DMA);
}

static inline void t2_gather_wait(void)
{
    if (t2_gather_state != T2G_BUSY) return;
    uint32_t t0 = DWT->CYCCNT;
    while (t2_gather_state == T2G_BUSY) {
        if ((DWT->CYCCNT - t0) > T2G_TIMEOUT_CYC) { t2_chain_abort(); break; }
    }
    __DMB();
}
#endif

/* Register-blocked convolution of the gathered windows: the same structure as
 * do_t1_phase, so each output sample still sums its taps in ascending order and
 * the saturating accumulation is bit-identical to accumulating tap by tap. Rows
 * are T2_GATHER_SPAN int16 apart and 4-byte aligned, as is either half of one,
 * so the pair loads are aligned. */
REVERB_CONV_FN static void t2_conv_blocked(const int16_t *rows, int32_t *acc, int count, const int16_t *gains)
{
    for (int c = 0; c < REVERB_BLOCK; c += REVERB_ACC_CHUNK) {
        int32_t a0 = acc[c], a1 = acc[c+1], a2 = acc[c+2], a3 = acc[c+3];
        const int16_t *src = rows + c;
        for (int t = 0; t < count; t++, src += T2_GATHER_SPAN) {
            const uint32_t gain = (uint32_t)(int32_t)gains[t];
            uint32_t p0 = *((const u32_alias *)(src));
            uint32_t p1 = *((const u32_alias *)(src + 2));
            a0 = qadd_sat(a0, smulbb(p0, gain));
            a1 = qadd_sat(a1, smultb(p0, gain));
            a2 = qadd_sat(a2, smulbb(p1, gain));
            a3 = qadd_sat(a3, smultb(p1, gain));
        }
        acc[c] = a0; acc[c+1] = a1; acc[c+2] = a2; acc[c+3] = a3;
    }
}

/* ==== Tap generation ==== */
/* Every stage's first tap, so the first sound arrives 3 x this after the
 * input (4 ms at 32). T0's modulation only adds delay and each stage writes
 * its block before reading, so a stage needs no minimum of its own; T2's
 * fetch-ahead needs more than two blocks (see t2_gather). */
#define TAP_MIN_OFFSET   32
#define TAP_MIN_GRID     16


#ifdef VELVET_REVERB_HOST
/* Test hook (host only): when 0, tap 0 is jittered like the others instead of
 * being pinned to TAP_MIN_OFFSET. Lets the spectral harness A/B whether the
 * pinned, triple-correlated early tap is responsible for a fixed spectral peak.
 * Firmware build always pins tap 0 (this symbol does not exist there). */
int host_pin_tap0 = 1;
#define TAP0_PINNED(j) ((j) == 0 && host_pin_tap0)
#else
#define TAP0_PINNED(j) ((j) == 0)
#endif

/* Generate one mono sparse-velvet stage. `apply_taper` controls whether the
 * cosine tail-taper is baked into the gains (T0 yes, T1/T2 no — they get the
 * decay envelope applied dynamically per-block instead). */
static void generate_mono_stage(uint32_t *outOffsets, int16_t *outGains,
                                int *outCount, int wantTaps, int windowSamples,
                                int apply_taper)
{
    int n = wantTaps;
    int usable = windowSamples - TAP_MIN_OFFSET;
    if (usable < TAP_MIN_GRID) usable = TAP_MIN_GRID;
    int maxTaps = usable / TAP_MIN_GRID;
    if (n > maxTaps) n = maxTaps;

    int gridSize = usable / n;
    int jitterRange = gridSize - TAP_MIN_GRID;
    if (jitterRange < 1) jitterRange = 1;

    float norm = 1.0f / sqrtf((float)n);
    int taperLen = n * 15 / 100;
    if (taperLen < 1) taperLen = 1;

    for (int j = 0; j < n; j++) {
        /* Pin tap 0 to exactly TAP_MIN_OFFSET so every stage has a tap at
         * the very start of its window — guarantees an early reflection
         * regardless of window size. */
        int jit = TAP0_PINNED(j) ? 0 : (int)(xorshift32() % (uint32_t)jitterRange);
        int pos = TAP_MIN_OFFSET + j * gridSize + jit;
        if (pos >= windowSamples) pos = TAP_MIN_OFFSET + j * gridSize;

        float sign = (xorshift32() & 1) ? 1.0f : -1.0f;
        float win = 1.0f;
        if (apply_taper && j >= n - taperLen) {
            int fromEnd = n - 1 - j;
            win = 0.5f * (1.0f - cosf(PI_F * (float)fromEnd / (float)taperLen));
        }

        outOffsets[j] = (uint32_t)(pos & ~1);
        outGains[j] = (int16_t)(sign * norm * win * TAP_GAIN_HEADROOM * 32767.0f);
    }
    *outCount = n;
}

/* Each tap's ladder holds one rung, its own position: Size only decides how
 * much of each stage's span sounds (compute_tap_states fades a tap in as the
 * window reaches it), so the taps keep the same spacing, and the stage the
 * same taps per second, at every Size. */
static void build_single_rung_ladders(const uint32_t *offL, const uint32_t *offR, int count,
                                      uint32_t *ladL, uint32_t *ladR, uint8_t *ladderCount)
{
    for (int k = 0; k < count; k++) {
        ladL[k * MAX_LADDER_LEVELS] = offL[k];
        if (offR) ladR[k * MAX_LADDER_LEVELS] = offR[k];
        ladderCount[k] = 1;
    }
}

/* Stretch offsets laid out in 24 kHz samples onto the REVERB_FS_HZ grid. */
#if REVERB_RATE_MULT > 1
static void scale_offsets_to_rate(uint32_t *v, int n)
{
    for (int i = 0; i < n; i++) v[i] *= (uint32_t)REVERB_RATE_MULT;
}
#endif

/* The layout, and the ladders built from it, are generated in 24 kHz samples
 * (TAP_MIN_OFFSET, TAP_MIN_GRID and the window maxima below are all 24 kHz
 * figures) and only then scaled to the running rate. That keeps the PRNG draws,
 * and so every tap position in seconds, identical at either rate, which makes a
 * 24 vs 48 kHz comparison a comparison of bandwidth and nothing else. Scaling by
 * an integer keeps the offsets even for the pair reads. */
/* Removal rank per tap, for the Density macro: lower rank = kept longer.
 * Repeatedly removes the tap whose nearest neighbour is closest (the most
 * redundant), so the taps left at any count stay spread across the window.
 * A small bias toward removing late taps first keeps the early ones, and the
 * first two taps are only removed once four or fewer remain. O(N²), at
 * IR-gen only. Offsets must be ascending, as the generators make them. */
static void compute_tap_ranks(const uint32_t *offsets, float *ranks, int count)
{
    if (count <= 0) return;
    if (count == 1) { ranks[0] = 0.0f; return; }

    int remaining[MAX_T0_TAPS > MAX_T1_TAPS ? (MAX_T0_TAPS > MAX_T2_TAPS ? MAX_T0_TAPS : MAX_T2_TAPS)
                                            : (MAX_T1_TAPS > MAX_T2_TAPS ? MAX_T1_TAPS : MAX_T2_TAPS)];
    int remainingLen = count;
    for (int i = 0; i < count; i++) remaining[i] = i;

    const int protectedCount = 2;
    const int protectedFloor = 4;
    const uint32_t minOffset = offsets[0];
    const uint32_t maxOffset = offsets[count - 1];
    const float span = (float)((maxOffset > minOffset) ? (maxOffset - minOffset) : 1u);
    const float earlyBiasScale = 0.15f * span / (float)(count - 1);

    /* Removed first = highest rank. */
    int nextRank = count - 1;
    while (remainingLen > 1) {
        int best = -1;
        float bestScore = 1e30f;
        for (int li = 0; li < remainingLen; li++) {
            const int t = remaining[li];
            if (t < protectedCount && remainingLen > protectedFloor) continue;
            const uint32_t pos = offsets[t];
            const float left  = (li > 0) ? (float)(pos - offsets[remaining[li - 1]]) : 1e30f;
            const float right = (li + 1 < remainingLen) ? (float)(offsets[remaining[li + 1]] - pos) : 1e30f;
            const float nearest = left < right ? left : right;
            const float score = nearest + earlyBiasScale * (1.0f - (float)(pos - minOffset) / span);
            if (score < bestScore) { bestScore = score; best = li; }
        }
        if (best < 0) best = remainingLen - 1;
        ranks[remaining[best]] = (float)nextRank--;
        for (int i = best + 1; i < remainingLen; i++) remaining[i - 1] = remaining[i];
        remainingLen--;
    }
    ranks[remaining[0]] = 0.0f;
}

/* How much of a tap the current count keeps: 1 below the boundary, 0 above,
 * and a smooth part-fade for the one tap the fractional count is crossing. */
static inline float density_fade(float rank, float count_morph)
{
    float x = count_morph - rank;
    if (x >= 1.0f) return 1.0f;
    if (x <= 0.0f) return 0.0f;
    return x * x * (3.0f - 2.0f * x);
}

/* Density wander, in T2 only: below its full set, the taps that sound don't
 * stay put, or the sparse long stage would repeat one fixed rhythm through
 * the whole tail (T0 and T1 just thin, in the fixed order of compute_tap_ranks,
 * which keeps them velvet); but every pattern it moves through is velvet,
 * one tap to each stretch of the stage. A stage thinned to
 * K taps is split into K equal stretches of its span, velvet's cells, and
 * each stretch sounds one of the stage's taps that fall in it (the full set
 * has several to each at these counts; a stretch with none is left as the
 * gap velvet allows). Each stretch moves on to another of its own taps every
 * WANDER_TURN_S, chosen at random, the stretches at staggered times; the
 * change is an equal-power crossfade between the two taps of that stretch
 * over the last WANDER_FADE_S of the turn. Over the last tap's worth of
 * the count, up to the full set, the pattern fades into the full set, which
 * (no wander) is exactly as before. The energy compensation holds the level
 * as it goes. */
/* At full speed each stretch keeps a tap for WANDER_TURN_S, the last
 * WANDER_FADE_S of it fading to the next, so the tail keeps changing without
 * a tap being heard to switch. */
#define WANDER_TURN_S  5.0f
#define WANDER_FADE_S  2.0f
#define WANDER_ELIG    0.1f     /* energy, relative to the loudest tap: -10 dB */
#define WANDER_MAX_TAPS  (MAX_T0_TAPS > MAX_T1_TAPS ? (MAX_T0_TAPS > MAX_T2_TAPS ? MAX_T0_TAPS : MAX_T2_TAPS) \
                                                   : (MAX_T1_TAPS > MAX_T2_TAPS ? MAX_T1_TAPS : MAX_T2_TAPS))
/* Any change of pattern other than a turn (the knob moving the count, Size
 * moving the span) is smoothed per tap over about WANDER_GLIDE_S, so it is a
 * short fade rather than a tap switched in a couple of milliseconds, heard
 * as zipping while Density turns. And the number of stretches changes in
 * whole steps, with WANDER_HYST of a tap's hysteresis, so a slow turn changes
 * it once per step rather than back and forth. */
#define WANDER_GLIDE_S   0.12f
#define WANDER_HYST      0.75f
typedef struct {
    float pos;                 /* each stretch's progress through its taps, in taps */
    uint32_t seed;             /* which taps the stretches pick (L and R differ) */
    uint32_t last_block;       /* block of the last recompute */
    int stretches;             /* 0 until the wander first runs */
    float w[WANDER_MAX_TAPS];  /* each tap's weight, gliding */
} WanderState;
static WanderState t2_wander, t2_wander_r;
static uint8_t t0TapOrder[MAX_T0_TAPS], t1TapOrder[MAX_T1_TAPS], t2TapOrder[MAX_T2_TAPS];  /* tap index by rank */
static uint32_t wander_blocks;      /* blocks run, for the wander's clock */
static uint8_t density_wandering;   /* any stage wandering: keep its gains recomputing */

/* The wander, on (0 holds the taps still). As taps come and go, each stage's
 * total tap energy holds: the level compensation carries whichever taps are
 * sounding, with the stage make-up (stage_fit) taking what a Q15 gain cannot
 * hold. */
#ifndef DENSITY_WANDER
#define DENSITY_WANDER 1
#endif

/* 1 from half the taps down, falling to 0 at the full set. */
static inline float wander_amount(float count_morph, int n)
{
#if !DENSITY_WANDER
    (void)count_morph; (void)n;
    return 0.0f;
#endif
    float x = ((float)n - count_morph) / (0.5f * (float)n);
    if (x < 0.0f) x = 0.0f; else if (x > 1.0f) x = 1.0f;
    return x;
}

static void build_tap_order(const float *ranks, uint8_t *order, int n)
{
    for (int t = 0; t < n; t++) {
        int r = (int)ranks[t];
        if (r >= 0 && r < n) order[r] = (uint8_t)t;
    }
}

/* A stretch's choice of tap for its v-th turn: any of its n, at random. */
static inline uint32_t wander_pick(uint32_t seed, uint32_t run, uint32_t v, uint32_t n)
{
    uint32_t h = seed ^ (run * 0x9E3779B1u) ^ (v * 0x85EBCA77u);
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return h % n;
}

/* Thins pre[] (the stage's taps before Density, in time order, at offsets
 * off[]) to the count, with the wander if ws is given, and returns the
 * thinned energy. */
static float density_apply(float *pre, const uint32_t *off, const float *ranks, const uint8_t *order, int n,
                           float count_morph, WanderState *ws)
{
    (void)order;
    if (!ws) {
        float thin = 0.0f;
        for (int t = 0; t < n; t++) {
            pre[t] *= density_fade(ranks[t], count_morph);
            thin += pre[t] * pre[t];
        }
        return thin;
    }
    const float speed = wander_amount(count_morph, n);
    const float dt = (float)(wander_blocks - ws->last_block) * ((float)REVERB_BLOCK / (float)REVERB_FS_HZ);
    ws->last_block = wander_blocks;
    float thin = 0.0f;
    if (count_morph >= (float)n || !DENSITY_WANDER) {
        ws->stretches = 0;
        for (int t = 0; t < n; t++) {
            pre[t] *= density_fade(ranks[t], count_morph);
            thin += pre[t] * pre[t];
        }
        return thin;
    }

    /* The taps in the span now (Size leaves the rest silent), and the count
     * as a share of them. */
    uint8_t idx[WANDER_MAX_TAPS];
    int ne = 0;
    for (int t = 0; t < n; t++) if (pre[t] != 0.0f) idx[ne++] = (uint8_t)t;
    if (ne == 0) return 0.0f;
    float K = count_morph / (float)n * (float)ne;
    if (K > (float)ne) K = (float)ne;
    if (K < 1.0f) K = 1.0f;
    const int entering = (ws->stretches == 0);
    if (entering || fabsf(K - (float)ws->stretches) > WANDER_HYST) ws->stretches = (int)(K + 0.5f);
    if (ws->stretches < 1) ws->stretches = 1;
    if (ws->stretches > ne) ws->stretches = ne;
    K = (float)ws->stretches;

    ws->pos += speed * dt * (1.0f / WANDER_TURN_S);
    /* The fade as a share of a stretch's time on a tap (which lengthens as
     * the wander slows), so it stays WANDER_FADE_S long. */
    float fadeShare = (speed > 0.0f) ? WANDER_FADE_S * speed * (1.0f / WANDER_TURN_S) : 0.5f;
    if (fadeShare > 0.5f) fadeShare = 0.5f;
    if (ws->pos >= 65536.0f) ws->pos -= 65536.0f;

    float w[sizeof idx / sizeof idx[0]];
    for (int i = 0; i < ne; i++) w[i] = 0.0f;
    /* K equal stretches of the span, which runs from the first tap to one
     * average gap past the last. */
    const float first = (float)off[idx[0]];
    float span = (float)off[idx[ne - 1]] - first;
    if (ne > 1) span *= (float)ne / (float)(ne - 1);
    const float perUnit = (span > 0.0f) ? K / span : 0.0f;
    int start = 0;
    while (start < ne) {
        const uint32_t run = (uint32_t)(((float)off[idx[start]] - first) * perUnit);
        int end = start + 1;
        while (end < ne && (uint32_t)(((float)off[idx[end]] - first) * perUnit) == run) end++;
        /* The stretch's taps that can carry it: those within WANDER_ELIG of
         * its loudest (a tap deep in the stretch's decay, made up to the
         * stage's level, would need more than the compensation gives, and
         * the stage would drop out). */
        uint8_t cand[WANDER_MAX_TAPS];
        float emax = 0.0f;
        for (int i = start; i < end; i++) { const float e = pre[idx[i]] * pre[idx[i]]; if (e > emax) emax = e; }
        uint32_t len = 0;
        for (int i = start; i < end; i++) if (pre[idx[i]] * pre[idx[i]] >= WANDER_ELIG * emax) cand[len++] = (uint8_t)i;
        if (len == 1) {
            w[cand[0]] = 1.0f;
        } else {
            /* Staggered: each stretch is a golden-ratio share of a turn apart. */
            float stag = (float)run * 0.618034f + (float)(ws->seed & 0xFFu) * (1.0f / 256.0f);
            stag -= floorf(stag);
            const float ph = ws->pos + stag;
            const uint32_t v = (uint32_t)ph;
            float x = (ph - (float)v - (1.0f - fadeShare)) / fadeShare;
            if (x < 0.0f) x = 0.0f; else if (x > 1.0f) x = 1.0f;
            x = x * x * (3.0f - 2.0f * x);
            const uint32_t cur = wander_pick(ws->seed, run, v, len);
            uint32_t nxt = wander_pick(ws->seed, run, v + 1u, len);
            if (nxt == cur) nxt = (cur + 1u) % len;
            const float a = 0.5f * PI_F * x;
            w[cand[cur]] += cosf(a);
            w[cand[nxt]] += sinf(a);
        }
        start = end;
    }
    /* Over the last tap of the count, into the full set. */
    const float top = count_morph - (float)(n - 1);
    float full[sizeof idx / sizeof idx[0]];
    for (int t = 0; t < n; t++) full[t] = 0.0f;
    for (int i = 0; i < ne; i++) full[idx[i]] = w[i];
    const float glide = 1.0f - expf(-dt * (1.0f / WANDER_GLIDE_S));
    for (int t = 0; t < n; t++) {
        float g = full[t];
        if (top > 0.0f) g += top * (1.0f - g);
        if (entering) ws->w[t] = (pre[t] != 0.0f) ? 1.0f : 0.0f;   /* from the full set */
        ws->w[t] += glide * (g - ws->w[t]);
        pre[t] *= ws->w[t];
    }
    for (int t = 0; t < n; t++) thin += pre[t] * pre[t];
    return thin;
}

void velvet_reverb_regenerate_taps(void)
{
    /* T0 — early. Taper baked in (cosine tail fade). */
    generate_mono_stage(t0TapOffsets, t0TapGains, &t0TapCount,
                        MAX_T0_TAPS, T0_WINDOW_MAX_SAMPLES / REVERB_RATE_MULT, /*apply_taper=*/0);

    /* T1 — middle. No taper; decay envelope applied dynamically per-block. */
    generate_mono_stage(t1TapOffsets, t1TapGains, &t1TapCount,
                        MAX_T1_TAPS, T1_WINDOW_MAX_SAMPLES / REVERB_RATE_MULT, /*apply_taper=*/0);

    /* T2 — stereo: independent L/R offsets, shared gain magnitude, no taper. */
    const int t2Max = T2_DURATION_MAX_SAMPLES / REVERB_RATE_MULT;
    int t2Num = MAX_T2_TAPS;
    int usable2 = t2Max - TAP_MIN_OFFSET;
    if (usable2 < TAP_MIN_GRID) usable2 = TAP_MIN_GRID;
    int maxTaps2 = usable2 / TAP_MIN_GRID;
    if (t2Num > maxTaps2) t2Num = maxTaps2;

    int gridSize2 = usable2 / t2Num;
    int jitterRange2 = gridSize2 - TAP_MIN_GRID;
    if (jitterRange2 < 1) jitterRange2 = 1;

    float norm2 = 1.0f / sqrtf((float)t2Num);

    for (int j = 0; j < t2Num; j++) {
        /* Tap 0 pinned to TAP_MIN_OFFSET (see generate_mono_stage); R's is moved
         * on by REVERB_T2_R_TAP0_MS below. */
        int jitL = TAP0_PINNED(j) ? 0 : (int)(xorshift32() % (uint32_t)jitterRange2);
        int jitR = TAP0_PINNED(j) ? 0 : (int)(xorshift32() % (uint32_t)jitterRange2);
        int posL = TAP_MIN_OFFSET + j * gridSize2 + jitL;
        int posR = TAP_MIN_OFFSET + j * gridSize2 + jitR;
        if (posL >= t2Max) posL = TAP_MIN_OFFSET + j * gridSize2;
        if (posR >= t2Max) posR = TAP_MIN_OFFSET + j * gridSize2;
        if (j == 0) posR += (int)(REVERB_T2_R_TAP0_MS * 24.0f);

        float sign = (xorshift32() & 1) ? 1.0f : -1.0f;

        t2TapOffsetsL[j] = (uint32_t)(posL & ~1);
        t2TapOffsetsR[j] = (uint32_t)(posR & ~1);
        t2TapGains[j] = (int16_t)(sign * norm2 * TAP_GAIN_HEADROOM * 32767.0f);
    }
    t2TapCount = t2Num;
#if REVERB_T2_R_SIGNS
    {
        uint32_t s = 0x2545F491u;   /* own sequence: leaves the layout's draws alone */
        for (int j = 0; j < t2Num; j++) {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            t2FlipR[j] = (j > 0) && (s & 1u);
        }
    }
#endif

    compute_tap_ranks(t0TapOffsets,  t0TapRanks, t0TapCount);
    compute_tap_ranks(t1TapOffsets,  t1TapRanks, t1TapCount);
    compute_tap_ranks(t2TapOffsetsL, t2TapRanks, t2TapCount);   /* L sets the order; R shares the gain */
    build_tap_order(t0TapRanks, t0TapOrder, t0TapCount);
    build_tap_order(t1TapRanks, t1TapOrder, t1TapCount);
    build_tap_order(t2TapRanks, t2TapOrder, t2TapCount);

    /* Each stage's reference energy for the loudness-preservation gain comp:
     * that of its taps inside Size's midpoint window (the DLD's), so every
     * Size plays at the level the DLD's does. Computed once per IR regen,
     * in 24 kHz samples like the offsets here. */
    {
        const float r0 = reverb_macro_params[MP_T0_WINDOW].bound_mid * 24.0f;
        const float r1 = reverb_macro_params[MP_T1_WINDOW].bound_mid * 24.0f;
        const float r2 = reverb_macro_params[MP_T2_DURATION].bound_mid * 24000.0f;
        /* The generators normalise each stage over all its taps; rescale so
         * the taps inside the reference window carry the DLD's energy (its
         * taps at 1/sqrt(count) each), and Size's level comp starts there. */
        int n0 = 0, n1 = 0, n2 = 0;
        for (int i = 0; i < t0TapCount; i++) n0 += (float)t0TapOffsets[i] < r0;
        for (int i = 0; i < t1TapCount; i++) n1 += (float)t1TapOffsets[i] < r1;
        for (int i = 0; i < t2TapCount; i++) n2 += (float)t2TapOffsetsL[i] < r2;
        const float k0 = n0 > 0 ? sqrtf((float)t0TapCount / (float)n0) : 1.0f;
        const float k1 = n1 > 0 ? sqrtf((float)t1TapCount / (float)n1) : 1.0f;
        const float k2 = n2 > 0 ? sqrtf((float)t2TapCount / (float)n2) : 1.0f;
        for (int i = 0; i < t0TapCount; i++) t0TapGains[i] = (int16_t)floorf((float)t0TapGains[i] * k0 + 0.5f);
        for (int i = 0; i < t1TapCount; i++) t1TapGains[i] = (int16_t)floorf((float)t1TapGains[i] * k1 + 0.5f);
        for (int i = 0; i < t2TapCount; i++) t2TapGains[i] = (int16_t)floorf((float)t2TapGains[i] * k2 + 0.5f);
        t0_full_energy = t1_full_energy = t2_full_energy = 0.0f;
        for (int i = 0; i < t0TapCount; i++)
            if ((float)t0TapOffsets[i] < r0) { const float g = (float)t0TapGains[i]; t0_full_energy += g * g; }
        for (int i = 0; i < t1TapCount; i++)
            if ((float)t1TapOffsets[i] < r1) { const float g = (float)t1TapGains[i]; t1_full_energy += g * g; }
        /* T2's comp works from its decay-shaped energy (see
         * recompute_eff_gains_t2), and at the reference window lands on
         * T2_COMP_AT_MID, where the DLD's settles. */
        for (int i = 0; i < t2TapCount; i++)
            if ((float)t2TapOffsetsL[i] < r2) {
                const float g = (float)t2TapGains[i] * expf(-9.2103404f * (float)t2TapOffsetsL[i] / r2);
                t2_full_energy += g * g;
            }
        t2_full_energy *= T2_COMP_AT_MID * T2_COMP_AT_MID;
    }

    /* Per-tap ladders (one rung each, see build_single_rung_ladders). */
    build_single_rung_ladders(t0TapOffsets, NULL, t0TapCount, t0TapLadder, NULL, t0TapLadderCount);
    build_single_rung_ladders(t1TapOffsets, NULL, t1TapCount, t1TapLadder, NULL, t1TapLadderCount);
    build_single_rung_ladders(t2TapOffsetsL, t2TapOffsetsR, t2TapCount,
                              t2TapLadderL, t2TapLadderR, t2TapLadderCount);

#if REVERB_RATE_MULT > 1
    scale_offsets_to_rate(t0TapOffsets,  MAX_T0_TAPS);
    scale_offsets_to_rate(t1TapOffsets,  MAX_T1_TAPS);
    scale_offsets_to_rate(t2TapOffsetsL, MAX_T2_TAPS);
    scale_offsets_to_rate(t2TapOffsetsR, MAX_T2_TAPS);
    scale_offsets_to_rate(t0TapLadder,  MAX_T0_TAPS * MAX_LADDER_LEVELS);
    scale_offsets_to_rate(t1TapLadder,  MAX_T1_TAPS * MAX_LADDER_LEVELS);
    scale_offsets_to_rate(t2TapLadderL, MAX_T2_TAPS * MAX_LADDER_LEVELS);
    scale_offsets_to_rate(t2TapLadderR, MAX_T2_TAPS * MAX_LADDER_LEVELS);
#endif
    for (int t = 0; t < MAX_T1_TAPS; t++)
        t1OnsetBoost[t] = 1.0f + T1_ONSET_GAIN *
            expf(-(float)t1TapOffsets[t] * (1000.0f / (T1_ONSET_MS * (float)REVERB_FS_HZ)));

    /* T0 per-tap Lexicon-LFO phase table: each tap gets a fixed golden-angle
     * decorrelated phase so the early cluster shimmers diffusely. */
    for (int k = 0; k < MAX_T0_TAPS; k++) {
        float ph = (float)k * GOLDEN_ANGLE_RAD;
        t0TapModCos[k] = cosf(ph);
        t0TapModSin[k] = sinf(ph);
    }
}

/* Per-stage fade-zone minimum samples — keeps the relocation bell smooth even
 * at very short window settings. T2's floor is larger because its window is. */
#define T0_FADE_MIN  (192.0f * (float)REVERB_RATE_MULT)
#define T1_FADE_MIN  (192.0f * (float)REVERB_RATE_MULT)
#define T2_FADE_MIN  (512.0f * (float)REVERB_RATE_MULT)
/* Relocation crossfade width as a fraction of the window. */
#define FADE_WINDOW_FRAC  0.06f

/* ==== Init ==== */
/* The sustain loops' drift LFOs (see loop_drift_step). */
static uint32_t loop_lfo_ph[4];
static float loop_lfo_rw[4], loop_lfo[4];
static uint32_t loop_lfo_seed = 0x2545F491u;

void velvet_reverb_init(void)
{
    /* Apply default macro positions so the reverb_* targets reflect the
     * macro-derived values rather than the raw initialisers at the top of
     * this file. Density and Decay start at 0; Tone at 0.83, the position the
     * macro bounds were captured at. */
    velvet_reverb_recompute_macros();

    {
        uint32_t *p = (uint32_t *)t0_ring;
        uint32_t *end = (uint32_t *)((uint8_t *)t0_ring + T0_RING_SAMPLES * 2);
        while (p < end) *p++ = 0;
    }
    {
        uint32_t *p = (uint32_t *)t1_ring;
        uint32_t *end = (uint32_t *)((uint8_t *)t1_ring + T1_RING_SAMPLES * 2);
        while (p < end) *p++ = 0;
    }
    {
        uint32_t *p = (uint32_t *)T2_RING_BASE;
        uint32_t *end = (uint32_t *)(T2_RING_BASE + T2_RING_SAMPLES * 2);
        while (p < end) *p++ = 0;
    }
    memset(t0_ring_r, 0, sizeof t0_ring_r);
    memset(t1_ring_r, 0, sizeof t1_ring_r);
    memset(t2_ring_r, 0, T2_RING_SAMPLES * sizeof(int16_t));

    for (int i = 0; i < REVERB_BLOCK; i++) {
        accT0[i] = 0; accT1[i] = 0; accL[i] = 0; accR[i] = 0;
        accT0_r[i] = 0; accT1_r[i] = 0;
    }
    for (int b = 0; b < REVERB_IN_NBUF; b++)
        for (int i = 0; i < REVERB_BLOCK; i++) in_ring[b][i] = in_ring_r[b][i] = 0;
    for (int b = 0; b < REVERB_OUT_NBUF; b++) {
        for (int i = 0; i < REVERB_OUT_BLOCK; i++) {
            outL[b][i] = 0; outR[b][i] = 0;
        }
    }
    prev_out_L = 0; prev_out_R = 0;
    for (int p = 0; p < 2; p++) {
        for (int i = 0; i < REVERB_OUT_BLOCK; i++) {
            lim_buf_L[p][i] = 0.0f; lim_buf_R[p][i] = 0.0f;
        }
    }
    lim_phase = 0;
    lim_gain = 1.0f; lim_gain_target = 1.0f;
    for (int i = 0; i < LIM_HOLD_BLOCKS; i++) lim_hold[i] = 1.0f;
    lim_hold_idx = 0;
    push_acc = push_acc_r = 0; push_phase = 0;
    in_wr_seq = 0; in_rd_seq = 0;
    input_fill_idx = 0;
    input_ready = in_ring[0];
    input_ready_r = in_ring_r[0];
    for (uint32_t i = 0; i < INPUT_DELAY_SAMPLES; i++) input_delay[i] = input_delay_r[i] = 0;
    id_w = 0; id_left = 0;
    id_to = (int32_t)(id_target + 0.5f);
    id_pos = id_smooth = (float)id_to;
    input_queue_max = 0;
    input_drop_count = 0;
    blocks_produced = 0;
    rd_seq_L = 0; rd_seq_R = 0;
    reader_buf_L = 0; reader_buf_R = 0;
    reader_started_L = reader_started_R = 0;
    out_idx_L = 0; out_idx_R = 0;

    /* Biquads */
    hpf_L.z1 = hpf_L.z2 = 0.0f;  hpf_R.z1 = hpf_R.z2 = 0.0f;
    lpf_L.z1 = lpf_L.z2 = 0.0f;  lpf_R.z1 = lpf_R.z2 = 0.0f;
    biquad_hpf(&hpf_L, reverb_hpf_hz, (float)REVERB_CODEC_HZ);
    biquad_copy_coeffs(&hpf_R, &hpf_L);
    TONE_LPF(&lpf_L, reverb_lpf_hz, (float)REVERB_CODEC_HZ);
    biquad_copy_coeffs(&lpf_R, &lpf_L);
    last_hpf_hz = reverb_hpf_hz;
    last_lpf_hz = reverb_lpf_hz;
    oct_init();
    lpf_hz_morph = reverb_lpf_hz;
    hpf_hz_morph = reverb_hpf_hz;

    /* ---- Pre-delay sustain engine ---- */
    for (uint32_t i = 0; i < PRE_DELAY_LINE_SAMPLES; i++) {
        predelay_a[i] = predelay_b[i] = 0.0f;
        predelay_a_r[i] = predelay_b_r[i] = 0.0f;
    }
    predelay_wh = 0;
    pre_mod_phase_a = 0.0f; pre_mod_phase_b = PI_F;
    pre_hp_a = pre_hp_b = pre_hp_a_r = pre_hp_b_r = 0.0f;
    /* All zero is a valid cubic window too: the lines it describes were just
     * zeroed. */
    memset(&pre_rd_a, 0, sizeof pre_rd_a);
    memset(&pre_rd_b, 0, sizeof pre_rd_b);
    memset(&pre_rd_a_r, 0, sizeof pre_rd_a_r);
    memset(&pre_rd_b_r, 0, sizeof pre_rd_b_r);
    pre_rd_a.i = pre_rd_b.i = pre_rd_a_r.i = pre_rd_b_r.i = INT32_MIN;
    {
        static const float ph0[4] = { 0.1f, 0.45f, 0.7f, 1.3f };   /* Cosmos's starting phases */
        for (int j = 0; j < 4; j++) {
            loop_lfo_ph[j] = (uint32_t)(ph0[j] * 2147483648.0f);
            loop_lfo_rw[j] = 0.0f;
            loop_lfo[j] = ph0[j] < 1.0f ? ph0[j] : 2.0f - ph0[j];
        }
        loop_lfo_seed = 0x2545F491u;
    }
    duck_env = 0.0f;
    fb_morph = -1.0f;
    pre_low_shelf_a.z1 = pre_low_shelf_a.z2 = 0.0f;
    pre_high_shelf_a.z1 = pre_high_shelf_a.z2 = 0.0f;
    pre_low_shelf_b.z1 = pre_low_shelf_b.z2 = 0.0f;
    pre_high_shelf_b.z1 = pre_high_shelf_b.z2 = 0.0f;
    biquad_lowshelf(&pre_low_shelf_a,  reverb_fb_low_shelf_hz,  reverb_fb_low_shelf_db,  (float)REVERB_FS_HZ);
    biquad_copy_coeffs(&pre_low_shelf_b, &pre_low_shelf_a);
    biquad_highshelf(&pre_high_shelf_a, reverb_fb_high_shelf_hz, reverb_fb_high_shelf_db, (float)REVERB_FS_HZ);
    biquad_copy_coeffs(&pre_high_shelf_b, &pre_high_shelf_a);
    pre_low_shelf_a_r = pre_low_shelf_a;   pre_low_shelf_b_r = pre_low_shelf_b;
    pre_high_shelf_a_r = pre_high_shelf_a; pre_high_shelf_b_r = pre_high_shelf_b;
    last_fb_ls_hz = reverb_fb_low_shelf_hz;  last_fb_ls_db = reverb_fb_low_shelf_db;
    last_fb_hs_hz = reverb_fb_high_shelf_hz; last_fb_hs_db = reverb_fb_high_shelf_db;
    fb_hs_hz_morph = reverb_fb_high_shelf_hz;
    pre_hp_coeff   = 1.0f - expf(-TWO_PI_F * PRE_HP_FREQ_HZ / (float)REVERB_FS_HZ);
    duck_atk_coeff = 1.0f - expf(-1.0f / (DUCK_ATK_SEC * (float)REVERB_FS_HZ));
    t0_tap_mod_phase = 0.0f;
    t0_mod_depth_last = -1.0f;

    /* Morph state — initialise to targets so the first block produces a
     * meaningful output (avoids "fading up from zero" silence on boot). */
    t0_count_morph = reverb_t0_count_target;
    t1_count_morph = reverb_t1_count_target;
    t2_count_morph = reverb_t2_count_target;
    memset(&t2_wander, 0, sizeof t2_wander);
    memset(&t2_wander_r, 0, sizeof t2_wander_r);
    t2_wander.seed = 0x51ED270Bu;
    t2_wander_r.seed = 0xA3C59AC3u;   /* R picks its own taps */
    wander_blocks = 0;
    density_wandering = 0;
    t0_window_morph = reverb_t0_window_ms_target * (float)REVERB_FS_HZ / 1000.0f;
    t1_window_morph = reverb_t1_window_ms_target * (float)REVERB_FS_HZ / 1000.0f;
    t2_window_morph = reverb_t2_duration_s_target * (float)REVERB_FS_HZ;
    t2_fade_morph = reverb_t2_fade_s_target * (float)REVERB_FS_HZ;
    t0_tap_comp_morph = 1.0f;
    t1_tap_comp_morph = 1.0f;
    t2_tap_comp_morph = 1.0f;
    predelay_a_morph = reverb_predelay_a_s * (float)REVERB_FS_HZ;
    predelay_b_morph = reverb_predelay_b_s * (float)REVERB_FS_HZ;
    memset(&pre_len_a, 0, sizeof pre_len_a);
    memset(&pre_len_b, 0, sizeof pre_len_b);
    pre_len_a.time = predelay_a_morph;
    pre_len_b.time = predelay_b_morph;
    pre_len_a_r = pre_len_a; pre_len_b_r = pre_len_b;

    /* Pre-compute the exponential envelope LUT — exp(-9.21034 × frac) gives
     * an 80 dB drop over the window. Linearly interpolated at lookup. */
    for (int i = 0; i < ENV_FRAC_BINS; i++) {
        float frac = (float)i / (float)(ENV_FRAC_BINS - 1);
        exp_env_lut[i] = expf(-9.21034f * frac);
    }

    velvet_reverb_regenerate_taps();

#ifndef VELVET_REVERB_HOST
    dwt_init();
#endif
#if !defined(VELVET_REVERB_HOST) && !DMA2_DISABLED_FOR_DIAGNOSTIC
    RCC->AHB1ENR |= RCC_AHB1ENR_DMA2EN;
    __DSB();
    DMA2_Stream1->CR = 0;
    while (DMA2_Stream1->CR & DMA_SxCR_EN) {}
    DMA2->LIFCR = 0x00000F40;
    /* 32-bit transfers (PSIZE = MSIZE = word): half the bus transactions of
     * 16-bit ones. Valid because every T2 window starts at an even sample: block
     * indices are multiples of REVERB_BLOCK and tap offsets are forced even, so
     * source and destination are always 4-byte aligned. No bursts: a 4-beat burst
     * must not cross a 1 KB boundary, and an arbitrary window can. */
    DMA2_Stream1->CR = T2_DMA_CR;
    DMA2_Stream1->FCR = DMA_SxFCR_DMDIS | DMA_SxFCR_FTH;
    /* T2 gather chain interrupt. The grouping is read rather than assumed:
     * passing a literal group number encodes the wrong priority. */
    NVIC_SetPriority(DMA2_Stream1_IRQn, NVIC_EncodePriority(NVIC_GetPriorityGrouping(), 2, 0));
    NVIC_EnableIRQ(DMA2_Stream1_IRQn);
#endif
    t2_gather_state = T2G_INVALID;
    pd_init();

    /* Seed effGains arrays so the first 3 blocks (before recompute round-
     * robin catches up) produce meaningful output rather than silence. */
    for (int i = 0; i < t0TapCount; i++) effGains_t0[i] = t0TapGains[i];
    for (int i = 0; i < t1TapCount; i++) effGains_t1[i] = t1TapGains[i];
    for (int i = 0; i < t2TapCount; i++) effGains_t2[i] = t2TapGains[i];
#if REVERB_T2_R_SIGNS
    for (int i = 0; i < t2TapCount; i++) effGains_t2R[i] = t2FlipR[i] ? (int16_t)-t2TapGains[i] : t2TapGains[i];
#else
    for (int i = 0; i < t2TapCount; i++) effGains_t2R[i] = t2TapGains[i];
#endif

    /* Seed effOff arrays from each tap's main offset (rung 0). Without this,
     * the first block reads from offset 0 (near-zero delay = feedback path)
     * for every tap. */
    for (int i = 0; i < t0TapCount; i++)
        effOffT0[i] = t0TapOffsets[i];
    for (int i = 0; i < t1TapCount; i++)
        effOffT1[i] = t1TapOffsets[i];
    for (int i = 0; i < t2TapCount; i++) {
        effOffT2L[i] = t2TapOffsetsL[i];
        effOffT2R[i] = t2TapOffsetsR[i];
    }
}

/* ==== Saturate helpers ==== */

/* Soft knee in full-scale units (1.0 == full scale). Transparent (unity, like
 * a float path) for |x| <= SOFT_KNEE_T; above it a rational knee that is
 * C1-continuous at the threshold (value and slope match the linear region) and
 * asymptotes to ±1.0. Replaces the old hard clip so overs round off musically
 * instead of clipping harshly. */
static inline float soft_clip_knee(float x, float t)
{
    float a = (x < 0.0f) ? -x : x;
    if (a <= t) return x;
    float k = 1.0f - t;
    float e = a - t;
    float y = t + k * e / (k + e);   /* -> 1.0, slope 1 at threshold */
    return (x < 0.0f) ? -y : y;
}

static inline float soft_clip_unit(float x) { return soft_clip_knee(x, SOFT_KNEE_T); }

/* Rounding into the int16 rings. The cascade's rings are int16, and the
 * conversions used to truncate toward zero: anything under one step vanished
 * and quiet signals were crushed into steps, so a reverb tail decayed through
 * an audible staircase into a gate (on the Beads, where the wet output is made
 * up ×4, a 'decimated' grain under every tail). Each write into a ring now
 * rounds with triangular dither of +-1 step, so a tail fades smoothly into a
 * low, even hiss instead. */
static uint32_t dither_state = 0x9E3779B9u;
static inline float tpdf_dither(void)
{
    uint32_t x = dither_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    const uint32_t a = x;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    dither_state = x;
    return (float)((int32_t)(a >> 9) + (int32_t)(x >> 9)) * (1.0f / 8388608.0f) - 1.0f;   /* -1..1 */
}
/* An int16-scale value into int16: dithered, rounded, clamped. */
static inline int16_t to_ring16(float y)
{
    y += tpdf_dither();
    int32_t i = (int32_t)(y + (y >= 0.0f ? 0.5f : -0.5f));
    if (i > 32767) i = 32767; else if (i < -32768) i = -32768;
    return (int16_t)i;
}

/* Q30 accumulator (Q15 sample × Q15 gain, summed) -> int16, soft-clipped at
 * full scale. 2^30 = full scale in the Q30 domain. Used at the stage bridges
 * and the output read. */
static inline int16_t soft_saturate_q15(int32_t x)
{
    float v = (float)x * (1.0f / 1073741824.0f);   /* 2^30 == full scale */
    return (int16_t)(soft_clip_unit(v) * 32767.0f);
}
/* The same, with a stage's make-up gain (see stage_fit). */
static inline int16_t soft_saturate_q15_g(int32_t x, float g)
{
    float v = (float)x * (g * (1.0f / 1073741824.0f));
    return to_ring16(soft_clip_unit(v) * 32767.0f);
}

/* Hard int16 clamp — kept for coefficient clamping (effGains) and anywhere a
 * true hard limit is wanted. */
static inline int16_t clamp_f_to_int16(float v)
{
    if (v > 32767.0f) return 32767;
    if (v < -32767.0f) return -32767;
    return (int16_t)v;
}

/* Soft int16 clamp for signal-path values already in int16 scale (recirc
 * feedback writes, final output). Same knee as soft_saturate_q15. */
static inline int16_t soft_clip_int16(float v)
{
    return to_ring16(soft_clip_unit(v * (1.0f / 32768.0f)) * 32767.0f);
}

/* ==== Output limiter ====
 *
 * The wet signal has about 7 dB more peak gain than the output can carry: the
 * pre-delay feedback loop roughly doubles the peak of a sustained input and the
 * stereo width adds a further 1.8 dB, so a full-scale input produces peaks near
 * 2.2x full scale. Its RMS, though, sits 8-12 dB BELOW full scale — this is a
 * crest-factor problem, not a level problem, and only 0.01-1.5 % of samples are
 * involved. That distinction is what makes a limiter the right answer: turning
 * the reverb (or its input) down scales peak and average together and just makes
 * it quieter, whereas limiting takes the peaks down and leaves the level alone.
 *
 * Why it matters more than ordinary clipping would: the cascade runs at 24 kHz,
 * so harmonics generated above 4 kHz land past Nyquist and fold back down as
 * broadband hash. That reads as noise rather than as distortion, and it became
 * audible once the allpass fractional delay let high frequencies sustain in the
 * loop, giving the clipper HF-rich material to work on.
 *
 * Design. The gain is derived from the peak of the block being computed but
 * applied to the PREVIOUS block, which buys a free block of look-ahead: the ramp
 * finishes exactly as the block that demanded it starts, so the gain is already
 * down when the peak arrives and there is no overshoot to distort. Costs one
 * block (0.67 ms) of latency. Attack is therefore one block; release is eased so
 * a single peak does not duck the tail audibly. The gain is stereo-linked, since
 * independent channel gains would modulate the stereo image.
 *
 * Ceiling sits below the output knee so that in normal operation the knee is
 * never reached and the output path is completely transparent — the knee is left
 * in place only to absorb the small overshoot a biquad step response can add on
 * top of the peak measured before it. */
/* Wet output gain on the Beads, ahead of the limiter so loud tails are
 * limited rather than clipped. It was 2 (twice the DLD's level), but the dry
 * signal is added to the wet after this and with a hot input the sum went
 * over full scale; the DLD's own level leaves room for both. */
#ifndef REVERB_OUT_GAIN
#define REVERB_OUT_GAIN 1.0f
#endif
#define LIM_CEILING     0.95f    /* of full scale; peak is measured post-biquad  */
#define SOFT_KNEE_OUT   0.97f    /* output-only knee; bridges keep SOFT_KNEE_T   */
/* One-pole release at the 1500 Hz block rate. This is the whole trade: peaks in
 * a dense reverb tail arrive roughly once per block, so a slow release never gets
 * to recover between them and degenerates into a static gain cut — which is just
 * turning the reverb down, the one outcome we are trying to avoid. Too fast and
 * the gain itself modulates at audio rates and sprays sidebands. Swept in
 * test/host_test/clip_hash.c against both RMS retained and hash produced. */
/* 0.03 ~ 22 ms. Measured: at 0 dBFS input, 80 ms costs 3.3 dB of output RMS and
 * 22 ms costs 2.7 dB, while going faster than this starts letting peaks through
 * (the gain rises within the block being emitted) for very little further gain. */
#ifndef LIM_RELEASE
#define LIM_RELEASE     0.03f
#endif

static inline int16_t clip_out_int16(float v)
{
    const float y = soft_clip_knee(v * (1.0f / 32768.0f), SOFT_KNEE_OUT) * 32767.0f;
    return (int16_t)(y + (y >= 0.0f ? 0.5f : -0.5f));   /* rounded (the output is at full level) */
}

#ifdef VELVET_REVERB_HOST
/* Same conversion, tallied. Used only at the final output write so the count is
 * over output samples and not over every bridge as well. */
static inline int16_t clip_out_counted(float v)
{
    float u = v * (1.0f / 32768.0f);
    float a = u < 0.0f ? -u : u;
    if (a > host_dbg_final_peak) host_dbg_final_peak = a;
    host_dbg_final_n++;
    host_dbg_final_sumsq += (double)u * (double)u;
    if (a > SOFT_KNEE_OUT) host_dbg_final_knee++;
    if (a >= 1.0f)         host_dbg_final_over++;
    return clip_out_int16(v);
}
#define SOFT_CLIP_OUT clip_out_counted
#else
#define SOFT_CLIP_OUT clip_out_int16
#endif

/* ==== Per-block effGains recompute (ladder-driven) ====
 *
 * Each tap migrates one rung down each time the window shrinks past its
 * current rung. compute_tap_states picks the rung whose band contains the
 * current window and computes a bell-shaped window-gain that is 0 at every
 * rung boundary, so the discrete position swap happens at gain = 0 (no click).
 *
 * Recompute pass:
 *   1. compute_tap_states → effOff[], effWinGain[]
 *   2. for each tap: pre-comp effGain = baseGain × winGain × env
 *      (density is fixed at MAX — there is no per-tap density fade)
 *   3. sum (pre-comp²) → comp = sqrt(full_energy / energy), morphed
 *   4. final effGain = clamp_int16(pre-comp × comp_morph)
 */

/* Walk each tap's ladder to find which rung's band contains the current
 * window. Picks the smaller-offset rung (lower edge of the band) as the
 * effective read position, with a bell-shaped gain that is 0 at both the
 * upper and lower rung boundaries — so the discrete position swap at each
 * boundary happens in silence. */
static void compute_tap_states_mono(const uint32_t *ladder, const uint8_t *ladderCount,
                                    int count, float window, float fade,
                                    uint32_t *effOff, float *effWinGain)
{
    float invFade = (fade > 1e-6f) ? 1.0f / fade : 0.0f;
    for (int k = 0; k < count; k++) {
        const uint32_t *lad = &ladder[k * MAX_LADDER_LEVELS];
        int nLevels = ladderCount[k];
        float L0 = (float)lad[0];

        if (window >= L0) {
            /* Above the topmost rung — ramp in from 0 at window = L_0. */
            effOff[k] = lad[0];
            effWinGain[k] = smoothstep01((window - L0) * invFade);
            continue;
        }

        /* Walk rungs to find the band containing the window. */
        int foundI = -1;
        for (int i = 0; i < nLevels - 1; i++) {
            if (window > (float)lad[i + 1]) { foundI = i; break; }
        }
        if (foundI < 0) {
            /* Below the deepest rung — silent. */
            effOff[k] = lad[nLevels - 1];
            effWinGain[k] = 0.0f;
            continue;
        }
        float upper = (float)lad[foundI];
        float lower = (float)lad[foundI + 1];
        /* min(s(a), s(b)) == s(min(a, b)) exactly, since smoothstep never
         * decreases: one evaluation instead of two. */
        float dUp = (upper - window) * invFade;
        float dDn = (window - lower) * invFade;
        effOff[k] = lad[foundI + 1];
        effWinGain[k] = smoothstep01((dUp < dDn) ? dUp : dDn);
    }
}

/* T2 variant — band check uses the L ladder; R picks the matching rung
 * index. L and R ladders share their jitter at every rung, so they index
 * the same band, just with different absolute read positions for stereo
 * decorrelation. */
static void compute_tap_states_lr(const uint32_t *ladderL, const uint32_t *ladderR,
                                  const uint8_t *ladderCount, int count,
                                  float window, float fade,
                                  uint32_t *effOffL, uint32_t *effOffR,
                                  float *effWinGain)
{
    float invFade = (fade > 1e-6f) ? 1.0f / fade : 0.0f;
    for (int k = 0; k < count; k++) {
        const uint32_t *ladL = &ladderL[k * MAX_LADDER_LEVELS];
        const uint32_t *ladR = &ladderR[k * MAX_LADDER_LEVELS];
        int nLevels = ladderCount[k];
        float L0 = (float)ladL[0];

        if (window >= L0) {
            effOffL[k] = ladL[0];
            effOffR[k] = ladR[0];
            effWinGain[k] = smoothstep01((window - L0) * invFade);
            continue;
        }
        int foundI = -1;
        for (int i = 0; i < nLevels - 1; i++) {
            if (window > (float)ladL[i + 1]) { foundI = i; break; }
        }
        if (foundI < 0) {
            effOffL[k] = ladL[nLevels - 1];
            effOffR[k] = ladR[nLevels - 1];
            effWinGain[k] = 0.0f;
            continue;
        }
        float upper = (float)ladL[foundI];
        float lower = (float)ladL[foundI + 1];
        float dUp = (upper - window) * invFade;   /* see compute_tap_states_mono */
        float dDn = (window - lower) * invFade;
        effOffL[k] = ladL[foundI + 1];
        effOffR[k] = ladR[foundI + 1];
        effWinGain[k] = smoothstep01((dUp < dDn) ? dUp : dDn);
    }
}

/* Loudness compensation for Density: brings a thinned stage back to the
 * energy the full tap set has under the same window fades and envelope, so
 * Density changes the texture and not the level. Measuring before the
 * envelope instead let the sparse end come out ~6 dB loud, because the taps
 * it keeps are the early ones, which T2's decay envelope barely attenuates.
 * The window comp after it is computed from the full set, exactly as before,
 * so full Density is bit-identical to the reverb without the control.
 * Applied directly rather than morphed: the count it follows is already
 * smoothed. Capped at what a single tap needs; clamp_f_to_int16 holds the
 * resulting gains inside int16. */
/* A safety limit only: one tap standing in for a whole stage (116 in T0) can
 * need sqrt(116) ~ 11, and more if it is one of the stage's quieter taps
 * (the wander picks any tap of a run). Gains past what a Q15 gain holds are
 * carried by the stage's make-up instead (see stage_fit). */
#define DENSITY_COMP_MAX  40.0f
/* A stage's gains past what a Q15 gain holds (a few taps carrying the whole
 * stage at low Density) are scaled down to fit, and the stage's output is
 * scaled back up by the same in float where its accumulator is converted
 * (soft_saturate_q15_g). Without this the gains clipped and the stage went
 * quiet, by up to 9 dB, whenever a quiet tap was carrying it. Returns the
 * make-up, >= 1. */
static float t0_makeup = 1.0f, t1_makeup = 1.0f, t2_makeup_l = 1.0f, t2_makeup_r = 1.0f;

/* Gain ramps. A stage's gains are recomputed every few blocks (one stage a
 * block, in turn), and they used to jump there and hold for the block: when
 * Size or Density moved, each stage's output stepped every 2 ms, heard as
 * zipper noise. Now each recompute keeps the gains it replaces, and for the
 * next GAIN_RAMP samples (until that stage's next recompute) the stage runs
 * with both sets and crossfades from the old output to the new, so its gains
 * move in a straight line. The cost is a second pass of that stage's taps
 * while it ramps. */
#define GAIN_RAMP  (3 * REVERB_BLOCK)
static int16_t effGains_t0_prev[MAX_T0_TAPS] CCM_ATTR;
static int16_t effGains_t1_prev[MAX_T1_TAPS] CCM_ATTR;
static int16_t effGains_t2_prev[MAX_T2_TAPS] CCM_ATTR;
static int16_t effGains_t2R_prev[MAX_T2_TAPS] CCM_ATTR;
static float t0_makeup_prev = 1.0f, t1_makeup_prev = 1.0f, t2_makeup_l_prev = 1.0f, t2_makeup_r_prev = 1.0f;
static int32_t t0_ramp_left = 0, t1_ramp_left = 0, t2_ramp_left = 0;   /* samples */
/* The ramping stages' second outputs, from the old gains. */
static int32_t accT0_p[REVERB_BLOCK] CCM_ATTR, accT0_r_p[REVERB_BLOCK] CCM_ATTR;
static int32_t accT1_p[REVERB_BLOCK] CCM_ATTR, accT1_r_p[REVERB_BLOCK] CCM_ATTR;
static int32_t accL_p[REVERB_BLOCK] CCM_ATTR, accR_p[REVERB_BLOCK] CCM_ATTR;

/* How far through its ramp a stage is at sample n of this block, 0..1. */
static inline float ramp_frac(int32_t left, int n)
{
    return (float)(GAIN_RAMP - left + n + 1) * (1.0f / (float)GAIN_RAMP);
}
/* A Q30 accumulator at full scale 1.0 with its make-up, crossfaded from the
 * old gains' (p, mkp) to the new (x, mk) by f. */
static inline float acc_blend(int32_t p, float mkp, int32_t x, float mk, float f)
{
    return ((float)p * mkp + f * ((float)x * mk - (float)p * mkp)) * (1.0f / 1073741824.0f);
}
static float stage_fit(float *g, int n, float scale)
{
    float m = 0.0f;
    for (int t = 0; t < n; t++) { const float a = fabsf(g[t] * scale); if (a > m) m = a; }
    const float lim = 32000.0f;
    const float mk = m > lim ? m / lim : 1.0f;
    const float k = scale / mk;
    for (int t = 0; t < n; t++) g[t] *= k;
    return mk;
}

static inline float density_comp(float full, float thinned)
{
    if (thinned <= 1e-3f || full <= 1e-3f) return 1.0f;
    float c = sqrtf(full / thinned);
    return c > DENSITY_COMP_MAX ? DENSITY_COMP_MAX : c;
}

#define T0_TAPER_FRAC  0.15f
static void recompute_eff_gains_t0(void)
{
    memcpy(effGains_t0_prev, effGains_t0, sizeof effGains_t0_prev);
    t0_makeup_prev = t0_makeup;
    t0_ramp_left = GAIN_RAMP;
    int count = t0TapCount;
    if (count <= 0) return;

    float window = t0_window_morph;
    float fade = window * FADE_WINDOW_FRAC;
    if (fade < T0_FADE_MIN) fade = T0_FADE_MIN;

    float effWin[MAX_T0_TAPS];
    compute_tap_states_mono(t0TapLadder, t0TapLadderCount, count, window, fade,
                            effOffT0, effWin);

    /* T0 has no decay envelope, only a cosine taper over the last
     * T0_TAPER_FRAC of the window, which moves with it. */
    float pre[MAX_T0_TAPS];
    float energy = 0.0f, thinEnergy = 0.0f;
    const float taperStart = window * (1.0f - T0_TAPER_FRAC);
    const float invTaper = 1.0f / (window * T0_TAPER_FRAC);
    for (int t = 0; t < count; t++) {
        float g = (float)t0TapGains[t] * effWin[t];
        const float off = (float)effOffT0[t];
        if (off > taperStart) {
            float x = (window - off) * invTaper;
            if (x < 0.0f) x = 0.0f;
            g *= 0.5f * (1.0f - cosf(PI_F * x));
        }
        energy += g * g;
        pre[t] = g;
    }
    thinEnergy = density_apply(pre, effOffT0, t0TapRanks, t0TapOrder, count, t0_count_morph, NULL);
    const float densComp = density_comp(energy, thinEnergy);
    for (int t = 0; t < count; t++) pre[t] *= densComp;

    float comp = 1.0f;
    if (energy > 1e-3f && t0_full_energy > 1e-3f) {
        comp = sqrtf(t0_full_energy / energy);
        if (comp > TAP_COMP_CLAMP) comp = TAP_COMP_CLAMP;
    }
    t0_comp_goal = comp;
    t0_tap_comp_morph += ALPHA_WINDOW_MORPH * (comp - t0_tap_comp_morph);

    t0_makeup = stage_fit(pre, count, t0_tap_comp_morph);
    for (int t = 0; t < count; t++) effGains_t0[t] = clamp_f_to_int16(pre[t]);
}

/* T1's decay envelope, for small Sizes.
 *
 * T1 is the middle of the T0 -> T1 -> T2 cascade, and with flat taps a T1
 * window of a few hundred ms is longer than a short T2 fade itself:
 * convolved with T2 it holds the tail near its peak for a couple of hundred ms
 * and then lets T2's decay take over, a gated shape heard as a late swell and a
 * tail that cuts off (at Decay 0.35 the energy peaked at 240 ms, and the early
 * decay ran at half the rate of the late). Fading T1's taps across its window
 * brings the energy forward while the window stays long enough to keep T2's
 * copies overlapping.
 *
 * Full depth up to T1_ENV_FULL_MS of window, none from T1_ENV_NONE_MS (just
 * above the default Size's 536 ms), so the large settings keep their bloom; in
 * between the depth follows the window, and so glides with the knob. */
#define T1_ENV_DB        30.0f
#define T1_ENV_FULL_MS  250.0f
#define T1_ENV_NONE_MS  560.0f

static float t1_env_depth_db(float window)
{
    const float w_ms = window * (1000.0f / (float)REVERB_FS_HZ);
    float x = (T1_ENV_NONE_MS - w_ms) * (1.0f / (T1_ENV_NONE_MS - T1_ENV_FULL_MS));
    if (x < 0.0f) x = 0.0f; else if (x > 1.0f) x = 1.0f;
    return T1_ENV_DB * x;
}

static void recompute_eff_gains_t1(void)
{
    memcpy(effGains_t1_prev, effGains_t1, sizeof effGains_t1_prev);
    t1_makeup_prev = t1_makeup;
    t1_ramp_left = GAIN_RAMP;
    int count = t1TapCount;
    if (count <= 0) return;

    float window = t1_window_morph;
    float fade = window * FADE_WINDOW_FRAC;
    if (fade < T1_FADE_MIN) fade = T1_FADE_MIN;

    float effWin[MAX_T1_TAPS];
    compute_tap_states_mono(t1TapLadder, t1TapLadderCount, count, window, fade,
                            effOffT1, effWin);

    float pre[MAX_T1_TAPS];
    float energy = 0.0f, thinEnergy = 0.0f;
    for (int t = 0; t < count; t++) {
        float g = (float)t1TapGains[t] * effWin[t];
        energy += g * g;
        pre[t] = g;
    }
    thinEnergy = density_apply(pre, effOffT1, t1TapRanks, t1TapOrder, count, t1_count_morph, NULL);
    const float densComp = density_comp(energy, thinEnergy);
    for (int t = 0; t < count; t++) pre[t] *= densComp;

    float comp = 1.0f;
    if (energy > 1e-3f && t1_full_energy > 1e-3f) {
        comp = sqrtf(t1_full_energy / energy);
        if (comp > TAP_COMP_CLAMP) comp = TAP_COMP_CLAMP;
    }
    t1_comp_goal = comp;
    t1_tap_comp_morph += ALPHA_WINDOW_MORPH * (comp - t1_tap_comp_morph);

    /* Short-decay envelope (see T1_ENV_DB), rescaled so T1 keeps the energy it
     * would have without it: it reshapes T1, it does not make it quieter. */
    const float depth = t1_env_depth_db(window);
    if (depth > 0.0f) {
        const float invLen = 1.0f / window;
        const float span = depth * (1.0f / 80.0f);      /* exp_env_lut spans 80 dB */
        float envEnergy = 0.0f;
        for (int t = 0; t < count; t++) {
            pre[t] *= exp_decay_from_frac((float)effOffT1[t] * invLen * span);
            envEnergy += pre[t] * pre[t];
        }
        if (envEnergy > 1e-3f) {
            const float k = sqrtf(energy / envEnergy);
            for (int t = 0; t < count; t++) pre[t] *= k;
        }
    }

    /* Onset: each T1 tap adds another copy of T0's dense pattern, so with
     * flat T1 gains the tail's energy climbs ~9 dB over the first ~150 ms.
     * Weighting T1's earliest taps up (t1OnsetBoost) makes the tail start at
     * level; rescaled so T1 keeps its energy. */
    {
        float e0 = 0.0f, e1 = 0.0f;
        for (int t = 0; t < count; t++) {
            e0 += pre[t] * pre[t];
            pre[t] *= t1OnsetBoost[t];
            e1 += pre[t] * pre[t];
        }
        if (e1 > 1e-3f) { const float k = sqrtf(e0 / e1); for (int t = 0; t < count; t++) pre[t] *= k; }
    }

    t1_makeup = stage_fit(pre, count, t1_tap_comp_morph);
    for (int t = 0; t < count; t++) effGains_t1[t] = clamp_f_to_int16(pre[t]);
}

static void recompute_eff_gains_t2(void)
{
    memcpy(effGains_t2_prev, effGains_t2, sizeof effGains_t2_prev);
    memcpy(effGains_t2R_prev, effGains_t2R, sizeof effGains_t2R_prev);
    t2_makeup_l_prev = t2_makeup_l; t2_makeup_r_prev = t2_makeup_r;
    t2_ramp_left = GAIN_RAMP;
    int count = t2TapCount;
    if (count <= 0) return;

    float window = t2_window_morph;
    float fade = window * FADE_WINDOW_FRAC;
    if (fade < T2_FADE_MIN) fade = T2_FADE_MIN;
    float invWindow = (window > 1e-6f) ? 1.0f / window : 0.0f;
    /* The taps fall 80 dB over the fade length, which is the window from
     * Size's midpoint up and shorter than it below. */
    const float invFade = (t2_fade_morph > 1e-6f) ? 1.0f / t2_fade_morph : 0.0f;

    float effWin[MAX_T2_TAPS];
    compute_tap_states_lr(t2TapLadderL, t2TapLadderR, t2TapLadderCount, count,
                          window, fade, effOffT2L, effOffT2R, effWin);

    /* refEnergy is the stage's energy with the fade spanning the window: the
     * level compensation works from it, and the taps are scaled to keep it, so
     * a fade shorter than the window shortens the tail without making it
     * quieter. When the two are equal, so are the energies. */
    float pre[MAX_T2_TAPS];
    float energy = 0.0f, thinEnergy = 0.0f, refEnergy = 0.0f;
    for (int t = 0; t < count; t++) {
        const float off = (float)effOffT2L[t];
        const float base = (float)t2TapGains[t] * effWin[t];
        const float r = base * exp_decay_from_frac(off * invWindow);
        float g = base * exp_decay_from_frac(off * invFade);
        refEnergy += r * r;
        energy += g * g;
        pre[t] = g;
    }
    /* T2 is the stereo stage, L and R reading at their own offsets. Thinned
     * to a few taps, a shared selection would leave the width to whichever
     * taps sound (with only the pinned first tap, L and R are the same tap:
     * mono). So R wanders on its own, half the circle from L, and each side
     * keeps its own few taps and its own level. At the full set both are
     * every tap, as before. */
    float preR[MAX_T2_TAPS];
    for (int t = 0; t < count; t++) preR[t] = pre[t];
    thinEnergy = density_apply(pre, effOffT2L, t2TapRanks, t2TapOrder, count, t2_count_morph, &t2_wander);
    const float thinEnergyR = density_apply(preR, effOffT2R, t2TapRanks, t2TapOrder, count, t2_count_morph, &t2_wander_r);
    float densComp = density_comp(energy, thinEnergy), densCompR = density_comp(energy, thinEnergyR);
    if (refEnergy != energy && energy > 1e-6f) { const float k = sqrtf(refEnergy / energy); densComp *= k; densCompR *= k; }
    for (int t = 0; t < count; t++) { pre[t] *= densComp; preR[t] *= densCompR; }

    float comp = 1.0f;
    if (refEnergy > 1e-3f && t2_full_energy > 1e-3f) {
        comp = sqrtf(t2_full_energy / refEnergy);
        if (comp > TAP_COMP_CLAMP) comp = TAP_COMP_CLAMP;
    }
    t2_comp_goal = comp;
    t2_tap_comp_morph += ALPHA_WINDOW_MORPH * (comp - t2_tap_comp_morph);

    /* Each side to fit its own Q15 gains, with its own make-up. */
    t2_makeup_l = stage_fit(pre, count, t2_tap_comp_morph);
    t2_makeup_r = stage_fit(preR, count, t2_tap_comp_morph);
    for (int t = 0; t < count; t++) {
        effGains_t2[t] = clamp_f_to_int16(pre[t]);
        effGains_t2R[t] = clamp_f_to_int16(preR[t]);
#if REVERB_T2_R_SIGNS
        if (t2FlipR[t]) effGains_t2R[t] = (int16_t)-effGains_t2R[t];
#endif
    }
#ifdef VELVET_REVERB_HOST
    for (int t = 0; t < count; t++) host_effwin_t2[t] = effWin[t];
#endif
}

/* ==== Per-block morph state update (inside the block timer — cheap) ====
 *
 * Advances morph state toward targets (one-pole IIR) and refreshes recirc
 * tap offsets to track the morphed windows. Both are cheap (~5 µs CPU).
 * The expensive effGains recompute is done OUTSIDE the block timer — see
 * background_eff_gains_update() below. */
static void update_morph_state(void)
{
    /* Throttled macro recompute. apply_*_macro is called from params.c at
     * main-loop rate (~1 kHz); recomputing the param table there fires
     * ~14 powf calls (~17 µs) on every iteration just to track 12-bit ADC
     * jitter. The morph IIRs immediately downstream have a ~30 ms tau, so
     * anything faster than ~100 Hz updates the morph targets in a way the
     * smoother can't track. Run at most once per 16 audio blocks (~11 ms)
     * and only when a macro value actually moved past the dead-band. */
    static uint8_t macro_throttle_counter = 0;
    if (++macro_throttle_counter >= 16) {
        macro_throttle_counter = 0;
        if (macros_dirty) {
            macros_dirty = 0;
            velvet_reverb_recompute_macros();
        }
    }

    t0_count_morph += ALPHA_WINDOW_MORPH * (reverb_t0_count_target - t0_count_morph);
    t1_count_morph += ALPHA_WINDOW_MORPH * (reverb_t1_count_target - t1_count_morph);
    t2_count_morph += ALPHA_WINDOW_MORPH * (reverb_t2_count_target - t2_count_morph);
    wander_blocks++;
    loop_thin_morph += ALPHA_WINDOW_MORPH * (loop_thin_target - loop_thin_morph);
    if (fabsf(loop_thin_target - loop_thin_morph) < 1e-5f) loop_thin_morph = loop_thin_target;
    density_wandering = wander_amount(t2_count_morph, t2TapCount) > 0.0f;
    float t0_win_target = reverb_t0_window_ms_target  * (float)REVERB_FS_HZ / 1000.0f;
    float t1_win_target = reverb_t1_window_ms_target  * (float)REVERB_FS_HZ / 1000.0f;
    float t2_win_target = reverb_t2_duration_s_target * (float)REVERB_FS_HZ;
    t0_window_morph += ALPHA_WINDOW_MORPH * (t0_win_target - t0_window_morph);
    t1_window_morph += ALPHA_WINDOW_MORPH * (t1_win_target - t1_window_morph);
    t2_window_morph += ALPHA_WINDOW_MORPH * (t2_win_target - t2_window_morph);
    t2_fade_morph += ALPHA_WINDOW_MORPH * (reverb_t2_fade_s_target * (float)REVERB_FS_HZ - t2_fade_morph);

    /* The loop lengths. They are not glided: a moving delay in the feedback
     * loop re-pitches the whole tail. do_predelay steps to them (see PdLen).
     * Feedback/mix/shelf are gain-like and read directly there. */
    predelay_a_morph = reverb_predelay_a_s * (float)REVERB_FS_HZ;
    predelay_b_morph = reverb_predelay_b_s * (float)REVERB_FS_HZ;

    /* Smooth the Tone-macro cutoffs so biquad coeffs track a continuous sweep
     * (no staircase zipper). do_finalize / do_predelay recompute from these. */
    lpf_hz_morph   += ALPHA_WINDOW_MORPH * (reverb_lpf_hz          - lpf_hz_morph);
    /* Shifting down drops the HPF by the same interval (an octave down takes
     * the default 170 Hz to 85), so the shifted-down tail keeps its low end.
     * The morph glides it. */
    float hpf_goal = reverb_hpf_hz;
    if (pitch_mix_target > 0.0f && pitch_semis_target < 0.0f)
        hpf_goal *= powf(2.0f, pitch_semis_target * (1.0f / 12.0f));
    hpf_hz_morph   += ALPHA_WINDOW_MORPH * (hpf_goal               - hpf_hz_morph);
    fb_hs_hz_morph += ALPHA_WINDOW_MORPH * (reverb_fb_high_shelf_hz - fb_hs_hz_morph);
}

/* ==== Background effGains recompute (outside the block timer) ====
 *
 * Two conflicting requirements:
 *   1. To avoid audible step artefacts in the convolution output the
 *      effGains must be re-derived from the current morph state often
 *      (every block is ideal).
 *   2. The full 3-stage recompute is ~90 µs of main-loop wall time —
 *      enough to push the main-loop iteration past the inter-block
 *      period when run every block, causing drops.
 *
 * Resolution: detect target changes with per-parameter dead-bands and
 * recompute every block for ~40 ms after each change (the morph IIR's
 * settle window). When the user is not turning knobs, recompute pauses
 * and effGains sit static — same shape as Phase 1's behaviour, no drops.
 * A slow periodic refresh (every ~67 ms) corrects any morph-state drift
 * during long steady periods. */

#define COMP_SETTLED  0.002f
#define EFFGAINS_SETTLE_BLOCKS    180   /* ~120 ms of recompute after a target change (> the ~90 ms morph tau) */
#define EFFGAINS_PERIODIC_BLOCKS  100   /* ~67 ms safety refresh interval */

/* Per-target dead-bands. Set to "comfortably above ADC/morph noise floor"
 * so genuine user gestures register but pot jitter doesn't. */
#define DB_WINDOW_TARGET   0.50f      /* ms */
#define DB_DURATION_TARGET 0.01f      /* seconds */
#define DB_COUNT_TARGET    0.05f      /* taps */

/* The same dead-bands expressed in the morph state's own units (samples at
 * fs_reverb), so the convergence test below can be judged on the same scale as
 * the change detector above. Keeping them consistent is the whole point: see the
 * comment on the periodic refresh for what happened when they disagreed. */
#define DB_WINDOW_SAMPLES   (DB_WINDOW_TARGET   * (float)REVERB_FS_HZ / 1000.0f)  /*  12 */
#define DB_DURATION_SAMPLES (DB_DURATION_TARGET * (float)REVERB_FS_HZ)            /* 240 */

static float last_t0_win_target = -1e9f, last_t1_win_target = -1e9f, last_t2_dur_target = -1e9f;
static float last_t2_fade_target = -1e9f;
static float last_t0_cnt_target = -1e9f, last_t1_cnt_target = -1e9f, last_t2_cnt_target = -1e9f;
static uint16_t effgains_settle_counter = 0;
static uint16_t effgains_periodic_counter = 0;

static void background_eff_gains_update(void)
{
#ifdef VELVET_REVERB_HOST
    host_dbg_bg_blocks++;
#endif
    /* Watch user-facing targets for movement: the windows (Size) and the tap
     * counts (Density). */
    int changed = 0;
    if (fabsf(reverb_t0_count_target - last_t0_cnt_target) > DB_COUNT_TARGET) {
        last_t0_cnt_target = reverb_t0_count_target; changed = 1;
    }
    if (fabsf(reverb_t1_count_target - last_t1_cnt_target) > DB_COUNT_TARGET) {
        last_t1_cnt_target = reverb_t1_count_target; changed = 1;
    }
    if (fabsf(reverb_t2_count_target - last_t2_cnt_target) > DB_COUNT_TARGET) {
        last_t2_cnt_target = reverb_t2_count_target; changed = 1;
    }
    if (fabsf(reverb_t0_window_ms_target - last_t0_win_target) > DB_WINDOW_TARGET) {
        last_t0_win_target = reverb_t0_window_ms_target; changed = 1;
    }
    if (fabsf(reverb_t1_window_ms_target - last_t1_win_target) > DB_WINDOW_TARGET) {
        last_t1_win_target = reverb_t1_window_ms_target; changed = 1;
    }
    if (fabsf(reverb_t2_duration_s_target - last_t2_dur_target) > DB_DURATION_TARGET) {
        last_t2_dur_target = reverb_t2_duration_s_target; changed = 1;
    }
    if (fabsf(reverb_t2_fade_s_target - last_t2_fade_target) > DB_DURATION_TARGET) {
        last_t2_fade_target = reverb_t2_fade_s_target; changed = 1;
    }

    /* Size moves each stage's energy a long way, so the level comps must
     * arrive, not stop wherever the settle window leaves them. */
    const int comps_moving = fabsf(t0_comp_goal - t0_tap_comp_morph) > COMP_SETTLED ||
                             fabsf(t1_comp_goal - t1_tap_comp_morph) > COMP_SETTLED ||
                             fabsf(t2_comp_goal - t2_tap_comp_morph) > COMP_SETTLED;
    if (changed || density_wandering || comps_moving) effgains_settle_counter = EFFGAINS_SETTLE_BLOCKS;

    /* Slow periodic refresh — catches morph-state drift during long stationary
     * periods so effGains don't fossilise out of sync. Skipped once the morph has
     * settled, because a recompute would then reproduce the same gains: pure
     * wasted work adding a ~67 ms-periodic spike (up to ~217 µs on T2) able to
     * push a block past its 667 µs deadline.
     *
     * Two things here were wrong and produced an audible click every 67 ms.
     *
     * The settled test used an absolute 1.0 sample while the change detector above
     * uses a dead-band worth 240 samples of t2w. Any target movement between those
     * two was both too small to count as a change AND too large to count as
     * settled, so it fell through the gap: no settle window was granted, yet the
     * refresh fired every time. Pot jitter lands in that gap constantly. Both
     * tests now use the same dead-bands, so anything too small to be a gesture is
     * also too small to be worth refreshing for.
     *
     * And the refresh granted a single block, which recomputes ONE stage (the
     * round-robin below) after 67 ms of drift — snapping that stage's tap gains
     * across the whole accumulated gap in one step. That is exactly the "audible
     * step artefact in the convolution output" this file's header warns about.
     * Granting the full settle window instead lets the gains track the morph in
     * ~20 small steps per stage, each well inside the 30 ms morph tau. It costs
     * more only when a refresh is genuinely needed, which after the threshold fix
     * means the morph really is far from its target. */
    if (++effgains_periodic_counter >= EFFGAINS_PERIODIC_BLOCKS) {
        effgains_periodic_counter = 0;
        if (effgains_settle_counter == 0) {
            const float t0w = reverb_t0_window_ms_target  * (float)REVERB_FS_HZ / 1000.0f;
            const float t1w = reverb_t1_window_ms_target  * (float)REVERB_FS_HZ / 1000.0f;
            const float t2w = reverb_t2_duration_s_target * (float)REVERB_FS_HZ;
#ifdef DIAG_LEGACY_EFFGAINS_REFRESH
            /* The pre-fix policy, kept so test/host_test/effgains_step.c can A/B
             * the change rather than take the reasoning on trust. Not built into
             * any firmware target. */
            int settled =
                fabsf(t0w - t0_window_morph) < 1.0f &&
                fabsf(t1w - t1_window_morph) < 1.0f &&
                fabsf(t2w - t2_window_morph) < 1.0f;
            if (!settled) effgains_settle_counter = 1;
#else
            int settled =
                fabsf(t0w - t0_window_morph) < DB_WINDOW_SAMPLES &&
                fabsf(t1w - t1_window_morph) < DB_WINDOW_SAMPLES &&
                fabsf(t2w - t2_window_morph) < DB_DURATION_SAMPLES &&
                fabsf(reverb_t2_fade_s_target * (float)REVERB_FS_HZ - t2_fade_morph) < DB_DURATION_SAMPLES &&
                fabsf(reverb_t0_count_target - t0_count_morph) < DB_COUNT_TARGET &&
                fabsf(reverb_t1_count_target - t1_count_morph) < DB_COUNT_TARGET &&
                fabsf(reverb_t2_count_target - t2_count_morph) < DB_COUNT_TARGET;
            if (!settled) effgains_settle_counter = EFFGAINS_SETTLE_BLOCKS;
#endif
        }
    }

    if (effgains_settle_counter == 0) return;
    effgains_settle_counter--;

    /* Rotate stages across successive blocks rather than running all three
     * per block. With 3 stages and a ~60-block settle window after a target
     * change, each stage still gets ~20 recomputes during the 40-ms settle
     * — every ~2 ms, well below the 30-ms morph IIR tau. Caps per-block BG
     * cost at one stage's recompute (~75 µs instead of ~225) so the main
     * loop's (block + bg_eff) total stays well under the 666 µs deadline
     * even while every macro is being CV-modulated. */
#ifdef VELVET_REVERB_HOST
    if (host_dbg_bg_blocks >= host_dbg_eff_lastblk) {
        uint32_t gap = host_dbg_bg_blocks - host_dbg_eff_lastblk;
        host_dbg_eff_lastblk = host_dbg_bg_blocks;
        host_dbg_eff_calls++;
        if (gap > 10u) {
            host_dbg_eff_isolated++;
            if (gap > host_dbg_eff_maxgap) host_dbg_eff_maxgap = gap;
        }
    }
#endif

    static int stage_round_robin = 0;
    switch (stage_round_robin) {
        case 0: recompute_eff_gains_t0(); break;
        case 1: recompute_eff_gains_t1(); break;
        case 2: recompute_eff_gains_t2(); break;
        default: break;
    }
    stage_round_robin = (stage_round_robin + 1) % 3;
}


/* ==== Phase handlers ==== */

/* Linear-interpolated read of a float ring at a (possibly fractional, possibly
 * negative) position. Retained for the harness A/B
 * (build with PREDELAY_INTERP_LINEAR); the pre-delay loop uses the allpass or
 * the cubic below, for the reasons set out there. */
__attribute__((always_inline)) static inline float read_ring_interp(const float *buf, uint32_t mask, float pos)
{
    int32_t i = (int32_t)pos;
    float f = pos - (float)i;
    if (f < 0.0f) { f += 1.0f; i -= 1; }
    float a = buf[(uint32_t)i & mask];
    float b = buf[((uint32_t)i + 1u) & mask];
    return a + f * (b - a);
}

#if REVERB_DECIM == 2
/* Fractional read via a 1st-order Thiran allpass, for the pre-delay feedback
 * lines at 24 kHz.
 *
 * Linear interpolation is a lowpass — at a half-sample offset its gain is
 * cos(ω/2), which is -3 dB at a quarter of the sample rate. Used once that is
 * inaudible, but here it sits INSIDE a loop that content traverses every 4440
 * samples, 5.4 times a second, so the loss compounds. Measured on the loop with
 * the shelves removed (test/host_test/hf_decay.c): 0.65 dB per traversal at 4 kHz
 * and 2.79 dB at 8 kHz, against nothing at all below 2 kHz. That is an 8 kHz RT60
 * of 4 s under mids that ring effectively forever, and it is why the tails were
 * mid-heavy with no shimmer. A 4-point cubic only softens it (0 dB at 4 kHz but
 * still 1.08 dB at 8 kHz) and needs 4 SDRAM floats per read instead of 2.
 *
 * An allpass has unity magnitude at every frequency by construction, so it costs
 * nothing per traversal no matter how many traversals there are, and it reads a
 * single SDRAM float — less bus traffic than the linear read it replaces, which
 * matters because SDRAM contention with the codec ISR is what ruled the cubic out.
 * What it approximates is the group delay, not the magnitude; the error shows up
 * as mild high-frequency dispersion, which in a reverb loop is diffusion.
 *
 * Conditioning: the sample is read 2 ahead of the target position and the allpass
 * supplies a delay of d = 2 - frac, so d stays in (1, 2] and the coefficient
 * a = (1-d)/(1+d) = (frac-1)/(3-frac) stays in [-1/3, 0). Bounded well away from
 * the |a| = 1 stability edge, and — the reason for choosing a single expression
 * over the usual branch that keeps d near 1 — continuous in frac. A branch at
 * frac = 0.5 would step a from +1/3 to -1/5 while both sides agree on the
 * position, which is a filter discontinuity, i.e. a click.
 *
 * Continuous in frac is not the same as continuous in time, though. The read
 * moves on by one sample per sample, and the loop modulation drifts it across
 * a whole sample now and then (about 30 times a second per line at the stock
 * depth). There frac wraps from 1 to 0, the sample read moves on by one more,
 * and the allpass's delay near Nyquist (always one sample, whatever frac) jumps
 * by a whole sample. The loop recirculates those jumps as broadband hiss: on a
 * sine at bright Tone, 56 dB under the tone from 1 to 8 kHz, against 68 dB
 * with the loop modulation off.
 *
 * So a crossing is crossfaded. The allpass that was running carries on past
 * the boundary on its old sample (its frac then a little outside [0, 1), where
 * it stays well conditioned: d between about 0.65 and 2.4), a second one
 * starts on the new sample, primed with the first one's last output so it
 * starts settled, and the output fades from the first to the second over
 * PD_AP_XFADE samples. The read moves at most ~0.35 of a sample during a fade
 * even at full modulation. A jump bigger than that (a loop length step, a
 * reset) starts over on the new sample. */
#define PD_AP_XFADE  32
__attribute__((always_inline)) static inline
float ap_read_step(const float *buf, uint32_t mask, int32_t i, float f, float *pin, float *pout)
{
    const float x = buf[((uint32_t)i + 2u) & mask];
    const float a = (f - 1.0f) / (3.0f - f);
    const float y = a * (x - *pout) + *pin;
    *pin = x; *pout = y;
    return y;
}

/* a - b as a ring offset in [-N/2, N/2), N = mask + 1: the read position is
 * a ring index, so it wraps when the write index does. */
__attribute__((always_inline)) static inline
int32_t ring_diff(int32_t a, int32_t b, uint32_t mask)
{
    const uint32_t h = (mask >> 1) + 1u;
    return (int32_t)((((uint32_t)a - (uint32_t)b) + h) & mask) - (int32_t)h;
}

__attribute__((always_inline)) static inline
float read_ring_allpass(const float *buf, uint32_t mask, float pos, PdReadState *st)
{
    int32_t i = (int32_t)pos;
    if ((float)i > pos) i -= 1;                 /* floor */
    if (st->i == INT32_MIN) {
        st->i = i;
        st->xf = 0;
        return ap_read_step(buf, mask, i, pos - (float)i, &st->prev_in, &st->prev_out);
    }
    /* Where the running allpass reads now (one on from its last read), in
     * pos's frame. */
    const int32_t ia = i - ring_diff(i, st->i + 1, mask);
    const float fa = pos - (float)ia;
    if (st->xf == 0) {
        if (ia == i) {
            st->i = ia;
            return ap_read_step(buf, mask, ia, fa, &st->prev_in, &st->prev_out);
        }
        if (fa < -0.5f || fa >= 1.5f) {
            st->i = i;
            return ap_read_step(buf, mask, i, pos - (float)i, &st->prev_in, &st->prev_out);
        }
        /* A crossing: start the second allpass on the new sample. */
        st->bi = i - 1;
        st->b_in = buf[((uint32_t)i + 1u) & mask];
        st->b_out = st->prev_out;
        st->xf = PD_AP_XFADE;
    }
    const int32_t ib = i - ring_diff(i, st->bi + 1, mask);
    const float fb = pos - (float)ib;
    if (fa < -0.5f || fa >= 1.5f || fb < -0.5f || fb >= 1.5f) {
        /* Moved too far during the fade for either to stay conditioned. */
        st->xf = 0;
        st->i = i;
        return ap_read_step(buf, mask, i, pos - (float)i, &st->prev_in, &st->prev_out);
    }
    const float ya = ap_read_step(buf, mask, ia, fa, &st->prev_in, &st->prev_out);
    const float yb = ap_read_step(buf, mask, ib, fb, &st->b_in, &st->b_out);
    st->i = ia;
    st->bi = ib;
    const float t = (float)(PD_AP_XFADE - st->xf + 1) * (1.0f / (float)PD_AP_XFADE);
    if (--st->xf == 0) {
        st->i = ib;
        st->prev_in = st->b_in;
        st->prev_out = st->b_out;
    }
    return ya + t * t * (3.0f - 2.0f * t) * (yb - ya);
}
#else
/* Fractional read via a 4-point cubic (Catmull-Rom), for the pre-delay feedback
 * lines at 48 kHz.
 *
 * The allpass above is flat in magnitude but its high-frequency delay jumps a
 * whole sample at each crossing, which it has to crossfade. Interpolating FIRs
 * like this one have no such jumps: at frac -> 1 and at the next index with
 * frac = 0 they read the same sample.
 *
 * At 48 kHz the cubic's per-traversal loss, which ruled it out at 24 kHz, has
 * moved up an octave: nothing measurable through 12 kHz, about 1.1 dB per
 * traversal at 16 kHz. So tails darken only in the top octave.
 *
 * SDRAM: four taps, but the integer part of the read position advances by one
 * per output sample except when the modulation carries it across a whole
 * sample, so the window slides and each read costs one new SDRAM float, the
 * same as the allpass. The window is refilled only when the integer part steps
 * by 0 or 2 (or the line index wraps). That matters because reverb SDRAM reads slow the codec ISR. The
 * window can be trusted from one read to the next because every read is at
 * least PRE_MIN_TIME_SAMPLES behind the write head, so nothing it holds is
 * rewritten in between. */
__attribute__((always_inline)) static inline
float read_ring_cubic(const float *buf, uint32_t mask, float pos, PdReadState *st)
{
    int32_t i = (int32_t)pos;
    float f = pos - (float)i;
    if (f < 0.0f) { f += 1.0f; i -= 1; }
    if (i == st->i + 1) {
        st->y0 = st->y1; st->y1 = st->y2; st->y2 = st->y3;
        st->y3 = buf[((uint32_t)i + 2u) & mask];
    } else {
        st->y0 = buf[((uint32_t)i - 1u) & mask];
        st->y1 = buf[(uint32_t)i & mask];
        st->y2 = buf[((uint32_t)i + 1u) & mask];
        st->y3 = buf[((uint32_t)i + 2u) & mask];
    }
    st->i = i;
    const float y0 = st->y0, y1 = st->y1, y2 = st->y2, y3 = st->y3;
    float a = -0.5f * y0 + 1.5f * y1 - 1.5f * y2 + 0.5f * y3;
    float c =        y0 - 2.5f * y1 + 2.0f * y2 - 0.5f * y3;
    float d = -0.5f * y0 + 0.5f * y2;
    return ((a * f + c) * f + d) * f + y1;
}
#endif

/* Pick the pre-delay read once, so the two lines can never diverge. */
#if defined(PREDELAY_INTERP_LINEAR)
#define PREDELAY_READ(buf, pos, st)  read_ring_interp((buf), PRE_DELAY_LINE_MASK, (pos))
#elif REVERB_DECIM == 2
#define PREDELAY_READ(buf, pos, st)  read_ring_allpass((buf), PRE_DELAY_LINE_MASK, (pos), (st))
#else
#define PREDELAY_READ(buf, pos, st)  read_ring_cubic((buf), PRE_DELAY_LINE_MASK, (pos), (st))
#endif

/* poly sine on [-π, π], used for the pre-delay line modulation.
 *
 * The polynomial is a Taylor series about 0, so it is accurate near 0 and wrong
 * at the ends: it returns ±0.524 at ±π where a true sine is 0. do_predelay folds
 * its phase into (-π, π], so feeding the raw phase in stepped the returned value
 * by 1.05 at every fold — which moved the read position of a delay line 7.55
 * samples instantaneously (0.31 ms at 24 kHz), splicing it 0.7 times a second.
 * That was audible as a distinct click, and it was invisible to every peak-based
 * probe because a splice does not change the block's amplitude. See
 * test/host_test/predelay_wrap.c, which measures the step at the wrap sample.
 *
 * Folding to [-π/2, π/2] via sin(x) = sin(π - x) keeps the argument in the range
 * the series is good for, so the value is continuous across the fold (both sides
 * approach 0) and stays within 0.005 of a true sine. The residual kink in the
 * slope at ±π/2 is a slope change, not a step, at a sub-Hz rate. */
__attribute__((always_inline)) static inline float fast_sin_pi(float x)
{
    if (x >  PI_2_F)      x =  PI_F - x;
    else if (x < -PI_2_F) x = -PI_F - x;
    float x2 = x * x;
    return x * (1.0f + x2 * (-0.16666666f + x2 * 0.00833333f));
}

/* ==== Pre-delay line I/O ====
 *
 * The two lines live in SDRAM. The loop reads them in place, one fractional read
 * per line per sample, but its writes are collected for the whole block and
 * written back with one DMA transfer per line. That is exact: the minimum delay
 * (PRE_MIN_TIME_SAMPLES, 5 ms) is far longer than a block, so no read in a block
 * can see that block's writes, and deferring them changes nothing that is read.
 *
 * The octave feed heads copy their stretches of line in by DMA too (see
 * oct_fetch). Every copy has a DMA2 stream of its own, so a block starts them all
 * at once and collects them only when it needs the data:
 *
 *   Stream 2       line write-backs (A finished within its block, B at the next)
 *   Streams 3, 5   octave head A: its stretch, and the outgoing head's
 *   Streams 6, 7   octave head B: likewise
 *
 * Stream 1 is the T2 gather chain and Streams 0 and 4 the ADCs. Buffers the DMA
 * touches are in SRAM, since it cannot reach CCM. */
#define PD_DMA_WB      2u
#define PD_DMA_OCT_A   3u
#define PD_DMA_OCT_AX  5u
#define PD_DMA_OCT_B   6u
#define PD_DMA_OCT_BX  7u
#define PD_CPU         0xFFu   /* no stream: copied by CPU (the reverse heads) */

static float pd_node_a[REVERB_BLOCK] __attribute__((aligned(4)));
static float pd_node_b[REVERB_BLOCK] __attribute__((aligned(4)));
static float pd_node_a_r[REVERB_BLOCK] __attribute__((aligned(4)));
static float pd_node_b_r[REVERB_BLOCK] __attribute__((aligned(4)));
static float pd_px    [REVERB_BLOCK] CCM_ATTR;
static float pd_px_r  [REVERB_BLOCK] CCM_ATTR;
static float pd_fbeff [REVERB_BLOCK] CCM_ATTR;
static float pd_frz   [REVERB_BLOCK] CCM_ATTR;   /* freeze amount per sample, 0..1 */

/* Freeze: the loop stops taking input and stops decaying. Each line's next
 * node becomes its raw read, so the loop gain is exactly 1 at every
 * frequency: the shelves and the DC high-pass are bypassed, the input is not
 * written, and at 24 kHz the fractional read is an allpass, lossless by
 * construction. The delay modulation holds still while frozen, since a moving
 * read under unity feedback would keep re-pitching the held audio (and the
 * allpass hisses at each whole-sample crossing, see read_ring_allpass).
 * The input is kept out of the loop but still reaches the taps: it joins the
 * cascade feed directly, at the level its first pass through the loop node
 * would give it unfrozen, so new sound is reverberated over the held loop.
 * It comes after the input delay, so TIME still delays it. All of it ramps
 * over FREEZE_FADE_SAMPLES, so engaging or releasing never clicks; the
 * filters keep running underneath, so release picks up from settled state. */
#define FREEZE_FADE_SAMPLES  (480.0f * (float)REVERB_RATE_MULT)   /* 20 ms */
/* The cascade feed takes the lines by reverb_delay_mix. A frozen loop is the
 * whole sound, so the feed takes at least this much of it while frozen, should
 * that mix be set lower. */
#define FREEZE_MIX_MIN       0.7f
/* Cross-feed between the channels, so a sound on one side reverberates on
 * both. Each is x + k * (other - x): exact when the sides match, so a mono
 * input is untouched.
 *   STEREO_LOOP_XFEED  each loop pass hands this share of a line's node to
 *     the other side's line, so an off-centre sound's loop tail spreads
 *     evenly within a second or so (the difference between the sides keeps
 *     1 - 2k of itself per pass);
 *   STEREO_T2_XFEED    the T1 -> T2 bridge does the same at once, so the far
 *     side's tail starts with the near one, ~10 dB down, rather than waiting
 *     a loop pass. The early stages (T0, T1) stay on their own side. */
#ifndef STEREO_LOOP_XFEED
#define STEREO_LOOP_XFEED    0.1f
#endif
#ifndef STEREO_T2_XFEED
#define STEREO_T2_XFEED      0.25f
#endif
static volatile uint8_t freeze_request = 0;
static float freeze_amt = 0.0f;

void velvet_reverb_set_freeze(int on)
{
    freeze_request = on ? 1u : 0u;
}

/* Reverse: the cascade feed reads the lines backwards. A pair of feed heads
 * (rev_head_a/b) reads each line at rate -1, jumping a loop length as the
 * pitch heads do (see "Pitch shift" below), and the feed crossfades from the
 * lines' forward reads to them over FREEZE_FADE_SAMPLES. The loop itself keeps
 * recirculating forwards, so its tail is unchanged and release goes straight
 * back to it. The pitch heads run backwards too while reversed, so the pitch
 * shift carries on. Like freeze, reverse takes the feed at least
 * FREEZE_MIX_MIN from the lines. */
static volatile uint8_t reverse_request = 0;
static float reverse_amt = 0.0f;
static float pd_rev   [REVERB_BLOCK] CCM_ATTR;   /* reverse amount per sample, 0..1 */

void velvet_reverb_set_reverse(int on)
{
    reverse_request = on ? 1u : 0u;
}
static uint32_t pd_wb_idx = 0;          /* ring index of the pending line-B write-back */
static uint8_t  pd_wb_pending = 0;

#if defined(VELVET_REVERB_HOST) || DMA2_DISABLED_FOR_DIAGNOSTIC
static inline void pd_kick(uint32_t sn, const float *src, float *dst, uint32_t n) { memcpy(dst, src, n * sizeof(float)); }
static inline int  pd_wait(uint32_t sn) { return 1; }
#else
#define PD_TIMEOUT_CYC     (180u * 50u)         /* 50 µs: many times the longest transfer */
/* Stream sn's registers, and where its flags sit in LISR/LIFCR (0-3) or HISR/HIFCR (4-7). */
#define PD_STREAM(sn)      ((DMA_Stream_TypeDef *)(DMA2_Stream0_BASE + 0x18u * (sn)))
#define PD_FLAG_SHIFT(sn)  ((((sn) & 3u) == 0u) ? 0u : (((sn) & 3u) == 1u) ? 6u : (((sn) & 3u) == 2u) ? 16u : 22u)
#define PD_FLAGS(sn)       (0x3Du << PD_FLAG_SHIFT(sn))
#define PD_TCIF(sn)        (0x20u << PD_FLAG_SHIFT(sn))
#define PD_TEIF(sn)        (0x08u << PD_FLAG_SHIFT(sn))
#define PD_ISR(sn)         (((sn) < 4u) ? &DMA2->LISR  : &DMA2->HISR)
#define PD_IFCR(sn)        (((sn) < 4u) ? &DMA2->LIFCR : &DMA2->HIFCR)

__attribute__((always_inline)) static inline void pd_kick(uint32_t sn, const float *src, float *dst, uint32_t n)
{
    DMA_Stream_TypeDef *st = PD_STREAM(sn);
    /* CPU writes to the source (node values) must be complete before the DMA reads. */
    __DSB();
    st->CR &= ~DMA_SxCR_EN;
    while (st->CR & DMA_SxCR_EN) {}
    *PD_IFCR(sn) = PD_FLAGS(sn);
    st->PAR  = (uint32_t)src;
    st->M0AR = (uint32_t)dst;
    st->NDTR = n;
    st->CR  |= DMA_SxCR_EN;
}
/* 1 if the transfer completed cleanly; 0 on error or timeout, in which case the
 * caller redoes it by CPU. */
static int pd_wait(uint32_t sn)
{
    DMA_Stream_TypeDef *st = PD_STREAM(sn);
    const uint32_t done = PD_TCIF(sn) | PD_TEIF(sn);
    volatile uint32_t *isr = PD_ISR(sn);
    uint32_t t0 = DWT->CYCCNT;
    while ((st->CR & DMA_SxCR_EN) && !(*isr & done)) {
        if ((DWT->CYCCNT - t0) > PD_TIMEOUT_CYC) {
            st->CR &= ~DMA_SxCR_EN;
            while (st->CR & DMA_SxCR_EN) {}
            return 0;
        }
    }
    __DMB();
    return !(*isr & PD_TEIF(sn));
}
#endif

/* A copy of len samples of a line, from ring index s into dst, on DMA stream sn,
 * or by CPU straight away when the span straddles the end of the ring. */
typedef struct {
    const float *line;
    float *dst;
    uint32_t s, len;
    uint8_t sn, kicked;
} PdFetch;

static void pd_fetch_start(PdFetch *f, const float *line, uint32_t s, uint32_t len, float *dst)
{
    f->line = line; f->dst = dst; f->s = s; f->len = len;
    f->kicked = (f->sn != PD_CPU) && (s + len <= PRE_DELAY_LINE_SAMPLES);
    if (f->kicked) { pd_kick(f->sn, &line[s], dst, len); return; }
    for (uint32_t k = 0; k < len; k++) dst[k] = line[(s + k) & PRE_DELAY_LINE_MASK];
}
/* Wait for a fetch if one is in flight, and redo it by CPU if the DMA failed. */
static void pd_fetch_finish(PdFetch *f)
{
    if (!f->kicked) return;
    f->kicked = 0;
    if (!pd_wait(f->sn))
        for (uint32_t k = 0; k < f->len; k++) f->dst[k] = f->line[(f->s + k) & PRE_DELAY_LINE_MASK];
}

/* Write a block of node values back at ring index idx. Blocks start at multiples
 * of REVERB_BLOCK and the line length is a multiple of it, so this never
 * straddles the ring end. */
static void pd_writeback_start(float *line, const float *node, uint32_t idx)
{
    pd_kick(PD_DMA_WB, node, &line[idx], REVERB_BLOCK);
}
static void pd_writeback_finish_now(float *line, const float *node, uint32_t idx)
{
    if (!pd_wait(PD_DMA_WB)) memcpy(&line[idx], node, REVERB_BLOCK * sizeof(float));
}

static void pd_init(void)
{
    pd_wb_pending = 0;
#if !defined(VELVET_REVERB_HOST) && !DMA2_DISABLED_FOR_DIAGNOSTIC
    /* Memory to memory, 32-bit (the lines are float), polled. High priority:
     * above the T2 chain (medium), since the CPU waits on the write-backs and the
     * octave copies must not queue behind the chain, and below the ADCs (very
     * high, see adc.c). */
    static const uint8_t streams[] = { PD_DMA_WB, PD_DMA_OCT_A, PD_DMA_OCT_AX, PD_DMA_OCT_B, PD_DMA_OCT_BX };
    for (unsigned i = 0; i < sizeof streams; i++) {
        DMA_Stream_TypeDef *st = PD_STREAM(streams[i]);
        st->CR = 0;
        while (st->CR & DMA_SxCR_EN) {}
        *PD_IFCR(streams[i]) = PD_FLAGS(streams[i]);
        st->CR = DMA_SxCR_DIR_1 | DMA_SxCR_MINC | DMA_SxCR_PINC
               | (2 << 11) | (2 << 13) | (2 << 16);
        st->FCR = DMA_SxFCR_DMDIS | DMA_SxFCR_FTH;
    }
#endif
}

/* The two loops on each side, after the Cosmos engine's family A: each
 * loop's feedback reads a share of the other loop (LOOP_BLUR, less over the
 * bottom of Density: see DENSITY_THIN_U), so an echo of one comes round again
 * at the other's length, and on, and the repeats multiply. And the two loops'
 * shares of the feed drift on slow random-walk
 * LFOs (LOOP_DRIFT), differently on each side.
 *
 * Cosmos's blend (own 1 - 0.49 blur, other 0.51 blur) keeps the level of
 * what the two loops share but loses that of what they don't, ~2 dB a pass
 * at 0.5, which its Regen (up to 1.066) makes back with a limiter behind it.
 * Here the blend is the rotation with the same shares (cos, sin), which
 * keeps every signal's level, so the feedback Size sets stays the decay.
 * Frozen, each line copies its own raw read, so a frozen loop holds exactly.
 *
 * With one gain per pass the shorter loop (A) went round more often and so
 * died sooner, and the blur pulled the longer loop's energy into it and
 * shortened the tail; so each loop's gain is Size's raised to its length
 * over LOOP_T60_REF_S (and its shelves' dB scaled the same), which keeps the
 * two decaying at one rate per second. */
#ifndef LOOP_BLUR
#define LOOP_BLUR    0.5f
#endif
#ifndef LOOP_DRIFT
#define LOOP_DRIFT   0.35f
#endif
#ifndef LOOP_T60_REF_S
#define LOOP_T60_REF_S   0.27f
#endif

/* One line's read for a whole block: modulated fractional read (and the
 * crossfade when its length steps). The read state is worked on in a local
 * copy so it can stay in registers. */
__attribute__((always_inline)) static inline
void pd_line_read(float *phase, PdReadState *rd, const float *line, float wbase, PdLen *len,
                  float modDepth, float modInc, float *r_out)
{
    PdReadState rs = *rd;
    float time = len->time;
    int32_t left = len->left;
    for (int n = 0; n < REVERB_BLOCK; n++) {
        *phase += modInc; if (*phase >= TWO_PI_F) *phase -= TWO_PI_F;
        float p = *phase; if (p > PI_F) p -= TWO_PI_F;
        float whf = wbase + (float)n;
        float extra = modDepth * (1.0f + fast_sin_pi(p));
        float r = PREDELAY_READ(line, whf - time - extra, &rs);
        if (left > 0) {
            const float r2 = PREDELAY_READ(line, whf - len->to - extra, &len->rs);
            const float t = (float)(PD_LEN_XFADE - left) * (1.0f / (float)PD_LEN_XFADE);
            r += t * t * (3.0f - 2.0f * t) * (r2 - r);
            if (--left == 0) { time = len->to; rs = len->rs; }
        }
        r_out[n] = r;
    }
    *rd = rs;
    len->time = time;
    len->left = left;
}

/* One line's node for a whole block from its own read and the other loop's
 * (the blur, 0..LOOP_BLUR): shelves, DC block, feedback, input. */
__attribute__((always_inline)) static inline
void pd_line_node(Biquad *lsh, Biquad *hsh, float *hp, const float *own, const float *other,
                  float otherSign, float blur, const float *px, float fbScale, float *node)
{
    const float o = 1.0f - 0.49f * blur, t = 0.51f * blur;
    const float norm = 1.0f / sqrtf(o * o + t * t);
    const float kOwn = o * norm, kOther = otherSign * t * norm;   /* cos, +-sin */
    for (int n = 0; n < REVERB_BLOCK; n++) {
        const float r = own[n];
        float rf = biquad_process(hsh, biquad_process(lsh, kOwn * r + kOther * other[n]));
        *hp += pre_hp_coeff * (rf - *hp);
        float nd = px[n] + fbScale * pd_fbeff[n] * (rf - *hp);
        nd += pd_frz[n] * (r - nd);              /* frozen: the raw read, unity gain */
#ifdef VELVET_REVERB_HOST
        { float m = nd < 0.0f ? -nd : nd;
          if (m > host_dbg_node_peak) host_dbg_node_peak = m;
          if (m > 4.0f) host_dbg_node_clamps++; }
#endif
        if (nd > 4.0f) nd = 4.0f; else if (nd < -4.0f) nd = -4.0f;
        node[n] = nd;
    }
}

/* The drift: four slow triangle LFOs (0..1) whose rates wander, as Cosmos's
 * (a ~13 s cycle, +-50%). Advanced once a block. */
static void loop_drift_step(void)
{
    static const float rate[4] = { 6.230e-6f, 7.533e-6f, 4.632e-6f, 5.933e-6f };   /* per sample at 24 kHz */
    for (int j = 0; j < 4; j++) {
        uint32_t x = loop_lfo_seed; x ^= x << 13; x ^= x >> 17; x ^= x << 5; loop_lfo_seed = x;
        const float rnd = (float)x * 2.3283064e-10f - 0.5f;
        float rw = loop_lfo_rw[j] + rnd * 7e-4f;
        loop_lfo_rw[j] = rw < -0.5f ? -0.5f : rw > 0.5f ? 0.5f : rw;
        loop_lfo_ph[j] += (uint32_t)(rate[j] * (loop_lfo_rw[j] + 1.0f) * (float)REVERB_BLOCK * 2147483648.0f);
        const float ph = (float)loop_lfo_ph[j] * (1.0f / 2147483648.0f);
        loop_lfo[j] = ph < 1.0f ? ph : 2.0f - ph;
    }
}

/* Starts a step to a new loop length when one is due (see PdLen). */
static void pd_len_update(PdLen *len, float target)
{
    if (len->left > 0 || fabsf(target - len->time) <= PD_LEN_STEP * len->time) return;
    len->to = target;
    len->left = PD_LEN_XFADE;
#if REVERB_DECIM == 2
    len->rs.prev_in = len->rs.prev_out = 0.0f;   /* faded in from 0, so its start is silent */
    len->rs.i = INT32_MIN;
    len->rs.xf = 0;
#else
    len->rs.i = INT32_MIN;                       /* the first read fills the window */
#endif
}

/* ==== Pitch shift: feed heads on the pre-delay loop ====
 *
 * Each pre-delay line gets a second read head that feeds the cascade at a
 * read rate r = 2^(semitones/12), anywhere from 0.5x (an octave down) to 2x
 * (an octave up). The feedback read stays at 1x, so the loop never
 * recirculates shifted audio: the tail comes out shifted once, rather than
 * climbing as shimmer.
 *
 * Reading at r, a head's delay changes by (1 - r) samples per sample: it
 * gains on the write head above 1x and falls back below it, so it has to
 * jump every so often. It jumps by exactly the line's loop length. One loop
 * back, the line holds nearly the same audio (the same recirculating tail,
 * one feedback pass older), so the jump lands on matching material and a
 * short crossfade covers the rest: fresh input that arrived in the meantime,
 * and the loop's modulation. That holds at any rate, so the pitch can move
 * freely (a 1 V/oct input): a rate change alters only the head's speed, never
 * its position, so it cannot click. Between jumps the head is clean
 * varispeed, and the two lines' jumps never line up. At exactly 1x the head
 * never moves and reads a plain delayed copy of the line.
 *
 * The shifted feed is blended with the normal one on an equal-power curve
 * set by the pitch mix: 0.5 is the equal blend the DLD's octave switch uses.
 *
 * A negative rate reads backwards (reverse): the delay grows by 1 + |r| per
 * sample, so the head falls back and jumps forward a loop, as below 1x. */
/* SDRAM cost. The lines live in SDRAM on a 16-bit bus, so a random float
 * read is slow, and the heads would need up to four per head per sample
 * (plus as many again during a jump's crossfade). Instead each head copies the
 * stretch of line it will cover in the coming block into fast memory once,
 * with sequential reads (at 2x that is ~34 samples, at 0.5x ~10), and the
 * per-sample reads come from the copy. For that to be valid a head never
 * reads audio written during the current block, hence OCT_MIN_DELAY, and
 * jumps happen only at block boundaries. The rate is held for the block. */
#define OCT_MIN_DELAY    ((float)REVERB_BLOCK + 2.0f)   /* reads stay behind this block's writes */
#define OCT_XFADE        (600 * REVERB_RATE_MULT)          /* 25 ms */
#define OCT_SPAN         (2 * REVERB_BLOCK + 2)         /* samples a 2x head covers in a block, plus interp */
#define OCT_RAMP_BINS    256
#define OCT_FADE_STEP    (1.0f / (480.0f * (float)REVERB_RATE_MULT))   /* per sample: 20 ms full mix change */
/* Pitch glide per block: tau ~5 ms, quick enough to follow a sequenced
 * 1 V/oct input without audible portamento, slow enough to smooth a pot. */
#define PITCH_ALPHA      0.13f

typedef struct {
    float d;               /* delay of the main feed head, in samples */
    float xd;              /* delay of the head being faded out */
    int32_t xleft, xlen;   /* crossfade samples left / total (xleft 0 = none) */
    uint32_t base, xbase;  /* line index (unmasked) of lb[0] / xlb[0] */
    PdFetch f, xf;         /* the copies into lb / xlb, in flight from block start */
    float lb[OCT_SPAN];    /* this block's stretch of line for each head */
    float xlb[OCT_SPAN];
} OctHead;

/* In SRAM, not CCM: their stretches are DMA targets (see oct_fetch). */
static OctHead oct_head_a __attribute__((aligned(4))), oct_head_b __attribute__((aligned(4)));
static OctHead rev_head_a __attribute__((aligned(4))), rev_head_b __attribute__((aligned(4)));
/* The right channel's: copied by CPU (there are DMA streams for one set). */
static OctHead oct_head_a_r, oct_head_b_r, rev_head_a_r, rev_head_b_r;
static uint8_t rev_active = 0;               /* reverse heads running */
static float oct_ramp[OCT_RAMP_BINS + 1];    /* sin^2 from 0 to 1 */
static float pitch_semis_morph = 0.0f;
static float pitch_mix_morph = 0.0f;         /* 0..1, ramps at OCT_FADE_STEP */
static uint8_t oct_active = 0;               /* heads running */

void velvet_reverb_set_pitch(float semitones)
{
    PROF_PANEL_IGNORED();
    if (semitones > PITCH_SEMIS_MAX) semitones = PITCH_SEMIS_MAX;
    if (semitones < -PITCH_SEMIS_MAX) semitones = -PITCH_SEMIS_MAX;
    pitch_semis_target = semitones;
}

void velvet_reverb_set_pitch_mix(float mix)
{
    PROF_PANEL_IGNORED();
    if (mix < 0.0f) mix = 0.0f; if (mix > 1.0f) mix = 1.0f;
    pitch_mix_target = mix;
}

/* The DLD's octave switch: an octave either way at the equal blend, or off.
 * Off leaves the pitch where it was, so the heads fade out at that pitch. */
void velvet_reverb_set_octave(int8_t octave)
{
    PROF_PANEL_IGNORED();
    if (octave != 0) pitch_semis_target = (octave > 0) ? 12.0f : -12.0f;
    pitch_mix_target = (octave != 0) ? 0.5f : 0.0f;
}

static void oct_init(void)
{
    for (int i = 0; i <= OCT_RAMP_BINS; i++) {
        float x = sinf(0.5f * PI_F * (float)i / (float)OCT_RAMP_BINS);
        oct_ramp[i] = x * x;
    }
    pitch_semis_morph = pitch_semis_target;
    pitch_mix_morph = 0.0f; oct_active = 0;
    memset(&oct_head_a, 0, sizeof oct_head_a);
    memset(&oct_head_b, 0, sizeof oct_head_b);
    oct_head_a.f.sn = PD_DMA_OCT_A; oct_head_a.xf.sn = PD_DMA_OCT_AX;
    oct_head_b.f.sn = PD_DMA_OCT_B; oct_head_b.xf.sn = PD_DMA_OCT_BX;
    memset(&rev_head_a, 0, sizeof rev_head_a);
    memset(&rev_head_b, 0, sizeof rev_head_b);
    rev_head_a.f.sn = rev_head_a.xf.sn = PD_CPU;
    rev_head_b.f.sn = rev_head_b.xf.sn = PD_CPU;
    memset(&oct_head_a_r, 0, sizeof oct_head_a_r);
    memset(&oct_head_b_r, 0, sizeof oct_head_b_r);
    memset(&rev_head_a_r, 0, sizeof rev_head_a_r);
    memset(&rev_head_b_r, 0, sizeof rev_head_b_r);
    oct_head_a_r.f.sn = oct_head_a_r.xf.sn = PD_CPU;
    oct_head_b_r.f.sn = oct_head_b_r.xf.sn = PD_CPU;
    rev_head_a_r.f.sn = rev_head_a_r.xf.sn = PD_CPU;
    rev_head_b_r.f.sn = rev_head_b_r.xf.sn = PD_CPU;
    rev_active = 0; reverse_amt = 0.0f;
}

/* Start a head where its sweep begins: a loop back when it will gain on the
 * write head, as close to it as allowed when it will fall back. */
static void oct_head_start(OctHead *h, float loop, float rate)
{
    h->d = (rate > 1.0f) ? loop + OCT_MIN_DELAY : OCT_MIN_DELAY;
    h->xleft = 0;
}

/* Start copying the stretch of line a head at delay d0 will read over the next
 * nsamp samples at the given rate. lb[0] is one sample older than its first
 * read, so the anti-alias average's extra tap fits. The stretch is older than
 * anything this block writes, so the copy can run alongside the block's own
 * write-backs. Reads land at lb[1 + n*rate .. 2 + n*rate], hence the length.
 * Reading backwards, the stretch ends at d0 and starts |rate| * (nsamp - 1)
 * further back. */
static void oct_fetch(PdFetch *f, float *lb, uint32_t *base, const float *line, uint32_t w0, float d0, float rate, int nsamp)
{
    const float span = (float)(nsamp - 1) * (rate < 0.0f ? -rate : rate);
    *base = w0 - (uint32_t)ceilf(rate < 0.0f ? d0 + span : d0) - 1u;
    uint32_t k = (uint32_t)(3.0f + span) + 1u;
    if (k > OCT_SPAN) k = OCT_SPAN;
    pd_fetch_start(f, line, *base & PRE_DELAY_LINE_MASK, k, lb);
}

/* Block start for one head: jump if due, then start its fetches (collected by
 * oct_head_land before the first sample is read). w0 is the write index at the
 * block's first sample, loop the line's current loop length. */
static void oct_head_begin_block(OctHead *h, const float *line, uint32_t w0, float loop, float rate)
{
    if (h->xleft == 0) {
        /* Gaining: jump back a loop while the outgoing head still has a whole
         * crossfade of room above OCT_MIN_DELAY (checked a block early, since
         * jumps only happen here). Falling back: jump forward a loop once a
         * whole loop of room has opened up. */
        const int gaining = rate > 1.0f;
        if (gaining ? (h->d < OCT_MIN_DELAY + (float)OCT_XFADE + (float)REVERB_BLOCK)
                    : (h->d >= loop + OCT_MIN_DELAY)) {
            int32_t len = OCT_XFADE;
            if (gaining && (int32_t)(h->d - OCT_MIN_DELAY) < len)
                len = (int32_t)(h->d - OCT_MIN_DELAY);    /* only with a very short loop */
            if (len > 0) {
                h->xd = h->d;
                h->xleft = h->xlen = len;
            }
            h->d += gaining ? loop : -loop;
        }
    }
    oct_fetch(&h->f, h->lb, &h->base, line, w0, h->d, rate, REVERB_BLOCK);
    /* The outgoing head only runs until its crossfade ends. Fetching a
     * whole block for it would reach past the write head (never read, but
     * wasted SDRAM reads). */
    if (h->xleft > 0)
        oct_fetch(&h->xf, h->xlb, &h->xbase, line, w0, h->xd, rate, (h->xleft < REVERB_BLOCK) ? h->xleft : REVERB_BLOCK);
}

static void oct_head_land(OctHead *h)
{
    pd_fetch_finish(&h->f);
    pd_fetch_finish(&h->xf);
}

/* Read at rel samples into a fetched stretch. Above 1x the read decimates the
 * line, so blend in the 2-tap average the input decimator uses, by `aa`: none
 * at 1x, all of it at 2x (as the octave-up head always did). */
__attribute__((always_inline)) static inline float oct_lb_read(const float *lb, float rel, float aa)
{
    int32_t i = (int32_t)rel;
    float f = rel - (float)i;
    float b = lb[i], c = lb[i + 1];
    float v = b + f * (c - b);
    float a = lb[i - 1];
    float avg = 0.5f * (v + a + f * (b - a));
    return v + aa * (avg - v);
}

/* A head's samples for the block that started at w0, into out[]. The heads are
 * independent of each other and of the mix, so each gets a pass of its own, with
 * its state in locals. */
static void oct_head_run(OctHead *h, uint32_t w0, float rate, float *out)
{
    const float step = 1.0f - rate;
    const float speed = (rate < 0.0f) ? -rate : rate;
    const float aa = (speed > 1.0f) ? speed - 1.0f : 0.0f;
    const float *lb = h->lb, *xlb = h->xlb;
    const uint32_t base = h->base, xbase = h->xbase;
    const int32_t xlen = h->xlen;
    float d = h->d, xd = h->xd;
    int32_t xleft = h->xleft;

    for (int n = 0; n < REVERB_BLOCK; n++) {
        float y = oct_lb_read(lb, (float)(int32_t)(w0 + (uint32_t)n - base) - d, aa);
        if (xleft > 0) {
            float yo = oct_lb_read(xlb, (float)(int32_t)(w0 + (uint32_t)n - xbase) - xd, aa);
            float w = oct_ramp[((xlen - xleft) * OCT_RAMP_BINS) / xlen];
            y = yo + w * (y - yo);
            xd += step;
            xleft--;
        }
        d += step;
        out[n] = y;
    }
    h->d = d; h->xd = xd; h->xleft = xleft;
}

static float oct_out_a[REVERB_BLOCK] CCM_ATTR;
static float oct_out_b[REVERB_BLOCK] CCM_ATTR;
static float rev_out_a[REVERB_BLOCK] CCM_ATTR;
static float rev_out_b[REVERB_BLOCK] CCM_ATTR;
static float oct_out_a_r[REVERB_BLOCK] CCM_ATTR;
static float oct_out_b_r[REVERB_BLOCK] CCM_ATTR;
static float rev_out_a_r[REVERB_BLOCK] CCM_ATTR;
static float rev_out_b_r[REVERB_BLOCK] CCM_ATTR;

#if defined(DIAG_REVERB_PROFILE) && !defined(VELVET_REVERB_HOST)
/* Device-side check of the DMA paths, which the host build replaces with memcpy
 * and so cannot test: on sampled blocks, prof_verify_dma compares what the DMA
 * delivered with a CPU read of the same memory. These record what this block
 * used. */
static int      prof_t2_h = -1;          /* half of the T2 rows used, or -1 */
static uint32_t prof_t2_bw;              /* block index they were used for */
static uint8_t  prof_oct_ran;            /* octave heads fetched this block */
static uint32_t prof_dma_checks, prof_dma_bad;
static uint32_t prof_queue_max;          /* deepest backlog poll picked up since the last report */
#endif

/* One channel's input-delay read at a fractional delay d (>= 0) behind w. */
static inline float id_read(const int16_t *ring, uint32_t w, float d)
{
    const int32_t i = (int32_t)d;
    const float f = d - (float)i;
    const float a = (float)ring[(w - (uint32_t)i) & INPUT_DELAY_MASK];
    const float b = (float)ring[(w - (uint32_t)i - 1u) & INPUT_DELAY_MASK];
    return a + f * (b - a);
}

static void do_input_delay(void)
{
    for (int n = 0; n < REVERB_BLOCK; n++) {
        const uint32_t w = id_w++;
        input_delay[w & INPUT_DELAY_MASK] = input_ready[n];
        input_delay_r[w & INPUT_DELAY_MASK] = input_ready_r[n];
        id_smooth += ID_SMOOTH * (id_target - id_smooth);
        if (id_left == 0) {
            const float diff = id_smooth - id_pos;
            if (diff > ID_JUMP || diff < -ID_JUMP) {
                id_to = (int32_t)(id_smooth + 0.5f);
                id_left = ID_XFADE;
            } else {
                id_pos += diff > ID_SLEW ? ID_SLEW : diff < -ID_SLEW ? -ID_SLEW : diff;
            }
        }
        float x = id_read(input_delay, w, id_pos), xr = id_read(input_delay_r, w, id_pos);
        if (id_left > 0) {
            const uint32_t rt = (w - (uint32_t)id_to) & INPUT_DELAY_MASK;
            const float t = (float)(ID_XFADE - id_left) * (1.0f / (float)ID_XFADE);
            const float k = t * t * (3.0f - 2.0f * t);
            x  += k * ((float)input_delay[rt] - x);
            xr += k * ((float)input_delay_r[rt] - xr);
            if (--id_left == 0) id_pos = (float)id_to;
        }
        id_out[n] = x;
        id_out_r[n] = xr;
    }
}

/* One channel's cascade feed for the block, from its input, its two line
 * nodes and its pitch and reverse heads. The block-wide weights are shared. */
typedef struct {
    float mix, wA, wB, dryK;
    float keep0, keep1, shift0, shift1;
    int oct_run, rev_run;
} PdMix;

static void pd_mix(const PdMix *pm, const float *pdpx, const float *idout,
                   const float *na, const float *nb, const float *oa, const float *ob,
                   const float *ra, const float *rb, float drift, float *out)
{
    const float to_unit = 1.0f / 32768.0f;
    const float invBlock = 1.0f / (float)REVERB_BLOCK;
    /* The drift moves the balance between the loops; their sum holds. */
    const float mix = pm->mix, wA = pm->wA * (1.0f + drift), wB = pm->wB * (1.0f - drift);
    for (int n = 0; n < REVERB_BLOCK; n++) {
        float px = pdpx[n];
        const float fz = pd_frz[n];
        const float rv = pm->rev_run ? pd_rev[n] : 0.0f;
        const float hold = fz > rv ? fz : rv;
        const float m = mix + hold * ((mix > FREEZE_MIX_MIN ? mix : FREEZE_MIX_MIN) - mix);
        float lines = wA * na[n] + wB * nb[n];
        if (pm->rev_run) lines += rv * (wA * ra[n] + wB * rb[n] - lines);
        float feed = px + m * (lines - px);
        feed += fz * pm->dryK * (idout[n] * to_unit);   /* frozen: input straight to the taps */

        if (pm->oct_run) {
            /* The heads advance every sample while running, at any mix, so
             * they stay in step with their fetched stretches. */
            float sA = wA * oa[n];
            float sB = wB * ob[n];
            /* Shifted counterpart of feed = px + mix * (nodeA + nodeB - px).
             * There is no separate shifted px: the lines carry it, but the
             * heads read the two lines at different delays, so it adds up
             * incoherently. (sA + sB) / sqrt(2) stands in for it. */
            float shifted = ((1.0f - m) * 0.70710678f + m) * (sA + sB);
            const float t = (float)n * invBlock;
            const float keep  = pm->keep0  + t * (pm->keep1  - pm->keep0);
            const float shift = pm->shift0 + t * (pm->shift1 - pm->shift0);
            feed = keep * feed + shift * shifted;
        }

        out[n] = feed;
    }
}

/* ==== Pre-delay sustain engine ====
 * Two modulated, damped feedback delay lines in front of the cascade. Reads
 * input_ready (int16), works in normalised ±1 float (so duck / node-clamp /
 * shelf math all operate on a float signal), and writes the cascade input into
 * predelay_out (still normalised; clamped to int16 in do_input_write_t0).
 * When feedback is 0 the loop is
 * bypassed and the dry input passes straight through. */
static void do_predelay(void)
{
    const float inv_fs = 1.0f / (float)REVERB_FS_HZ;
    const float to_unit = 1.0f / 32768.0f;

    do_input_delay();

    float fb = reverb_feedback;
    if (fb > 0.999f) fb = 0.999f;
    if (fb < 0.0f) fb = 0.0f;
    /* Size sets the loop's feedback directly; it glides here (~65 ms) so a
     * turn does not step the gain of a recirculating loop every block. */
    if (fb_morph < 0.0f) fb_morph = fb;
    fb_morph += 0.01f * (fb - fb_morph);
    fb = fb_morph;

    float timeA = predelay_a_morph;
    float timeB = predelay_b_morph;
    if (timeA < PRE_MIN_TIME_SAMPLES) timeA = PRE_MIN_TIME_SAMPLES;
    if (timeA > (float)PRE_DELAY_MAX_SAMPLES) timeA = (float)PRE_DELAY_MAX_SAMPLES;
    if (timeB < PRE_MIN_TIME_SAMPLES) timeB = PRE_MIN_TIME_SAMPLES;
    if (timeB > (float)PRE_DELAY_MAX_SAMPLES) timeB = (float)PRE_DELAY_MAX_SAMPLES;
    pd_len_update(&pre_len_a, timeA);
    pd_len_update(&pre_len_b, timeB);
    pd_len_update(&pre_len_a_r, timeA);
    pd_len_update(&pre_len_b_r, timeB);
    timeA = pre_len_a.time;   /* the lengths the loop reads at this block */
    timeB = pre_len_b.time;

    if (freeze_amt == 0.0f) {
        mod_amount_morph += MOD_AMOUNT_ALPHA * (mod_amount_target - mod_amount_morph);
        if (fabsf(mod_amount_target - mod_amount_morph) < 1e-4f) mod_amount_morph = mod_amount_target;
    }
    const float modShare = mod_depth_share(mod_amount_morph);
    const float modDepth = modShare * modShare * MOD_FB_DEPTH_MAX * (float)PRE_MOD_MAX_SAMPLES;
    float modRate = reverb_fb_mod_rate; if (modRate < 0.0f) modRate = 0.0f;
    float modInc = TWO_PI_F * modRate * inv_fs * (1.0f - freeze_amt);   /* held while frozen */

    float mix = reverb_delay_mix;
    if (mix < 0.0f) mix = 0.0f; else if (mix > 1.0f) mix = 1.0f;

    float duckAmount = reverb_duck_amount;
    if (duckAmount < 0.0f) duckAmount = 0.0f; else if (duckAmount > 1.0f) duckAmount = 1.0f;
    /* Release tracks the loop length. timeA/timeB are the current (morphed,
     * clamped) line lengths in samples, so the release is already in samples
     * and needs no conversion. The LONGER line sets it, so both lines get at
     * least one full traversal. */
    float loopSamples = (timeA > timeB) ? timeA : timeB;
    float duckRelSamples = reverb_duck_release_loops * loopSamples;
    const float duckRelMin = 0.01f * (float)REVERB_FS_HZ;
    if (!(duckRelSamples >= duckRelMin)) duckRelSamples = duckRelMin;
    float duckRelCoef = 1.0f - expf(-1.0f / duckRelSamples);

    float ls_db = reverb_fb_low_shelf_db, hs_db = reverb_fb_high_shelf_db;
#if REVERB_LOOP_T60_SHELVES
    if (reverb_t60_shelves_on) { ls_db = t60_ls_db; hs_db = t60_hs_db; }
#endif
    /* Each loop's gain and shelves at its length over LOOP_T60_REF_S. */
    const float tRef = LOOP_T60_REF_S * (float)REVERB_FS_HZ;
    const float abRatio = timeA / tRef, bbRatio = timeB / tRef;
    const float fbScaleA = (fb > 1e-6f) ? powf(fb, abRatio - 1.0f) : 1.0f;   /* fb^r / fb */
    const float fbScaleB = (fb > 1e-6f) ? powf(fb, bbRatio - 1.0f) : 1.0f;
    /* Shelf coeffs, recomputed only on change: A's for both A lines, B's for
     * both B lines. */
    if (reverb_fb_low_shelf_hz != last_fb_ls_hz || ls_db != last_fb_ls_db) {
        biquad_lowshelf(&pre_low_shelf_a, reverb_fb_low_shelf_hz, ls_db * abRatio, (float)REVERB_FS_HZ);
        biquad_lowshelf(&pre_low_shelf_b, reverb_fb_low_shelf_hz, ls_db * bbRatio, (float)REVERB_FS_HZ);
        biquad_copy_coeffs(&pre_low_shelf_a_r, &pre_low_shelf_a);
        biquad_copy_coeffs(&pre_low_shelf_b_r, &pre_low_shelf_b);
        last_fb_ls_hz = reverb_fb_low_shelf_hz; last_fb_ls_db = ls_db;
    }
    /* Tone-mapped high shelf sits IN the feedback loop, so a raw coeff jump
     * gets sustained/amplified — smooth the cutoff and use a small threshold
     * (db is fixed, compared directly). */
    if (fabsf(fb_hs_hz_morph - last_fb_hs_hz) > last_fb_hs_hz * 0.0003f ||
        hs_db != last_fb_hs_db) {
        biquad_highshelf(&pre_high_shelf_a, fb_hs_hz_morph, hs_db * abRatio, (float)REVERB_FS_HZ);
        biquad_highshelf(&pre_high_shelf_b, fb_hs_hz_morph, hs_db * bbRatio, (float)REVERB_FS_HZ);
        biquad_copy_coeffs(&pre_high_shelf_a_r, &pre_high_shelf_a);
        biquad_copy_coeffs(&pre_high_shelf_b_r, &pre_high_shelf_b);
        last_fb_hs_hz = fb_hs_hz_morph; last_fb_hs_db = hs_db;
    }

    /* The previous block's line-B write-back must land before anything reads the
     * lines again: the octave heads read as little as OCT_MIN_DELAY behind the
     * write head, which reaches back into the previous block. (The right
     * channel's lines are written back by CPU, at once.) */
    if (pd_wb_pending) {
        pd_wb_pending = 0;
        pd_writeback_finish_now(predelay_b, pd_node_b, pd_wb_idx);
    }
    const uint32_t w0   = predelay_wh;
    const uint32_t widx = w0 & PRE_DELAY_LINE_MASK;   /* this block's write position */

    if (fb <= 0.0f && oct_active == 0 && pitch_mix_target == 0.0f &&
        !freeze_request && freeze_amt == 0.0f &&
        !reverse_request && reverse_amt == 0.0f) {
        /* Bypass: dry input straight to the cascade (loop off). The lines
         * still take the input, which is exactly what the loop would write
         * with no feedback, so the pitch feed heads find fresh audio if the
         * pitch mix is turned up later. */
        for (int n = 0; n < REVERB_BLOCK; n++) {
            float px = id_out[n] * to_unit, pxr = id_out_r[n] * to_unit;
            predelay_out[n] = px;
            predelay_out_r[n] = pxr;
            pd_node_a[n] = px;
            pd_node_b[n] = px;
            pd_node_a_r[n] = pxr;
            pd_node_b_r[n] = pxr;
        }
        pd_writeback_start(predelay_a, pd_node_a, widx);
        pd_writeback_finish_now(predelay_a, pd_node_a, widx);
        pd_writeback_start(predelay_b, pd_node_b, widx);
        pd_wb_idx = widx; pd_wb_pending = 1;
        memcpy(&predelay_a_r[widx], pd_node_a_r, REVERB_BLOCK * sizeof(float));
        memcpy(&predelay_b_r[widx], pd_node_b_r, REVERB_BLOCK * sizeof(float));
        /* Keep the write head advancing so resuming feedback reads sane history. */
        predelay_wh += REVERB_BLOCK;
        if (predelay_wh >= REVERB_IDX_WRAP) predelay_wh -= REVERB_IDX_WRAP;
        return;
    }

    /* Pitch feed heads: start, stop and jumps happen here, at the block
     * boundary, and each head fetches its stretch of line. The rate glides
     * block to block; loop lengths use the modulation as it stands at the
     * block start. The right channel's heads run alongside the left's, in
     * step (same loop, same rate), on its own lines. */
    pitch_semis_morph += PITCH_ALPHA * (pitch_semis_target - pitch_semis_morph);
    float rate = powf(2.0f, pitch_semis_morph * (1.0f / 12.0f));
    if (rate > 2.0f) rate = 2.0f; else if (rate < 0.5f) rate = 0.5f;
    const int rev_on = reverse_request;
    if (rev_on) rate = -rate;                  /* the pitch heads read backwards too */
    const float mixTarget = pitch_mix_target;
    float fa = pre_mod_phase_a; if (fa > PI_F) fa -= TWO_PI_F;
    float fb_ = pre_mod_phase_b; if (fb_ > PI_F) fb_ -= TWO_PI_F;
    const float loopA = timeA + modDepth * (1.0f + fast_sin_pi(fa));
    const float loopB = timeB + modDepth * (1.0f + fast_sin_pi(fb_));
    if (!oct_active && mixTarget > 0.0f) {
        oct_active = 1;
        oct_head_start(&oct_head_a, loopA, rate);
        oct_head_start(&oct_head_b, loopB, rate);
        oct_head_start(&oct_head_a_r, loopA, rate);
        oct_head_start(&oct_head_b_r, loopB, rate);
    } else if (oct_active && mixTarget == 0.0f && pitch_mix_morph == 0.0f) {
        oct_active = 0;
    }
    const int oct_run = oct_active;
    if (!rev_active && rev_on) {
        rev_active = 1;
        oct_head_start(&rev_head_a, loopA, -1.0f);
        oct_head_start(&rev_head_b, loopB, -1.0f);
        oct_head_start(&rev_head_a_r, loopA, -1.0f);
        oct_head_start(&rev_head_b_r, loopB, -1.0f);
    } else if (rev_active && !rev_on && reverse_amt == 0.0f) {
        rev_active = 0;
    }
    const int rev_run = rev_active;
#if defined(DIAG_REVERB_PROFILE) && !defined(VELVET_REVERB_HOST)
    prof_oct_ran = (uint8_t)oct_run;
#endif
    /* Equal-power weights at the block's start and end mix positions, so the
     * per-sample loop only interpolates. */
    const float mix0 = pitch_mix_morph;
    float mix1 = mix0;
    {
        const float maxStep = OCT_FADE_STEP * (float)REVERB_BLOCK;
        float dm = mixTarget - mix0;
        if (dm > maxStep) dm = maxStep; else if (dm < -maxStep) dm = -maxStep;
        mix1 = mix0 + dm;
        pitch_mix_morph = mix1;
    }
    PdMix pm;
    pm.keep0 = cosf(0.5f * PI_F * mix0); pm.shift0 = sinf(0.5f * PI_F * mix0);
    pm.keep1 = cosf(0.5f * PI_F * mix1); pm.shift1 = sinf(0.5f * PI_F * mix1);
    if (oct_run) {
        oct_head_begin_block(&oct_head_a, predelay_a, w0, loopA, rate);
        oct_head_begin_block(&oct_head_b, predelay_b, w0, loopB, rate);
        oct_head_begin_block(&oct_head_a_r, predelay_a_r, w0, loopA, rate);
        oct_head_begin_block(&oct_head_b_r, predelay_b_r, w0, loopB, rate);
    }
    if (rev_run) {
        oct_head_begin_block(&rev_head_a, predelay_a, w0, loopA, -1.0f);
        oct_head_begin_block(&rev_head_b, predelay_b, w0, loopB, -1.0f);
        oct_head_begin_block(&rev_head_a_r, predelay_a_r, w0, loopA, -1.0f);
        oct_head_begin_block(&rev_head_b_r, predelay_b_r, w0, loopB, -1.0f);
    }

    const float wbase = (float)widx;

    /* Ducking, per sample, shared by all four lines: it follows the louder
     * side, so the two sides duck together and the image holds. */
    for (int n = 0; n < REVERB_BLOCK; n++) {
        float px = id_out[n] * to_unit, pxr = id_out_r[n] * to_unit;
        float rect = px < 0.0f ? -px : px;
        float rectr = pxr < 0.0f ? -pxr : pxr;
        if (rectr > rect) rect = rectr;
        if (rect > duck_env) duck_env += duck_atk_coeff * (rect - duck_env);
        else                 duck_env += duckRelCoef   * (rect - duck_env);
        float duck = duckAmount * duck_env; if (duck > 1.0f) duck = 1.0f;
        if (freeze_request) { freeze_amt += 1.0f / FREEZE_FADE_SAMPLES; if (freeze_amt > 1.0f) freeze_amt = 1.0f; }
        else                { freeze_amt -= 1.0f / FREEZE_FADE_SAMPLES; if (freeze_amt < 0.0f) freeze_amt = 0.0f; }
        pd_frz[n]   = freeze_amt;
        if (rev_on) { reverse_amt += 1.0f / FREEZE_FADE_SAMPLES; if (reverse_amt > 1.0f) reverse_amt = 1.0f; }
        else        { reverse_amt -= 1.0f / FREEZE_FADE_SAMPLES; if (reverse_amt < 0.0f) reverse_amt = 0.0f; }
        pd_rev[n]   = reverse_amt;
        pd_px[n]    = px * (1.0f - freeze_amt);   /* frozen: no new input */
        pd_px_r[n]  = pxr * (1.0f - freeze_amt);
        pd_fbeff[n] = fb * (1.0f - duck);
    }

    /* Line A, then line B while A's write-back runs. The right channel's
     * lines take the same modulation, from the same starting phases. */
    float phA_r = pre_mod_phase_a, phB_r = pre_mod_phase_b;
    {
        float rA[REVERB_BLOCK], rB[REVERB_BLOCK], rAr[REVERB_BLOCK], rBr[REVERB_BLOCK];
        pd_line_read(&pre_mod_phase_a, &pre_rd_a, predelay_a, wbase, &pre_len_a, modDepth, modInc, rA);
        pd_line_read(&pre_mod_phase_b, &pre_rd_b, predelay_b, wbase, &pre_len_b, modDepth, modInc, rB);
        pd_line_read(&phA_r, &pre_rd_a_r, predelay_a_r, wbase, &pre_len_a_r, modDepth, modInc, rAr);
        pd_line_read(&phB_r, &pre_rd_b_r, predelay_b_r, wbase, &pre_len_b_r, modDepth, modInc, rBr);
        const float blur = LOOP_BLUR * (1.0f - loop_thin_morph);
        pd_line_node(&pre_low_shelf_a, &pre_high_shelf_a, &pre_hp_a, rA, rB, 1.0f, blur, pd_px, fbScaleA, pd_node_a);
        pd_line_node(&pre_low_shelf_b, &pre_high_shelf_b, &pre_hp_b, rB, rA, -1.0f, blur, pd_px, fbScaleB, pd_node_b);
        pd_line_node(&pre_low_shelf_a_r, &pre_high_shelf_a_r, &pre_hp_a_r, rAr, rBr, 1.0f, blur, pd_px_r, fbScaleA, pd_node_a_r);
        pd_line_node(&pre_low_shelf_b_r, &pre_high_shelf_b_r, &pre_hp_b_r, rBr, rAr, -1.0f, blur, pd_px_r, fbScaleB, pd_node_b_r);
    }
    /* Cross-feed (STEREO_LOOP_XFEED), held off while frozen so a frozen loop
     * keeps exactly what it holds. */
    for (int n = 0; n < REVERB_BLOCK; n++) {
        const float k = STEREO_LOOP_XFEED * (1.0f - pd_frz[n]);
        const float al = pd_node_a[n], ar = pd_node_a_r[n];
        const float bl = pd_node_b[n], br = pd_node_b_r[n];
        pd_node_a[n] = al + k * (ar - al); pd_node_a_r[n] = ar + k * (al - ar);
        pd_node_b[n] = bl + k * (br - bl); pd_node_b_r[n] = br + k * (bl - br);
    }
    pd_writeback_start(predelay_a, pd_node_a, widx);
    pd_writeback_finish_now(predelay_a, pd_node_a, widx);
    pd_writeback_start(predelay_b, pd_node_b, widx);   /* finished at the next block's start */
    pd_wb_idx = widx; pd_wb_pending = 1;
    memcpy(&predelay_a_r[widx], pd_node_a_r, REVERB_BLOCK * sizeof(float));
    memcpy(&predelay_b_r[widx], pd_node_b_r, REVERB_BLOCK * sizeof(float));

    if (oct_run) {
        oct_head_land(&oct_head_a);
        oct_head_land(&oct_head_b);
        oct_head_land(&oct_head_a_r);
        oct_head_land(&oct_head_b_r);
        oct_head_run(&oct_head_a, w0, rate, oct_out_a);
        oct_head_run(&oct_head_b, w0, rate, oct_out_b);
        oct_head_run(&oct_head_a_r, w0, rate, oct_out_a_r);
        oct_head_run(&oct_head_b_r, w0, rate, oct_out_b_r);
    }
    if (rev_run) {
        oct_head_land(&rev_head_a);
        oct_head_land(&rev_head_b);
        oct_head_land(&rev_head_a_r);
        oct_head_land(&rev_head_b_r);
        oct_head_run(&rev_head_a, w0, -1.0f, rev_out_a);
        oct_head_run(&rev_head_b, w0, -1.0f, rev_out_b);
        oct_head_run(&rev_head_a_r, w0, -1.0f, rev_out_a_r);
        oct_head_run(&rev_head_b_r, w0, -1.0f, rev_out_b_r);
    }

    /* Mix into the cascade feed, plus the pitch heads. Line A's share fades
     * out as Density thins the echoes, B making up the level (the lines are
     * uncorrelated, so their powers add). */
    pm.mix = mix;
    pm.wA = 1.0f - loop_thin_morph; pm.wB = sqrtf(2.0f - pm.wA * pm.wA);
    /* Unfrozen, the input reaches the feed directly by (1 - mix) and through
     * both line nodes by mix: this is its total. Frozen, it is fed at that. */
    pm.dryK = 1.0f - mix + mix * (pm.wA + pm.wB);
    pm.oct_run = oct_run; pm.rev_run = rev_run;
    loop_drift_step();
    pd_mix(&pm, pd_px, id_out, pd_node_a, pd_node_b, oct_out_a, oct_out_b, rev_out_a, rev_out_b,
           LOOP_DRIFT * (loop_lfo[1] - loop_lfo[0]), predelay_out);
    pd_mix(&pm, pd_px_r, id_out_r, pd_node_a_r, pd_node_b_r, oct_out_a_r, oct_out_b_r, rev_out_a_r, rev_out_b_r,
           LOOP_DRIFT * (loop_lfo[3] - loop_lfo[2]), predelay_out_r);
    predelay_wh += REVERB_BLOCK;
    if (predelay_wh >= REVERB_IDX_WRAP) predelay_wh -= REVERB_IDX_WRAP;
}

static void do_input_write_t0(void)
{
    /* Attenuate the (pre-delayed) input into the cascade by REVERB_HEADROOM so
     * the whole int16 chain runs cooler; the output stage makes it back up.
     * predelay_out is normalised ±1 but blooms well past it during sustain
     * (the pre-delay node clamp is ±4, so predelay_out can reach ~±5). Use the
     * SAME soft knee as every other stage bridge (soft_clip_int16) rather than
     * a hard clamp: a hard int16 clamp injects an edge that the cascade diffuses
     * into an audible "whoosh", whereas the soft knee rounds those occasional
     * overs off smoothly. */
    for (int i = 0; i < REVERB_BLOCK; i++) {
        uint32_t wi = ring_write_idx + (uint32_t)i;
        float v = predelay_out[i] * (32768.0f * REVERB_HEADROOM);
#ifdef VELVET_REVERB_HOST
        /* Host-only: the soft knee hides how hard this stage is being driven, so
         * record it. Saturation here is a prime whoosh suspect (see above), and
         * it is level-dependent, which a fixed-level test would never reveal. */
        float mag = v < 0.0f ? -v : v;
        if (mag > host_dbg_t0_in_peak) host_dbg_t0_in_peak = mag;
        if (mag > SOFT_KNEE_T * 32768.0f) host_dbg_t0_sat_count++;   /* knee, not over */
#endif
        t0_ring[wi & T0_RING_MASK] = soft_clip_int16(v);
        t0_ring_r[wi & T0_RING_MASK] = soft_clip_int16(predelay_out_r[i] * (32768.0f * REVERB_HEADROOM));
    }
    ring_guard_sync(t0_ring, T0_RING_SAMPLES);
    ring_guard_sync(t0_ring_r, T0_RING_SAMPLES);
    block_write_idx = ring_write_idx;
    ring_write_idx += REVERB_BLOCK;
    /* Keep the free-running index in float-exact range. block_write_idx is cast to
     * float in the recirc/global read-position math (pos0 = (float)block_write_idx
     * - modOff); past 2^24 a float32 can't represent consecutive sample indices, so
     * the feedback tap positions quantize — audible as a slow "bit-crush" / sample-
     * rate-reduction after ~12 min of running. Wrap at 2^23, a multiple of every
     * (power-of-2) ring size, so the masked ring positions stay perfectly continuous
     * across the wrap and floats stay exact indefinitely. */
    if (ring_write_idx >= REVERB_IDX_WRAP) ring_write_idx -= REVERB_IDX_WRAP;

    for (int i = 0; i < REVERB_BLOCK; i++) accT0[i] = accT0_r[i] = 0;
}

/* Fast (SIMD pair-read) T0 convolution — used when the per-tap LFO is off. */
/* Wrap-aware Q15 SIMD accumulation of one delay-line tap into accT0:
 * accT0[n] += t0_ring[(base + n) & mask] * gain  for n in [0, REVERB_BLOCK).
 * `base` must already be masked into [0, T0_RING_SAMPLES). Shared by the fast
 * (unmodulated) path and the modulated path's two interpolation passes. */
static void accumulate_t0(const int16_t *ring, int32_t *acc, uint32_t base, uint32_t gain)
{
    /* Common case: the whole block reads contiguously without wrapping the
     * ring. Keep this a tight, fixed-count SIMD loop (REVERB_BLOCK is a small
     * even constant) so the compiler fully unrolls it — this is the hot path. */
    if (T0_RING_SAMPLES - base >= (uint32_t)REVERB_BLOCK) {
        const int16_t *src = &ring[base];
        for (int i = 0; i < REVERB_BLOCK; i += 2) {
            uint32_t s_pair = *((const u32_alias *)(src + i));
            acc[i]   = qadd_sat(acc[i],   smulbb(s_pair, gain));
            acc[i+1] = qadd_sat(acc[i+1], smultb(s_pair, gain));
        }
        return;
    }
    /* Rare case: the read window straddles the ring wrap. Segment 1 is the
     * contiguous run of n1 samples from base; segment 2 reads from ring start.
     * Pairs are only read within a segment, so a pair never straddles the wrap
     * (which would over-read / over-write acc); the odd boundary sample of
     * each segment is handled with a scalar multiply. */
    uint32_t n1 = T0_RING_SAMPLES - base;
    int i = 0;
    for (; i + 1 < (int)n1; i += 2) {
        uint32_t s_pair = *((const u32_alias *)(&ring[base + (uint32_t)i]));
        acc[i]   = qadd_sat(acc[i],   smulbb(s_pair, gain));
        acc[i+1] = qadd_sat(acc[i+1], smultb(s_pair, gain));
    }
    if (i < (int)n1) {
        acc[i] = qadd_sat(acc[i], smulbb((uint32_t)(uint16_t)ring[base + (uint32_t)i], gain));
        i++;
    }
    for (; i + 1 < REVERB_BLOCK; i += 2) {
        uint32_t s_pair = *((const u32_alias *)(&ring[(uint32_t)i - n1]));
        acc[i]   = qadd_sat(acc[i],   smulbb(s_pair, gain));
        acc[i+1] = qadd_sat(acc[i+1], smultb(s_pair, gain));
    }
    if (i < REVERB_BLOCK) {
        acc[i] = qadd_sat(acc[i], smulbb((uint32_t)(uint16_t)ring[(uint32_t)i - n1], gain));
    }
}

REVERB_CONV_FN static void do_t0_phase_fast(const int16_t *gains, int32_t *accl, int32_t *accr)
{
    int tapEnd = t0TapCount;
    for (int t = 0; t < tapEnd; t++) {
        uint32_t gain = (uint32_t)(int32_t)gains[t];
        uint32_t base = (block_write_idx - (uint32_t)effOffT0[t]) & T0_RING_MASK;
        accumulate_t0(t0_ring, accl, base, gain);
        accumulate_t0(t0_ring_r, accr, base, gain);
    }
}

/* Modulated T0: each tap's delay is pushed later by a per-tap LFO (one shared
 * phase, rotated by the tap's golden-angle offset) so the early reflections
 * shimmer, Lexicon-style. The extra delay d = depth * (1 + sin) is at least 0,
 * so reads stay in the past.
 *
 * The delay moves sample by sample: pass 1 works out each tap's linear-
 * interpolation weights at the block's start and at its end (the next block's
 * start), and pass 2 slides the weights between them across the block. It
 * used to hold each tap's delay for the whole block and step it at the next,
 * 1500 times a second, which put tones at multiples of 1500 Hz either side of
 * everything played (a 440 Hz sine got 1060, 1940, 2560, 4060 Hz ... about
 * 50 dB under it): heard as a sample-rate-reduction grit on sustained notes.
 *
 * A tap whose delay crosses a whole sample during the block reads three
 * samples instead of two. Its weights at each end are the linear
 * interpolation's (a triangle centred on the delay), so the read is exact at
 * both ends and in between is a blend of the two, with no step anywhere.
 *
 * Weights are integers in the tap gain's scale. Most taps stay within one
 * sample over a block; they read two samples, with both weights packed in one
 * word and stepped by a packed change each sample, so a tap still costs one
 * dual multiply-accumulate a sample (the change is rounded towards zero,
 * which leaves the last sample up to 15 units of gain short of the next
 * block's start). The few that cross a sample (one tap-block in twenty at the
 * stock depth, one in five at full modulation) run a plain loop, their
 * weights carried in Q4 so the end lands exactly. */
#define T0_MOD_WIN  3
static uint32_t t0_mod_base[MAX_T0_TAPS] CCM_ATTR;   /* oldest sample read, at the block's first sample */
/* Two-sample taps, first in the list: {older, newer} weights at the block's
 * start, their change a sample, and pass 2's running copy. */
static uint32_t t0_mod_pw[MAX_T0_TAPS] CCM_ATTR;
static uint32_t t0_mod_pd[MAX_T0_TAPS] CCM_ATTR;
static uint32_t t0_mod_cur[MAX_T0_TAPS] CCM_ATTR;
/* Three-sample taps, after them: weights at the start and change, Q4. */
static int32_t  t0_mod_w0[MAX_T0_TAPS][T0_MOD_WIN] CCM_ATTR;
static int32_t  t0_mod_dw[MAX_T0_TAPS][T0_MOD_WIN] CCM_ATTR;

/* The linear interpolation's weight for the sample j samples later than a
 * read at extra delay d: a triangle, 1 at d = j. */
static inline float t0_hat(float d, float j)
{
    float x = d - j; if (x < 0.0f) x = -x;
    return x < 1.0f ? 1.0f - x : 0.0f;
}

static inline uint32_t pack16(int32_t lo, int32_t hi)
{
    return ((uint32_t)(uint16_t)hi << 16) | (uint32_t)(uint16_t)lo;
}

REVERB_CONV_FN static void t0_mod_pass2(const int16_t *ring, int32_t *acc, int n2, int n);

REVERB_CONV_FN static void do_t0_phase_mod(float depth0, float depth1, float inc,
                                           const int16_t *gains, int32_t *accl, int32_t *accr)
{
    const int tapEnd = t0TapCount;
    const float ph0 = t0_tap_mod_phase, ph1 = ph0 + (float)REVERB_BLOCK * inc;
    const float sinP0 = sinf(ph0), cosP0 = cosf(ph0);
    const float sinP1 = sinf(ph1), cosP1 = cosf(ph1);

    /* Pass 1: per-tap float work, once per block: two-sample taps from the
     * front of the list, three-sample taps from the back. */
    int n2 = 0, n3 = MAX_T0_TAPS;
    for (int t = 0; t < tapEnd; t++) {
        const int32_t g = (int32_t)gains[t];
        if (g == 0) continue;
        const float d0 = depth0 * (1.0f + sinP0 * t0TapModCos[t] + cosP0 * t0TapModSin[t]);
        const float d1 = depth1 * (1.0f + sinP1 * t0TapModCos[t] + cosP1 * t0TapModSin[t]);
        const float dlo = d0 < d1 ? d0 : d1, dhi = d0 < d1 ? d1 : d0;
        int32_t lo = (int32_t)dlo; if (lo < 0) lo = 0;     /* floor (d >= 0) */
        /* A block moves the delay by well under a sample (at most ~0.15 at
         * full depth), so it crosses at most one whole sample. */
        const int win = ((int32_t)dhi > lo) ? 3 : 2;
        const uint32_t base = (block_write_idx - (uint32_t)effOffT0[t] - (uint32_t)(lo + win - 1)) & T0_RING_MASK;
        /* Window slot m holds the sample at extra delay lo + win-1-m: slot 0
         * is the oldest. */
        if (win == 2) {
            const float j0 = (float)(lo + 1), j1 = (float)lo;
            const int32_t a0 = (int32_t)((float)g * t0_hat(d0, j0)), a1 = (int32_t)((float)g * t0_hat(d1, j0));
            const int32_t b0 = (int32_t)((float)g * t0_hat(d0, j1)), b1 = (int32_t)((float)g * t0_hat(d1, j1));
            t0_mod_base[n2] = base;
            t0_mod_pw[n2] = pack16(a0, b0);
            t0_mod_pd[n2] = pack16((a1 - a0) / REVERB_BLOCK, (b1 - b0) / REVERB_BLOCK);
            n2++;
        } else {
            --n3;
            t0_mod_base[n3] = base;
            for (int m = 0; m < T0_MOD_WIN; m++) {
                const float j = (float)(lo + 2 - m);
                const int32_t w0 = (int32_t)((float)g * t0_hat(d0, j) * 16.0f);
                const int32_t w1 = (int32_t)((float)g * t0_hat(d1, j) * 16.0f);
                t0_mod_w0[n3][m] = w0;
                t0_mod_dw[n3][m] = (w1 - w0) / REVERB_BLOCK;
            }
        }
    }

    t0_mod_pass2(t0_ring, accl, n2, n3);
    t0_mod_pass2(t0_ring_r, accr, n2, n3);
}

/* Pass 2 of do_t0_phase_mod for one channel's ring, from pass 1's taps:
 * two-sample taps [0, n2), three-sample taps [n3, MAX_T0_TAPS). The ring's
 * guard region covers REVERB_BLOCK + 1 samples past a window's start, enough
 * for every two-sample window; a three-sample window starting in the ring's
 * last sample needs one more, and is read through the mask. */
REVERB_CONV_FN static void t0_mod_pass2(const int16_t *ring, int32_t *acc, int n2, int n3)
{
    for (int i = 0; i < n2; i++) t0_mod_cur[i] = t0_mod_pw[i];
    for (int c = 0; c < REVERB_BLOCK; c += REVERB_ACC_CHUNK) {
        int32_t a0 = acc[c], a1 = acc[c+1], a2 = acc[c+2], a3 = acc[c+3];
        for (int i = 0; i < n2; i++) {
            const int16_t *src = &ring[t0_mod_base[i] + (uint32_t)c];
            const uint32_t d = t0_mod_pd[i];
            uint32_t w = t0_mod_cur[i];
            a0 = qadd_sat(a0, smlad(*((const u32_alias *)(src + 0)), w, 0)); w = sadd16(w, d);
            a1 = qadd_sat(a1, smlad(*((const u32_alias *)(src + 1)), w, 0)); w = sadd16(w, d);
            a2 = qadd_sat(a2, smlad(*((const u32_alias *)(src + 2)), w, 0)); w = sadd16(w, d);
            a3 = qadd_sat(a3, smlad(*((const u32_alias *)(src + 3)), w, 0)); w = sadd16(w, d);
            t0_mod_cur[i] = w;
        }
        acc[c] = a0; acc[c+1] = a1; acc[c+2] = a2; acc[c+3] = a3;
    }
    for (int i = n3; i < MAX_T0_TAPS; i++) {
        const uint32_t base = t0_mod_base[i];
        int32_t wa = t0_mod_w0[i][0], wb = t0_mod_w0[i][1], wc = t0_mod_w0[i][2];
        const int32_t da = t0_mod_dw[i][0], db = t0_mod_dw[i][1], dc = t0_mod_dw[i][2];
        if (base + (uint32_t)REVERB_BLOCK + 1u < (uint32_t)(T0_RING_SAMPLES + REVERB_BLOCK)) {
            const int16_t *src = &ring[base];
            for (int k = 0; k < REVERB_BLOCK; k++) {
                const int32_t y = (wa >> 4) * src[k] + (wb >> 4) * src[k + 1] + (wc >> 4) * src[k + 2];
                acc[k] = qadd_sat(acc[k], y);
                wa += da; wb += db; wc += dc;
            }
        } else {
            for (int k = 0; k < REVERB_BLOCK; k++) {
                const int32_t y = (wa >> 4) * ring[(base + (uint32_t)k) & T0_RING_MASK]
                                + (wb >> 4) * ring[(base + (uint32_t)k + 1u) & T0_RING_MASK]
                                + (wc >> 4) * ring[(base + (uint32_t)k + 2u) & T0_RING_MASK];
                acc[k] = qadd_sat(acc[k], y);
                wa += da; wb += db; wc += dc;
            }
        }
    }
}

static void do_t0_phase(void)
{
    const float share = mod_depth_share(mod_amount_morph);
    const float depth = share * share * T0_TAP_MOD_MAX_SAMPLES;
    if (t0_ramp_left > 0)
        for (int i = 0; i < REVERB_BLOCK; i++) accT0_p[i] = accT0_r_p[i] = 0;
    if (depth <= 0.0f) {
        do_t0_phase_fast(effGains_t0, accT0, accT0_r);
        if (t0_ramp_left > 0) do_t0_phase_fast(effGains_t0_prev, accT0_p, accT0_r_p);
        t0_mod_depth_last = -1.0f;
        return;
    }
    const float depth0 = t0_mod_depth_last >= 0.0f ? t0_mod_depth_last : depth;
    float rate = reverb_t0_tap_mod_rate; if (rate < 0.0f) rate = 0.0f;
    float inc = TWO_PI_F * rate / (float)REVERB_FS_HZ;
    do_t0_phase_mod(depth0, depth, inc, effGains_t0, accT0, accT0_r);
    if (t0_ramp_left > 0) do_t0_phase_mod(depth0, depth, inc, effGains_t0_prev, accT0_p, accT0_r_p);
    t0_mod_depth_last = depth;
    /* Advance the shared LFO phase by the block. */
    t0_tap_mod_phase += (float)REVERB_BLOCK * inc;
    t0_tap_mod_phase = fmodf(t0_tap_mod_phase, TWO_PI_F);
    if (t0_tap_mod_phase < 0.0f) t0_tap_mod_phase += TWO_PI_F;
}

static void do_bridge_t0_to_t1(void)
{
    for (int i = 0; i < REVERB_BLOCK; i++) {
        uint32_t wi = block_write_idx + (uint32_t)i;
#ifdef VELVET_REVERB_HOST
        host_note_sat(accT0[i], &host_dbg_t01_peak, &host_dbg_t01_sat);
#endif
        if (t0_ramp_left > 0) {
            const float f = ramp_frac(t0_ramp_left, i);
            t1_ring[wi & T1_RING_MASK] = to_ring16(soft_clip_unit(acc_blend(accT0_p[i], t0_makeup_prev, accT0[i], t0_makeup, f)) * 32767.0f);
            t1_ring_r[wi & T1_RING_MASK] = to_ring16(soft_clip_unit(acc_blend(accT0_r_p[i], t0_makeup_prev, accT0_r[i], t0_makeup, f)) * 32767.0f);
        } else {
            t1_ring[wi & T1_RING_MASK] = soft_saturate_q15_g(accT0[i], t0_makeup);
            t1_ring_r[wi & T1_RING_MASK] = soft_saturate_q15_g(accT0_r[i], t0_makeup);
        }
        accT1[i] = 0;
        accT1_r[i] = 0;
    }
    t0_ramp_left = t0_ramp_left > REVERB_BLOCK ? t0_ramp_left - REVERB_BLOCK : 0;
    ring_guard_sync(t1_ring, T1_RING_SAMPLES);
    ring_guard_sync(t1_ring_r, T1_RING_SAMPLES);
}

/* Per-tap window start, computed once per block. Same reason as T0's: with the
 * tap loop inside the chunk loop, deriving the base per tap per chunk repeated
 * the subtract and mask four times over — and cost a reload of block_write_idx
 * from the stack each time, which was enough register pressure to push two of
 * the four accumulators out to the stack as well. */
static uint32_t t1_tap_base[MAX_T1_TAPS] CCM_ATTR;

/* One channel's T1, from the window starts in t1_tap_base. */
REVERB_CONV_FN static void t1_conv(const int16_t *ring, int32_t *acc, const int16_t *gains)
{
    const int tapEnd = t1TapCount;

    for (int c = 0; c < REVERB_BLOCK; c += REVERB_ACC_CHUNK) {
        int32_t a0 = acc[c], a1 = acc[c+1], a2 = acc[c+2], a3 = acc[c+3];

        for (int t = 0; t < tapEnd; t++) {
            const uint32_t gain = (uint32_t)(int32_t)gains[t];
            /* Guard region makes this contiguous for every base — no wrap split,
             * and no odd-length segment to run off the end of. */
            const int16_t *src = ring + t1_tap_base[t] + c;
            uint32_t p0 = *((const u32_alias *)(src));
            uint32_t p1 = *((const u32_alias *)(src + 2));
            a0 = qadd_sat(a0, smulbb(p0, gain));
            a1 = qadd_sat(a1, smultb(p0, gain));
            a2 = qadd_sat(a2, smulbb(p1, gain));
            a3 = qadd_sat(a3, smultb(p1, gain));
        }

        acc[c] = a0; acc[c+1] = a1; acc[c+2] = a2; acc[c+3] = a3;
    }
}

REVERB_CONV_FN static void do_t1_phase(void)
{
    const int tapEnd = t1TapCount;
    for (int t = 0; t < tapEnd; t++)
        t1_tap_base[t] = (block_write_idx - (uint32_t)effOffT1[t]) & T1_RING_MASK;
    t1_conv(t1_ring, accT1, effGains_t1);
    t1_conv(t1_ring_r, accT1_r, effGains_t1);
    if (t1_ramp_left > 0) {
        for (int i = 0; i < REVERB_BLOCK; i++) accT1_p[i] = accT1_r_p[i] = 0;
        t1_conv(t1_ring, accT1_p, effGains_t1_prev);
        t1_conv(t1_ring_r, accT1_r_p, effGains_t1_prev);
    }
}

/* T1 -> T2 bridge gain. The DLD saturates here (a tanh, its character); on
 * the Beads that put odd harmonics on everything, the 3rd only 38 dB down for a
 * sine at -20 dBFS and 18 dB down at -6 dBFS, and the cascade's reverberation
 * smeared them into a haze. The bridge is linear now, at the tanh's
 * small-signal gain so levels and decay are as they were, with the soft knee
 * after it for genuine overs. */
#define T2_BRIDGE_GAIN  (1.1f * 1.05f)

static void do_bridge_t1_to_t2(void)
{
    /* T1's output at full scale 1.0 (crossfaded while its gains ramp), the
     * cross-feed between the sides (STEREO_T2_XFEED), then the bridge gain and
     * soft clip into the T2 rings. */
    float vl[REVERB_BLOCK], vr[REVERB_BLOCK];
    for (int i = 0; i < REVERB_BLOCK; i++) {
#ifdef VELVET_REVERB_HOST
        host_note_sat(accT1[i], &host_dbg_t12_peak, &host_dbg_t12_sat);
#endif
        float l, r;
        if (t1_ramp_left > 0) {
            const float f = ramp_frac(t1_ramp_left, i);
            l = acc_blend(accT1_p[i], t1_makeup_prev, accT1[i], t1_makeup, f);
            r = acc_blend(accT1_r_p[i], t1_makeup_prev, accT1_r[i], t1_makeup, f);
        } else {
            l = (float)accT1[i] * (t1_makeup * (1.0f / 1073741824.0f));
            r = (float)accT1_r[i] * (t1_makeup * (1.0f / 1073741824.0f));
        }
        const float d = STEREO_T2_XFEED * (r - l);
        vl[i] = l + d;
        vr[i] = r - d;
    }
    t1_ramp_left = t1_ramp_left > REVERB_BLOCK ? t1_ramp_left - REVERB_BLOCK : 0;
    for (int i = 0; i < REVERB_BLOCK; i += 2) {
        int16_t s0 = to_ring16(soft_clip_unit(vl[i] * T2_BRIDGE_GAIN) * 32767.0f);
        int16_t s1 = to_ring16(soft_clip_unit(vl[i + 1] * T2_BRIDGE_GAIN) * 32767.0f);
        uint32_t packed = ((uint32_t)(uint16_t)s1 << 16) | (uint32_t)(uint16_t)s0;
        uint32_t wi = block_write_idx + (uint32_t)i;
        *((volatile u32_alias *)&t2_ring[wi & T2_RING_MASK]) = packed;
        int16_t r0 = to_ring16(soft_clip_unit(vr[i] * T2_BRIDGE_GAIN) * 32767.0f);
        int16_t r1 = to_ring16(soft_clip_unit(vr[i + 1] * T2_BRIDGE_GAIN) * 32767.0f);
        packed = ((uint32_t)(uint16_t)r1 << 16) | (uint32_t)(uint16_t)r0;
        *((volatile u32_alias *)&t2_ring_r[wi & T2_RING_MASK]) = packed;
    }
    for (int i = 0; i < REVERB_BLOCK; i++) { accL[i] = 0; accR[i] = 0; }
}

static void do_t2_phase(void)
{
    const int count = t2TapCount;
    if (count > 0) {
        t2_gather_wait();                        /* a chain still in flight lands first */
        int h = t2_gather_find(block_write_idx);
#ifdef VELVET_REVERB_HOST
        if (h < 0) host_dbg_t2_gather_late++;
        else if (h == 0) host_dbg_t2_gather_early++;
        else host_dbg_t2_gather_reuse++;
#endif
        if (h < 0) { t2_gather_sync(block_write_idx); h = 0; }
#if defined(DIAG_REVERB_PROFILE) && !defined(VELVET_REVERB_HOST)
        prof_t2_h = h; prof_t2_bw = block_write_idx;
#endif
        t2_conv_blocked(&t2_gather[0][h],           accL, count, effGains_t2);
        t2_conv_blocked(&t2_gather[MAX_T2_TAPS][h], accR, count, T2_GAINS_R);
        if (t2_ramp_left > 0) {
            for (int i = 0; i < REVERB_BLOCK; i++) accL_p[i] = accR_p[i] = 0;
            t2_conv_blocked(&t2_gather[0][h],           accL_p, count, effGains_t2_prev);
            t2_conv_blocked(&t2_gather[MAX_T2_TAPS][h], accR_p, count, effGains_t2R_prev);
        }
    }
}

static void do_finalize(void)
{
    /* Biquad coefficient recompute is expensive (~500 cycles per call —
     * sinf + cosf + division). With macros tied to ADC pots, raw `!=` would
     * recompute every block from sample-rate ADC noise. Hysteresis at 0.5 %
     * of the current freq is well below audibility (≪ ¼ semitone) but
     * caps recompute rate to a handful per second during a smooth sweep. */
    /* Track the SMOOTHED cutoff (lpf_hz_morph), recomputing whenever it has
     * moved a hair. The morph slews continuously (~30 ms tau) so this fires
     * every block during a sweep — each step is tiny and follows a smooth
     * curve (no zipper) — then stops once converged (no idle cost). The 0.03 %
     * threshold is ≪ audibility but coarse enough to halt at convergence. */
    float lpf_target = lpf_hz_morph;
    if (lpf_target < 200.0f) lpf_target = 200.0f;
    if (lpf_target > REVERB_LPF_MAX_HZ) lpf_target = REVERB_LPF_MAX_HZ;
    if (fabsf(lpf_target - last_lpf_hz) > last_lpf_hz * 0.0003f) {
        last_lpf_hz = lpf_target;
        TONE_LPF(&lpf_L, lpf_target, (float)REVERB_CODEC_HZ);
        biquad_copy_coeffs(&lpf_R, &lpf_L);
    }
    float hpf_target = hpf_hz_morph;
    if (hpf_target < 20.0f)   hpf_target = 20.0f;
    if (hpf_target > 2000.0f) hpf_target = 2000.0f;
#if REVERB_OUTPUT_TILT
    if (reverb_tilt_on) {
        if (reverb_tilt_db != last_tilt_db || reverb_tilt_hz != last_tilt_hz) {
            last_tilt_db = reverb_tilt_db; last_tilt_hz = reverb_tilt_hz;
            biquad_tilt(&hpf_L, reverb_tilt_hz, reverb_tilt_db, (float)REVERB_CODEC_HZ);
            biquad_copy_coeffs(&hpf_R, &hpf_L);
        }
        last_hpf_hz = -1.0f;   /* the high-pass is designed again when the tilt goes off */
    } else {
        last_tilt_db = -999.0f;
#endif
    if (fabsf(hpf_target - last_hpf_hz) > last_hpf_hz * 0.0003f) {
        last_hpf_hz = hpf_target;
        biquad_hpf(&hpf_L, hpf_target, (float)REVERB_CODEC_HZ);
        biquad_copy_coeffs(&hpf_R, &hpf_L);
    }
#if REVERB_OUTPUT_TILT
    }
#endif

    uint8_t write_buf = (uint8_t)(blocks_produced % REVERB_OUT_NBUF);
    int16_t *wL = outL[write_buf];
    int16_t *wR = outR[write_buf];
#if REVERB_DECIM == 2
    float pL = prev_out_L;
    float pR = prev_out_R;
#endif

#ifdef DIAG_NO_LIMITER
    /* Baseline for A/B, both sonic and for cycle counting: the pre-limiter output
     * path, writing straight through with no look-ahead buffer, peak scan or gain. */
    for (int i = 0; i < REVERB_BLOCK; i++) {
        float rawL, rawR;
        if (t2_ramp_left > 0) {
            const float f = ramp_frac(t2_ramp_left, i);
            rawL = soft_clip_unit(acc_blend(accL_p[i], t2_makeup_l_prev, accL[i], t2_makeup_l, f)) * (REVERB_OUT_MAKEUP * 32767.0f);
            rawR = soft_clip_unit(acc_blend(accR_p[i], t2_makeup_r_prev, accR[i], t2_makeup_r, f)) * (REVERB_OUT_MAKEUP * 32767.0f);
        } else {
            rawL = soft_clip_unit((float)accL[i] * (t2_makeup_l * (1.0f / 1073741824.0f))) * (REVERB_OUT_MAKEUP * 32767.0f);
            rawR = soft_clip_unit((float)accR[i] * (t2_makeup_r * (1.0f / 1073741824.0f))) * (REVERB_OUT_MAKEUP * 32767.0f);
        }
        float m = 0.5f * (rawL + rawR);
        float s = 0.5f * (rawL - rawR) * REVERB_STEREO_WIDTH;
        float curL = m + s;
        float curR = m - s;
#if REVERB_DECIM == 2
        float midL = (pL + curL) * 0.5f;
        float midR = (pR + curR) * 0.5f;
        wL[2 * i]     = SOFT_CLIP_OUT(biquad_process(&lpf_L, biquad_process(&hpf_L, midL)));
        wR[2 * i]     = SOFT_CLIP_OUT(biquad_process(&lpf_R, biquad_process(&hpf_R, midR)));
        wL[2 * i + 1] = SOFT_CLIP_OUT(biquad_process(&lpf_L, biquad_process(&hpf_L, curL)));
        wR[2 * i + 1] = SOFT_CLIP_OUT(biquad_process(&lpf_R, biquad_process(&hpf_R, curR)));
        pL = curL;
        pR = curR;
#else
        wL[i] = SOFT_CLIP_OUT(biquad_process(&lpf_L, biquad_process(&hpf_L, curL)));
        wR[i] = SOFT_CLIP_OUT(biquad_process(&lpf_R, biquad_process(&hpf_R, curR)));
#endif
    }
#else
    /* Pass 1: compute this block into the look-ahead buffer, tracking its peak.
     * First the stereo pair at the codec rate, before the tone filters, so the
     * wet effects (wet_fx.c) run on it and the tone filters shape what they
     * add. */
    static float fxL[REVERB_BLOCK * REVERB_DECIM], fxR[REVERB_BLOCK * REVERB_DECIM];
    for (int i = 0; i < REVERB_BLOCK; i++) {
        /* Mid-side stereo width on the reverb-rate stereo pair, applied before
         * the 2x upsample interp so the interpolated sample, the original
         * sample, and the carried-over prev all stay width-consistent. */
        /* Make up the internal headroom here (the input was attenuated by
         * REVERB_HEADROOM). soft_saturate_q15 soft-clips the cooled accumulator
         * at full scale first, so the makeup can't reintroduce a hard over. */
#ifdef VELVET_REVERB_HOST
        host_note_sat(accL[i], &host_dbg_out_peak, &host_dbg_out_sat);
        host_note_sat(accR[i], &host_dbg_out_peak, &host_dbg_out_sat);
#endif
        float rawL, rawR;
        if (t2_ramp_left > 0) {
            const float f = ramp_frac(t2_ramp_left, i);
            rawL = soft_clip_unit(acc_blend(accL_p[i], t2_makeup_l_prev, accL[i], t2_makeup_l, f)) * (REVERB_OUT_MAKEUP * (32767.0f / 32768.0f));
            rawR = soft_clip_unit(acc_blend(accR_p[i], t2_makeup_r_prev, accR[i], t2_makeup_r, f)) * (REVERB_OUT_MAKEUP * (32767.0f / 32768.0f));
        } else {
            /* In float, made up before the output is quantised: rounding it to
             * int16 first and then making it up (x4, then the wet gain)
             * left the output with ~13 bits. */
            rawL = soft_clip_unit((float)accL[i] * (t2_makeup_l * (1.0f / 1073741824.0f))) * (REVERB_OUT_MAKEUP * (32767.0f / 32768.0f));
            rawR = soft_clip_unit((float)accR[i] * (t2_makeup_r * (1.0f / 1073741824.0f))) * (REVERB_OUT_MAKEUP * (32767.0f / 32768.0f));
        }
        float m = 0.5f * (rawL + rawR);
        float s = 0.5f * (rawL - rawR) * REVERB_STEREO_WIDTH;
        float curL = m + s;
        float curR = m - s;
#if REVERB_DECIM == 2
        fxL[2 * i] = (pL + curL) * 0.5f;  fxR[2 * i] = (pR + curR) * 0.5f;
        fxL[2 * i + 1] = curL;            fxR[2 * i + 1] = curR;
        pL = curL;
        pR = curR;
#else
        fxL[i] = curL; fxR[i] = curR;
#endif
    }
    wet_fx_process(fxL, fxR, REVERB_BLOCK * REVERB_DECIM);

    float pk = 0.0f;
    float *nL = lim_buf_L[lim_phase];
    float *nR = lim_buf_R[lim_phase];
    /* Local copies of the filters for the loop, so their state stays in
     * registers rather than being stored and reloaded around every write to
     * nL/nR (which the compiler must assume could alias them). The widened
     * side can exceed full scale; the limiter below handles the final range.
     * The peak is of the values actually stored, i.e. after the biquads, so
     * the ceiling is exact and needs no margin. */
    Biquad hL = hpf_L, hR = hpf_R, lL = lpf_L, lR = lpf_R;
    for (int k = 0; k < REVERB_BLOCK * REVERB_DECIM; k++) {
        float oL = biquad_process(&lL, biquad_process(&hL, fxL[k] * (32768.0f * REVERB_OUT_GAIN)));
        float oR = biquad_process(&lR, biquad_process(&hR, fxR[k] * (32768.0f * REVERB_OUT_GAIN)));
        nL[k] = oL; nR[k] = oR;
        float a0 = oL < 0.0f ? -oL : oL;
        float a1 = oR < 0.0f ? -oR : oR;
        if (a0 > pk) pk = a0;
        if (a1 > pk) pk = a1;
    }

    hpf_L.z1 = hL.z1; hpf_L.z2 = hL.z2; hpf_R.z1 = hR.z1; hpf_R.z2 = hR.z2;
    lpf_L.z1 = lL.z1; lpf_L.z2 = lL.z2; lpf_R.z1 = lR.z1; lpf_R.z2 = lR.z2;

    /* Gain for the block just computed. Attack is immediate at block granularity
     * (the ramp below spends the previous block getting there); release eases. */
    const float ceil_i16 = LIM_CEILING * 32768.0f;
    float target = 1.0f;
    if (pk > ceil_i16) target = ceil_i16 / pk;
    /* Release only towards the lowest gain wanted over the last
     * LIM_HOLD_BLOCKS (16 ms). A block is shorter than a bass note's cycle, so
     * without the hold the gain dipped at each of the cycle's peaks and rose
     * between them: a ripple at twice the note's pitch, which put its 3rd
     * harmonic back on a low sine (-40 dB at 110 Hz with a hot input). */
    lim_hold[lim_hold_idx] = target;
    if (++lim_hold_idx >= LIM_HOLD_BLOCKS) lim_hold_idx = 0;
    float held = target;
    for (int i = 0; i < LIM_HOLD_BLOCKS; i++) if (lim_hold[i] < held) held = lim_hold[i];
    if (held < lim_gain_target) lim_gain_target = held;
    else lim_gain_target += LIM_RELEASE * (held - lim_gain_target);

    /* Pass 2: emit the PREVIOUS block, ramping the gain across it so it arrives
     * at the new target exactly as the block that demanded it begins. */
    const float *cL = lim_buf_L[lim_phase ^ 1];
    const float *cR = lim_buf_R[lim_phase ^ 1];
    float g  = lim_gain;
    float dg = (lim_gain_target - lim_gain) * (1.0f / (float)REVERB_OUT_BLOCK);
    for (int k = 0; k < REVERB_OUT_BLOCK; k++) {
        g += dg;
        wL[k] = SOFT_CLIP_OUT(cL[k] * g);
        wR[k] = SOFT_CLIP_OUT(cR[k] * g);
    }
    lim_gain  = lim_gain_target;
    lim_phase ^= 1;
#endif /* DIAG_NO_LIMITER */
#if REVERB_DECIM == 2
    prev_out_L = pL;
    prev_out_R = pR;
#endif

    t2_ramp_left = t2_ramp_left > REVERB_BLOCK ? t2_ramp_left - REVERB_BLOCK : 0;

    /* Publish: the block is fully written before blocks_produced exposes it. */
    __DMB();
    blocks_produced++;
}

/* ==== ISR-side: input buffer fill ==== */
void velvet_reverb_push_sample(int16_t input)
{
    velvet_reverb_push_stereo(input, input);
}

void velvet_reverb_push_stereo(int16_t left, int16_t right)
{
#if REVERB_DECIM == 2
    push_acc += left;
    push_acc_r += right;
    push_phase ^= 1;
    if (push_phase) return;

    int16_t avg = (int16_t)(push_acc >> 1);
    int16_t avg_r = (int16_t)(push_acc_r >> 1);
    push_acc = 0;
    push_acc_r = 0;
#else
    int16_t avg = left, avg_r = right;
#endif

    const uint32_t wseq = in_wr_seq;
    in_ring[wseq % REVERB_IN_NBUF][input_fill_idx] = avg;
    in_ring_r[wseq % REVERB_IN_NBUF][input_fill_idx] = avg_r;
    input_fill_idx++;

    if (input_fill_idx >= REVERB_BLOCK) {
        input_fill_idx = 0;
        /* Keep one slot reserved to fill into, so publishing can never land on
         * the slot poll is reading. Hence NBUF-1 usable depth. */
        uint32_t depth = wseq - in_rd_seq;
        if (depth < (uint32_t)(REVERB_IN_NBUF - 1)) {
            if ((uint8_t)(depth + 1u) > input_queue_max)
                input_queue_max = (uint8_t)(depth + 1u);
#ifndef VELVET_REVERB_HOST
            in_ready_cyc[wseq % REVERB_IN_NBUF] = DWT->CYCCNT;
#endif
            /* Publish only after the samples above are visible. */
            __DMB();
            in_wr_seq = wseq + 1u;
        } else {
            /* Queue full: the main loop is genuinely behind, not merely jittery.
             * The block just filled is discarded, which splices REVERB_BLOCK
             * samples out of the reverb's input, and the cascade diffuses that
             * step into an audible whoosh. With the queue this now takes
             * (REVERB_IN_NBUF - 1) block periods of lateness rather than one, so
             * a nonzero count here means real sustained overload rather than
             * ordinary scheduling jitter. */
            input_drop_count++;
#if !defined(VELVET_REVERB_HOST) && !defined(DIAG_REVERB_PROFILE)
            /* A profile build reports drops as a count instead (prof_step):
             * one event per drop would crowd the link exactly when a variant is
             * overloaded and its numbers matter most. */
            diag_log(DIAG_EVT_REVERB_DROP, 0);
#endif
        }
    }
}

#if defined(DIAG_REVERB_PROFILE) && !defined(VELVET_REVERB_HOST)
/* ==== Profile build: variant cycler ====
 *
 * One upload answers several questions: the build steps through runtime
 * variants of the optimisations, PROF_VARIANT_SECONDS each, and tags the stream
 * with which one is running (DIAG_VARIANT_MARK on the drop slot, every half
 * second) so the decoder can report each separately. Keep test/diag_decode.py's
 * VARIANTS table in step with this one.
 *
 * Load is worst case and repeatable: both macros sweep continuously (so the
 * effGains recompute runs as it would under constant modulation), on triangles
 * with incommensurate periods, and the panel's macro and octave controls are
 * ignored.
 *
 * The prefetch variants turn on the flash prefetch buffer (FLASH_ACR PRFTEN),
 * which SystemInit leaves off. It changes no output, only how fast code runs from
 * flash, and the codec ISR runs with it too. It was suspected, never shown, of
 * playing a part in the clicks heard while moving to 180 MHz, so listen for
 * channel-1 clicks in those stretches. */
#ifndef PROF_VARIANT_SECONDS
#define PROF_VARIANT_SECONDS 20u
#endif
static const struct { int8_t octave; uint8_t prefetch; } prof_variants[] = {
    { 0, 0 },   /* 0: everything on (what ships)  */
    { 0, 1 },   /* 1: ... with flash prefetch     */
    { 1, 0 },   /* 2: octave up                   */
    { 1, 1 },   /* 3: octave up, flash prefetch   */
};
#define PROF_NVARIANTS  ((uint32_t)(sizeof prof_variants / sizeof prof_variants[0]))
#define PROF_BLOCKS_PER_S  (REVERB_FS_HZ / REVERB_BLOCK)

/* Each CPU accounting category's share of the cycles since the last report
 * (see diag_log.h). The first call only takes the starting point. */
/* Runs at the end of a sampled block, after its stage timers. Everything compared
 * is final by then: T2 windows and octave stretches are older than this block's
 * writes, line A's write-back has been waited for, and the next T2 fetch has not
 * started. Eight T2 rows a time, in rotation, to keep the CPU reads short. */
static void prof_verify_dma(void)
{
    if (prof_t2_h >= 0) {
        static int r0 = 0;
        for (int k = 0; k < 8; k++) {
            const int r = (r0 + k) % T2_GATHER_ROWS;
            const int t = (r < MAX_T2_TAPS) ? r : r - MAX_T2_TAPS;
            if (t >= t2TapCount) continue;
            const uint32_t idx = (prof_t2_bw - t2_gather_off[r]) & T2_RING_MASK;
            for (int i = 0; i < REVERB_BLOCK; i++)
                if (t2_gather[r][prof_t2_h + i] != t2_row_ring(r)[(idx + (uint32_t)i) & T2_RING_MASK]) prof_dma_bad++;
            prof_dma_checks++;
        }
        r0 = (r0 + 8) % T2_GATHER_ROWS;
        prof_t2_h = -1;
    }
    if (pd_wb_pending) {                     /* this block's line-A write-back */
        for (int i = 0; i < REVERB_BLOCK; i++)
            if (predelay_a[pd_wb_idx + (uint32_t)i] != pd_node_a[i]) prof_dma_bad++;
        prof_dma_checks++;
    }
    if (prof_oct_ran) {
        const PdFetch *fs[4] = { &oct_head_a.f, &oct_head_a.xf, &oct_head_b.f, &oct_head_b.xf };
        for (int j = 0; j < 4; j++) {
            const PdFetch *f = fs[j];
            if (!f->dst) continue;
            if ((j & 1) && ((j < 2 ? oct_head_a.xleft : oct_head_b.xleft) <= 0)) continue;
            for (uint32_t k = 0; k < f->len; k++)
                if (f->dst[k] != f->line[(f->s + k) & PRE_DELAY_LINE_MASK]) prof_dma_bad++;
            prof_dma_checks++;
        }
        prof_oct_ran = 0;
    }
}

static void prof_report_cpu(void)
{
    static const uint8_t tags[DIAG_CPU_COUNT] = DIAG_CPU_TAGS;
    static uint32_t last_cyc, last_calls, last_ovr, last[DIAG_CPU_COUNT];
    static uint8_t primed = 0;
    uint32_t now[DIAG_CPU_COUNT], calls, cyc;

    uint32_t pm = __get_PRIMASK();
    __disable_irq();
    cyc = DWT->CYCCNT;
    calls = diag_irq_calls;
    for (int c = 0; c < DIAG_CPU_COUNT; c++) now[c] = diag_cpu_cycles[c];
    if (!pm) __enable_irq();

    if (primed) {
        const uint32_t elapsed = cyc - last_cyc;
        for (int c = 0; c < DIAG_CPU_COUNT; c++) {
            uint32_t share = (uint32_t)(((uint64_t)(now[c] - last[c]) * 10000u) / elapsed);
            diag_log(DIAG_EVT_REVERB_DROP, DIAG_SHARE(tags[c], share > 0xFFFFu ? 0xFFFFu : share));
        }
        uint32_t rate = (uint32_t)(((uint64_t)(calls - last_calls) * (SystemCoreClock / 4u)) / elapsed);
        diag_log(DIAG_EVT_REVERB_DROP, DIAG_SHARE(DIAG_TAG_IRQ_RATE, rate > 0xFFFFu ? 0xFFFFu : rate));
        uint32_t ovr = adc_overrun_count - last_ovr;
        if (ovr) diag_log(DIAG_EVT_REVERB_DROP, DIAG_SHARE(DIAG_TAG_ADC_OVR, ovr > 0xFFFFu ? 0xFFFFu : ovr));
        diag_log(DIAG_EVT_REVERB_DROP, DIAG_SHARE(DIAG_TAG_DMA_CHECKS, prof_dma_checks > 0xFFFFu ? 0xFFFFu : prof_dma_checks));
        if (prof_dma_bad)
            diag_log(DIAG_EVT_REVERB_DROP, DIAG_SHARE(DIAG_TAG_DMA_BAD, prof_dma_bad > 0xFFFFu ? 0xFFFFu : prof_dma_bad));
        prof_dma_checks = 0;
        prof_dma_bad = 0;
    }
    primed = 1;
    last_cyc = cyc;
    last_calls = calls;
    last_ovr = adc_overrun_count;
    for (int c = 0; c < DIAG_CPU_COUNT; c++) last[c] = now[c];
}

/* The handover and pipeline counters (see diag_log.h), each since the last
 * report. Counts are sent only when nonzero, the panel only when it has moved. */
static void prof_report_live(void)
{
    static uint32_t last_miss_l, last_miss_r, last_rep, last_skip, last_late;
    uint32_t pm = __get_PRIMASK();
    __disable_irq();
    const uint32_t ml = output_miss_count, mr = output_miss_count_r;
    const uint32_t rep = diag_ho_repeat, skip = diag_ho_skip, late = diag_codec_late;
    const uint32_t pmin = diag_ho_phase_min, pmax = diag_ho_phase_max;
    diag_ho_phase_min = 0xFFFFFFFFu;
    diag_ho_phase_max = 0;
    if (!pm) __enable_irq();

    diag_log(DIAG_EVT_REVERB_DROP, DIAG_LIVE(DIAG_LIVE_QUEUE_MAX, prof_queue_max));
    prof_queue_max = 0;
    if (ml != last_miss_l) diag_log(DIAG_EVT_REVERB_DROP, DIAG_LIVE(DIAG_LIVE_MISS_L, ml - last_miss_l));
    if (mr != last_miss_r) diag_log(DIAG_EVT_REVERB_DROP, DIAG_LIVE(DIAG_LIVE_MISS_R, mr - last_miss_r));
    if (rep != last_rep)   diag_log(DIAG_EVT_REVERB_DROP, DIAG_LIVE(DIAG_LIVE_HO_REPEAT, rep - last_rep));
    if (skip != last_skip) diag_log(DIAG_EVT_REVERB_DROP, DIAG_LIVE(DIAG_LIVE_HO_SKIP, skip - last_skip));
    if (late != last_late) diag_log(DIAG_EVT_REVERB_DROP, DIAG_LIVE(DIAG_LIVE_CODEC_LATE, late - last_late));
    last_miss_l = ml; last_miss_r = mr; last_rep = rep; last_skip = skip; last_late = late;
    if (pmax) {
        const uint32_t cyc_per_us = SystemCoreClock / 1000000u;
        diag_log(DIAG_EVT_REVERB_DROP, DIAG_LIVE(DIAG_LIVE_PHASE_MIN, pmin / cyc_per_us));
        diag_log(DIAG_EVT_REVERB_DROP, DIAG_LIVE(DIAG_LIVE_PHASE_MAX, pmax / cyc_per_us));
    }
#ifdef DIAG_PROFILE_LIVE
    static uint32_t last_decay = 0xFFFFu, last_tone = 0xFFFFu, last_oct = 0xFFFFu;
    const uint32_t decay = (uint32_t)(reverb_macros[MACRO_DECAY].value * 4095.0f);
    const uint32_t tone  = (uint32_t)(reverb_macros[MACRO_TONE].value * 4095.0f);
    const uint32_t oct   = (pitch_mix_target == 0.0f) ? 1u : (pitch_semis_target > 0.0f ? 2u : 0u);
    if (decay != last_decay) diag_log(DIAG_EVT_REVERB_DROP, DIAG_LIVE(DIAG_LIVE_DECAY, decay));
    if (tone != last_tone)   diag_log(DIAG_EVT_REVERB_DROP, DIAG_LIVE(DIAG_LIVE_TONE, tone));
    if (oct != last_oct)     diag_log(DIAG_EVT_REVERB_DROP, DIAG_LIVE(DIAG_LIVE_OCTAVE, oct));
    last_decay = decay; last_tone = tone; last_oct = oct;
#endif
}

#ifdef DIAG_PROFILE_LIVE
/* A live build leaves the panel, the octave and the flash prefetch alone, and
 * alternates only the reverb send handover (diag_handover_paired, see
 * looping_delay.c). Variant ids follow on from the cycled build's, so the
 * decoder can tell the two kinds of capture apart. */
static const uint8_t prof_live_paired[] = { 0, 1 };
#define PROF_LIVE_VARIANT0  8u
#undef  PROF_NVARIANTS
#define PROF_NVARIANTS  ((uint32_t)sizeof prof_live_paired)
#endif

static void prof_step(void)
{
    static uint32_t blk = 0;
    static uint8_t  cur = 0xFF;
    const uint32_t b = blk++;
    const uint8_t v = (uint8_t)((b / (PROF_BLOCKS_PER_S * PROF_VARIANT_SECONDS)) % PROF_NVARIANTS);
#ifdef DIAG_PROFILE_LIVE
    if (v != cur) {
        cur = v;
        diag_handover_paired = prof_live_paired[v];
    }
    const uint32_t mark = PROF_LIVE_VARIANT0 + v;
#else
    prof_override = 1;
    if (v != cur) {
        cur = v;
        pitch_semis_target = 12.0f * (float)prof_variants[v].octave;
        pitch_mix_target = prof_variants[v].octave ? 0.5f : 0.0f;
        if (prof_variants[v].prefetch) FLASH->ACR |= FLASH_ACR_PRFTEN;
        else                           FLASH->ACR &= ~FLASH_ACR_PRFTEN;
    }
    const uint32_t mark = v;
#endif
    if ((b % (PROF_BLOCKS_PER_S / 2u)) == 0u) {
        static uint32_t drops_reported = 0;
        const uint32_t drops = input_drop_count;
        uint32_t n = drops - drops_reported;
        drops_reported = drops;
        diag_log(DIAG_EVT_REVERB_DROP, DIAG_VARIANT_MARK | mark);
        if (n > DIAG_DROP_COUNT_MAX) n = DIAG_DROP_COUNT_MAX;
        if (n) diag_log(DIAG_EVT_REVERB_DROP, DIAG_DROP_COUNT | n);
        prof_report_cpu();
        prof_report_live();
    }

#ifndef DIAG_PROFILE_LIVE
    /* Decay 0.05..1 every 9 s (the full range: short decays included), Tone
     * 0.3..1 every 6.3 s. */
    float pd = (float)(b % (PROF_BLOCKS_PER_S * 9u)) / (float)(PROF_BLOCKS_PER_S * 9u);
    float pt = (float)(b % (PROF_BLOCKS_PER_S * 63u / 10u)) / (float)(PROF_BLOCKS_PER_S * 63u / 10u);
    float td = (pd < 0.5f) ? 2.0f * pd : 2.0f - 2.0f * pd;
    float tt = (pt < 0.5f) ? 2.0f * pt : 2.0f - 2.0f * pt;
    set_macro_value(MACRO_DECAY, 0.05f + 0.95f * td);
    set_macro_value(MACRO_SIZE,  0.05f + 0.95f * td);
    set_macro_value(MACRO_TONE,  0.30f + 0.70f * tt);
#endif
}
#endif

/* ==== Main-loop: drain one block ==== */
#ifndef REVERB_STAGE
#define REVERB_STAGE 4
#endif

void velvet_reverb_poll(void)
{
    /* Nothing queued? Unsigned compare is deliberate: it stays correct when the
     * sequence counters wrap. */
    if (in_wr_seq == in_rd_seq) return;
#if defined(DIAG_REVERB_PROFILE) && !defined(VELVET_REVERB_HOST)
    DiagCpuMark poll_mark;
    diag_cpu_enter(&poll_mark);
    if (in_wr_seq - in_rd_seq > prof_queue_max) prof_queue_max = in_wr_seq - in_rd_seq;
#endif
    input_ready = in_ring[in_rd_seq % REVERB_IN_NBUF];
    input_ready_r = in_ring_r[in_rd_seq % REVERB_IN_NBUF];
#if defined(DIAG_REVERB_PROFILE) && !defined(VELVET_REVERB_HOST)
    prof_step();
#endif

#ifndef VELVET_REVERB_HOST
#ifdef DIAG_REVERB_PROFILE
    /* Pick the one-in-N blocks that get fully instrumented this pass. Chosen
     * before any timing starts so every stage below agrees on whether it is
     * being sampled, which is what lets the per-stage numbers be summed and
     * reconciled against the block total. */
    /* Modulo, not a power-of-two mask, so the divisor can be chosen coprime with
     * the other periodic work in the block. With a mask the divisor must be a
     * power of two, and 128 shares all its factors with the 16-block macro
     * throttle in update_morph_state: the sampled blocks then land on the same
     * phase every time and the sample is not of the block population but of one
     * particular kind of block. That aliasing is what made reverb_morph look far
     * more expensive than it is. A prime divisor cannot phase-lock to anything. */
    static uint32_t prof_block_ctr = 0;
    diag_sample_block = ((++prof_block_ctr % DIAG_PROFILE_DIVISOR) == 0u);
#endif
    uint32_t i_blk_0; uint32_t t0 = diag_stage_start(&i_blk_0);
    /* Wall-clock lag from block-ready to poll pickup: how long the fresh input
     * block sat waiting on the main loop (includes any ISR + other main-loop
     * work that delayed the reverb). Large values => main loop not reaching
     * poll promptly, which lets the output reader run dry. */
    DIAG_LOG_BLK(DIAG_EVT_POLL_LATENCY, t0 - in_ready_cyc[in_rd_seq % REVERB_IN_NBUF]);
    /* Emit the cumulative bitcrush miss count ~once/sec (1500 blocks/s). */
    static uint16_t miss_emit_ctr = 0;
    if (++miss_emit_ctr >= 1500u) { miss_emit_ctr = 0; diag_log(DIAG_EVT_OUTPUT_MISS, output_miss_count); }
#endif

    /* Cheap per-block morph update (window/decay IIR + pre-delay time smooth).
     * The heavier effGains recompute is deferred to outside the block timer. */
#ifndef VELVET_REVERB_HOST
    uint32_t i_morph_0; uint32_t t_morph_0 = diag_stage_start(&i_morph_0);
#endif
    update_morph_state();
#ifndef VELVET_REVERB_HOST
    DIAG_LOG_BLK(DIAG_EVT_REVERB_MORPH, diag_stage_cycles(t_morph_0, i_morph_0));
#endif

#if REVERB_STAGE >= 1
    /* Pre-delay sustain engine runs on the input block, producing the
     * (feedforward) cascade input. Replaces the old per-stage/global recirc. */
#ifndef VELVET_REVERB_HOST
    uint32_t i_pre_0; uint32_t t_pre_0 = diag_stage_start(&i_pre_0);
#endif
    do_predelay();
#ifndef VELVET_REVERB_HOST
    DIAG_LOG_BLK(DIAG_EVT_REVERB_PREDELAY, diag_stage_cycles(t_pre_0, i_pre_0));
#endif
    do_input_write_t0();
#endif
#if REVERB_STAGE >= 2
    {
#ifndef VELVET_REVERB_HOST
        uint32_t i0_t0; uint32_t t0_t0 = diag_stage_start(&i0_t0);
#endif
        do_t0_phase();
#ifndef VELVET_REVERB_HOST
        DIAG_LOG_BLK(DIAG_EVT_REVERB_T0, diag_stage_cycles(t0_t0, i0_t0));
#endif
        do_bridge_t0_to_t1();
    }
    {
#ifndef VELVET_REVERB_HOST
        uint32_t i1_t0; uint32_t t1_t0 = diag_stage_start(&i1_t0);
#endif
        do_t1_phase();
#ifndef VELVET_REVERB_HOST
        DIAG_LOG_BLK(DIAG_EVT_REVERB_T1, diag_stage_cycles(t1_t0, i1_t0));
#endif
        do_bridge_t1_to_t2();
    }
#endif
#if REVERB_STAGE >= 3
    {
#ifndef VELVET_REVERB_HOST
        uint32_t i2_t0; uint32_t t2_t0 = diag_stage_start(&i2_t0);
#endif
        do_t2_phase();
#ifndef VELVET_REVERB_HOST
        DIAG_LOG_BLK(DIAG_EVT_REVERB_T2, diag_stage_cycles(t2_t0, i2_t0));
#endif
    }
#endif
#if REVERB_STAGE >= 4
#ifndef VELVET_REVERB_HOST
    uint32_t i_fin_0; uint32_t t_fin_0 = diag_stage_start(&i_fin_0);
#endif
    do_finalize();
#ifndef VELVET_REVERB_HOST
    DIAG_LOG_BLK(DIAG_EVT_REVERB_FINALIZE, diag_stage_cycles(t_fin_0, i_fin_0));
#endif
#endif

#ifndef VELVET_REVERB_HOST
    DIAG_LOG_BLK(DIAG_EVT_REVERB_BLOCK, DWT->CYCCNT - t0);
    /* ISR cycles that preempted this block (wall-clock block − this = pure reverb compute). */
    DIAG_LOG_BLK(DIAG_EVT_REVERB_ISR_IN_BLOCK, diag_isr_cycles - i_blk_0);
#endif

    /* Release the slot only now that its samples have been fully consumed, so
     * the ISR cannot refill it underneath us. */
    in_rd_seq = in_rd_seq + 1u;

    /* Background update — NOT inside the block timer but timed separately
     * so we can see how much CPU it eats between blocks. Whatever it costs
     * is unavailable to the next block; if it stretches past the input
     * buffer's lead time we miss a deadline even though the per-stage
     * numbers look fine. */
#ifndef VELVET_REVERB_HOST
    uint32_t i_bg_0; uint32_t t_bg_0 = diag_stage_start(&i_bg_0);
#endif
    background_eff_gains_update();
#ifndef VELVET_REVERB_HOST
    DIAG_LOG_BLK(DIAG_EVT_REVERB_BG_EFF, diag_stage_cycles(t_bg_0, i_bg_0));
#endif
#if defined(DIAG_REVERB_PROFILE) && !defined(VELVET_REVERB_HOST)
    if (diag_sample_block) prof_verify_dma();
    else { prof_t2_h = -1; prof_oct_ran = 0; }
#endif

    /* Start fetching the next block's T2 windows. Last, because it needs the
     * offsets the recompute above has just settled. */
    t2_gather_start();
#if defined(DIAG_REVERB_PROFILE) && !defined(VELVET_REVERB_HOST)
    diag_cpu_exit(&poll_mark, DIAG_CPU_REVERB, 0);
#endif
}

/* ==== Output ====
 * Each channel pulls one block per wrap from the output ring, holding a steady
 * ~1-block lag behind the newest published block so a single late poll doesn't
 * starve it. The two codec ISRs track the writer independently — no shared
 * read state, no order dependence between them.
 *
 * Lag policy (let newest = blocks_produced - 1 = newest published block, and
 * lag = newest - rd_seq = completed blocks buffered ahead of the reader):
 *   - advance the read sequence by one block per wrap;
 *   - if it passes newest (underrun: cushion fully drained), go back to
 *     TARGET_LAG behind and count an output_miss;
 *   - if lag grows beyond NBUF-1 (writer about to lap the reader), resync to
 *     the target lag, dropping stale audio.
 * Crucially the reader does NOT hug newest — it free-runs at one block per
 * wrap so the cushion floats around TARGET_LAG, absorbing poll jitter. The
 * reader thus reads slot (rd_seq % NBUF): always a completed block, never the
 * writer's in-progress slot. */
uint32_t velvet_reverb_blocks_done(void) { return blocks_produced; }
void velvet_reverb_overruns(uint32_t *drops, uint32_t *misses)
{
    *drops = input_drop_count;
    *misses = output_miss_count;
}

/* One reader's step to its next block. The reader waits until TARGET_LAG + 1
 * blocks exist and then starts TARGET_LAG behind the newest, so the cushion
 * the ring is for is really there: without that it started at the newest
 * block and stayed within one or two of it, so any poll more than a block
 * late replayed a block (heard as repeated fragments in sparse echoes). If
 * the cushion is ever used up, it goes back to TARGET_LAG behind rather than
 * hugging the newest. Until it starts it plays silence (slot 0, cleared). */
static inline uint8_t out_next_block(uint32_t *rd_seq, uint8_t *started, volatile uint32_t *misses)
{
    const uint32_t wr = blocks_produced;
    if (wr <= (uint32_t)REVERB_OUT_TARGET_LAG) return 0;
    const uint32_t newest = wr - 1u;
    if (!*started) {
        *started = 1;
        *rd_seq = newest - REVERB_OUT_TARGET_LAG;
    } else {
        (*rd_seq)++;
        const int32_t lag = (int32_t)(newest - *rd_seq);
        if (lag < 0) {
            *rd_seq = newest - REVERB_OUT_TARGET_LAG;
            (*misses)++;
        } else if (lag > (REVERB_OUT_NBUF - 1)) {
            *rd_seq = newest - REVERB_OUT_TARGET_LAG;
        }
    }
    return (uint8_t)(*rd_seq % REVERB_OUT_NBUF);
}

int16_t velvet_reverb_out_left(void)
{
    if (out_idx_L == 0) reader_buf_L = out_next_block(&rd_seq_L, &reader_started_L, &output_miss_count);
    int16_t s = outL[reader_buf_L][out_idx_L];
    out_idx_L = (uint8_t)((out_idx_L + 1) & (REVERB_OUT_BLOCK - 1));
    return s;
}

int16_t velvet_reverb_out_right(void)
{
    if (out_idx_R == 0) reader_buf_R = out_next_block(&rd_seq_R, &reader_started_R, &output_miss_count_r);
    int16_t s = outR[reader_buf_R][out_idx_R];
    out_idx_R = (uint8_t)((out_idx_R + 1) & (REVERB_OUT_BLOCK - 1));
    return s;
}

#ifdef VELVET_REVERB_HOST
/* ==== Host-only test hooks: tap relocation / uniformity probe ====
 * Set a stage's morphed window directly and recompute its effGains/effOff so
 * the host harness can inspect the tap relocation ladder. Settles the gain-comp
 * morph by recomputing a few
 * times so effGain magnitudes are stable. Density is fixed at MAX. */
int  host_t2_tap_count(void)         { return t2TapCount; }
uint32_t host_t2_effoff_l(int i)     { return effOffT2L[i]; }
uint32_t host_t2_effoff_r(int i)     { return effOffT2R[i]; }
float    host_t2_effwin(int i)       { return host_effwin_t2[i]; }
void host_t2_set_window_samples(float w)
{
    t2_window_morph = w;
    for (int s = 0; s < 32; s++) recompute_eff_gains_t2();
}
float host_t2_duration_max_samples(void) { return (float)T2_DURATION_MAX_SAMPLES; }
float host_t2_min_window_samples(void)   { return 0.2f * (float)REVERB_FS_HZ; }
#endif
