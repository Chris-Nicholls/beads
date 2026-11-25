/*
 * velvet_reverb.h — Velvet sparse-noise convolution reverb for DLD
 *
 * Architecture: three-stage cascade (T0 → T1 → T2) at REVERB_FS_HZ (24 kHz,
 * half the codec rate, or 48 kHz, the full codec rate), processed from the
 * main loop in blocks of 667 µs (16 samples at 24 kHz, 32 at 48 kHz).
 *
 *   T0: short window (≤ 80 ms), CCM ring, mono — early reflections
 *   T1: medium window (≤ 666 ms), CCM ring, mono — middle scattering
 *   T2: long window (≤ 8 s), SDRAM ring + DMA prefetch, stereo (via
 *       independent L/R read offsets with shared gain magnitude)
 *
 * Per-stage features:
 *   - Tap count fixed at MAX (density is not a control)
 *   - Window / duration (relocation ladder migrates taps as it shrinks)
 *   - Exponential decay envelope on T2 only (T0 taper baked in, T1 flat)
 *   - A pre-delay sustain engine (two modulated, damped feedback delay lines)
 *     sits in front of the cascade and supplies the long tail; the cascade
 *     itself runs purely feedforward. T0 also gets a per-tap Lexicon LFO.
 *
 * Output chain: stereo HPF + LPF biquads at the codec rate. At 24 kHz the LPF
 * also doubles as the reconstruction filter for the 2:1 linear-interp upsample.
 *
 * velvet_reverb_push_sample() is called once per CODEC sample from ch0 ISR.
 * velvet_reverb_poll() drains a ready block from the main loop.
 * velvet_reverb_out_left() / out_right() are called from the codec ISRs.
 */

#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Sample rates ----
 * The reverb runs at REVERB_FS_HZ; the codec at REVERB_CODEC_HZ. At 24 kHz the
 * input is decimated 2:1 and the output interpolated 2x; at 48 kHz neither.
 * Set it for the whole build (make REVERB_FS=48000), never per file:
 * looping_delay.c sizes the delay loops from REVERB_SDRAM_RESERVE, which
 * depends on it. */
#ifndef REVERB_FS_HZ
#define REVERB_FS_HZ   24000
#endif
#if REVERB_FS_HZ != 24000 && REVERB_FS_HZ != 48000
#error "REVERB_FS_HZ must be 24000 or 48000"
#endif
#define REVERB_CODEC_HZ   48000
/* Codec samples per reverb sample: 2 at 24 kHz, 1 at 48 kHz. */
#define REVERB_DECIM      (REVERB_CODEC_HZ / REVERB_FS_HZ)
/* Lengths below are specified in samples at 24 kHz and scaled by this, so every
 * time span (and the tap layout) is the same in seconds at either rate. */
#define REVERB_RATE_MULT  (REVERB_FS_HZ / 24000)

/* ---- SDRAM reverb region ----
 *
 * The T2 ring lives at the very top of SDRAM so it butts up against the end
 * of channel 1's loop region. looping_delay.c shrinks both delay channels
 * by REVERB_SDRAM_RESERVE bytes, which exactly equals the T2 ring's byte
 * count — so channel 1's write head stops at T2_RING_BASE. If T2_RING_BASE
 * sat anywhere lower, channel 1's last REVERB_SDRAM_RESERVE bytes would
 * overlap the T2 ring and the right delay would intermittently play back
 * T2 audio (the prior bug: a 24-kHz reverb tail played at 48 kHz reads as
 * an octave-up "ghost"). */
/* On the Beads: the ring is 131072 (5.46 s, 256 KiB of RAM_D1), and T2's
 * span runs to T2_DURATION_MAX_SAMPLES (5.33 s), using it all but the block
 * fetch's margin. */
#define T2_RING_SAMPLES  (131072UL * REVERB_RATE_MULT)
/* The reverb is stereo throughout: each channel has its own T2 ring, which
 * that side's taps read. */
#ifdef VELVET_REVERB_HOST
extern int16_t host_t2_ring_storage[T2_RING_SAMPLES];
extern int16_t host_t2_ring_r_storage[T2_RING_SAMPLES];
#define T2_RING_BASE     ((uintptr_t)host_t2_ring_storage)
#define T2_RING_R_BASE   ((uintptr_t)host_t2_ring_r_storage)
#elif defined(VELVET_REVERB_BEADS)
/* Beads: AXI SRAM (RAM_D1), both rings filling it, defined in beads.cc. */
extern int16_t beads_t2_ring_storage[T2_RING_SAMPLES];
extern int16_t beads_t2_ring_r_storage[T2_RING_SAMPLES];
#define T2_RING_BASE     ((uintptr_t)beads_t2_ring_storage)
#define T2_RING_R_BASE   ((uintptr_t)beads_t2_ring_r_storage)
#else
/* 0xD2000000 = SDRAM_BASE + SDRAM_SIZE (32 MiB SDRAM ends here).
 * T2 ring occupies [0xD2000000 - T2_RING_SAMPLES*2 .. 0xD2000000). */
#define T2_RING_BASE     (0xD2000000UL - (T2_RING_SAMPLES * 2UL))
#endif
#define T2_RING_MASK     (T2_RING_SAMPLES - 1)

/* The deepest T0 read is its longest window (T0_WINDOW_MAX_SAMPLES, 4800 at
 * full Size) plus the tap LFO's widest swing (2 x ~5 ms = 240 at full
 * modulation), ~5040 back, so the ring is the next power of two. */
#define T0_RING_SAMPLES  (8192UL * REVERB_RATE_MULT)
#define T0_RING_MASK     (T0_RING_SAMPLES - 1)

#define T1_RING_SAMPLES  (16384UL * REVERB_RATE_MULT)
#define T1_RING_MASK     (T1_RING_SAMPLES - 1)

/* ---- Pre-delay sustain-engine lines ----
 * Two modulated, damped feedback delay lines sit IN FRONT of the velvet
 * cascade (replacing the old per-stage/global recirculation). They store
 * float samples (normalised ±1, node-clamped ±4) so the high-feedback loop
 * doesn't accumulate int16 quantisation noise. On Beads they sit in RAM_D2
 * (2 × 64 KiB at 24 kHz). Line length is a power of two so the read/write
 * wraps are a cheap mask; it covers PRE_DELAY_MAX plus the modulation's
 * reach (up to 230 samples) and the pitch heads' crossfade room (a reverse
 * head at 2x, finishing its 25 ms crossfade, falls ~1900 behind a loop). */
#define PRE_DELAY_LINE_SAMPLES  (16384UL * REVERB_RATE_MULT)
#define PRE_DELAY_LINE_MASK     (PRE_DELAY_LINE_SAMPLES - 1)
#define PRE_DELAY_MAX_SAMPLES   (13800 * REVERB_RATE_MULT)  /* 0.575 s */

/* The input delay (the Beads panel's pre-delay): the reverb's input, int16 as
 * it arrives, held back by up to INPUT_DELAY_MAX_S before the sustain loop
 * and the cascade, so the whole reverb comes in later. A power-of-two ring
 * per channel; the pair fills RAM_D3 on the Beads, which is what sets the
 * maximum (16384 samples = 0.683 s, less a block). */
#define INPUT_DELAY_SAMPLES     (16384UL * REVERB_RATE_MULT)
#define INPUT_DELAY_MASK        (INPUT_DELAY_SAMPLES - 1)
#define INPUT_DELAY_MAX_S       0.68f
#ifdef VELVET_REVERB_HOST
extern int16_t host_input_delay_storage[INPUT_DELAY_SAMPLES];
extern int16_t host_input_delay_r_storage[INPUT_DELAY_SAMPLES];
#define INPUT_DELAY_BASE    ((uintptr_t)host_input_delay_storage)
#define INPUT_DELAY_R_BASE  ((uintptr_t)host_input_delay_r_storage)
#elif defined(VELVET_REVERB_BEADS)
extern int16_t beads_input_delay_storage[INPUT_DELAY_SAMPLES];
extern int16_t beads_input_delay_r_storage[INPUT_DELAY_SAMPLES];
#define INPUT_DELAY_BASE    ((uintptr_t)beads_input_delay_storage)
#define INPUT_DELAY_R_BASE  ((uintptr_t)beads_input_delay_r_storage)
#endif
#define PRE_MOD_MAX_SAMPLES     (720 * REVERB_RATE_MULT)    /* 0.030 s (depth = 1 swing) */

/* Two loops (A, the shorter, and B), each a pair of lines, left and right. */
#ifdef VELVET_REVERB_HOST
extern float host_predelay_a_storage[PRE_DELAY_LINE_SAMPLES];
extern float host_predelay_b_storage[PRE_DELAY_LINE_SAMPLES];
extern float host_predelay_a_r_storage[PRE_DELAY_LINE_SAMPLES];
extern float host_predelay_b_r_storage[PRE_DELAY_LINE_SAMPLES];
#define PRE_DELAY_A_BASE    ((uintptr_t)host_predelay_a_storage)
#define PRE_DELAY_B_BASE    ((uintptr_t)host_predelay_b_storage)
#define PRE_DELAY_A_R_BASE  ((uintptr_t)host_predelay_a_r_storage)
#define PRE_DELAY_B_R_BASE  ((uintptr_t)host_predelay_b_r_storage)
#elif defined(VELVET_REVERB_BEADS)
/* Beads: SRAM1/2 (RAM_D2), all four filling it, defined in beads.cc. */
extern float beads_predelay_a_storage[PRE_DELAY_LINE_SAMPLES];
extern float beads_predelay_b_storage[PRE_DELAY_LINE_SAMPLES];
extern float beads_predelay_a_r_storage[PRE_DELAY_LINE_SAMPLES];
extern float beads_predelay_b_r_storage[PRE_DELAY_LINE_SAMPLES];
#define PRE_DELAY_A_BASE    ((uintptr_t)beads_predelay_a_storage)
#define PRE_DELAY_B_BASE    ((uintptr_t)beads_predelay_b_storage)
#define PRE_DELAY_A_R_BASE  ((uintptr_t)beads_predelay_a_r_storage)
#define PRE_DELAY_B_R_BASE  ((uintptr_t)beads_predelay_b_r_storage)
#else
/* Two float lines directly below the T2 ring: A then B, contiguous. */
#define PRE_DELAY_A_BASE  (T2_RING_BASE - 2UL * PRE_DELAY_LINE_SAMPLES * 4UL)
#define PRE_DELAY_B_BASE  (T2_RING_BASE - 1UL * PRE_DELAY_LINE_SAMPLES * 4UL)
#endif

/* Reserve = T2 ring (int16) + both pre-delay lines (float), so channel 1's
 * write head stops at PRE_DELAY_A_BASE (below the whole reverb region). */
#define REVERB_SDRAM_RESERVE  (T2_RING_SAMPLES * 2UL + 2UL * PRE_DELAY_LINE_SAMPLES * 4UL)

/* ---- Block geometry ----
 * The block period is 667 µs (1500 blocks/s) at either rate, so every per-block
 * constant (morph smoothing, limiter release, recompute schedules) is
 * rate-independent. REVERB_OUT_BLOCK is the same block in codec samples. */
#define REVERB_BLOCK         (16 * REVERB_RATE_MULT)
#define REVERB_OUT_BLOCK     (REVERB_BLOCK * REVERB_DECIM)

/* Depth of the ISR-to-main-loop input queue, in blocks. (REVERB_IN_NBUF - 1)
 * blocks is how late the main loop may be before an input block is dropped; see
 * the rationale at in_ring in velvet_reverb.c. Here rather than in the .c so the
 * host harness can report the tolerance it is measuring. */
#ifndef REVERB_IN_NBUF
#define REVERB_IN_NBUF       16
#endif

/* ---- Tap limits (compile-time upper bounds; runtime target counts can go below) ---- */
/* Each stage holds its taps at the DLD's density (its 40/40/32 taps over the
 * windows at Size's midpoint: 69 ms, 536 ms, 3.2 s) across its longest span,
 * so Size changes how long they are spread over, not how dense they are. */
#ifndef MAX_T0_TAPS
#define MAX_T0_TAPS  116
#endif
#ifndef MAX_T1_TAPS
#define MAX_T1_TAPS  50
#endif
#ifndef MAX_T2_TAPS
#define MAX_T2_TAPS  130
#endif

/* ---- Maximum windows (samples at fs_reverb, also user-tunable upper bounds) ---- */
#define T0_WINDOW_MAX_SAMPLES   (4800 * REVERB_RATE_MULT)     /* 200 ms */
#define T1_WINDOW_MAX_SAMPLES   (16000 * REVERB_RATE_MULT)    /* ~666 ms */
/* 5.33 s, as far as the T2 ring reaches less 3072 samples for the block
 * fetch: the lower half of Size stretches T2 out to this. The 130 taps over
 * it are ~24 a second, ~2.5x the DLD's. */
#define T2_SPAN_MAX_S           5.3333333f
#define T2_DURATION_MAX_SAMPLES (128000 * REVERB_RATE_MULT)   /* 5.33 s */

#ifndef TAP_GAIN_HEADROOM
#define TAP_GAIN_HEADROOM  1.0f
#endif

/* ---- Tap ladders ----
 * The read positions compute_tap_states can walk a tap through as the window
 * changes. Each tap now keeps its own position at every Size (see
 * build_single_rung_ladders), so one rung is all it needs. */
#define MAX_LADDER_LEVELS  1

/* ---- Tap arrays (always generated at MAX count + MAX window; runtime
 *      morphing decides how many of them are active and how far the
 *      window extends.) ---- */
/* Density is fixed at MAX (all taps always active — there is no Density
 * control), so the per-tap removal-rank arrays are gone. Each stage always
 * generates MAX taps once at init; the relocation ladder still migrates them
 * one-at-a-time as the Decay-driven window shrinks. */
extern uint32_t t0TapOffsets[MAX_T0_TAPS];
extern int16_t  t0TapGains [MAX_T0_TAPS];   /* base gain incl. taper (no envelope, no comp) */

extern uint32_t t1TapOffsets[MAX_T1_TAPS];
extern int16_t  t1TapGains [MAX_T1_TAPS];

extern uint32_t t2TapOffsetsL[MAX_T2_TAPS];
extern uint32_t t2TapOffsetsR[MAX_T2_TAPS];
extern int16_t  t2TapGains   [MAX_T2_TAPS];

/* ---- Mutable runtime globals ----
 *
 * Targets are smoothed per-block with a one-pole IIR. Defaults match the
 * Phase-1 maxed-out state (full tap count, full window, near-linear decay,
 * recirc near clamp). Poke these from a debugger or hook them up to
 * hardware controls (Phase 3). */

/* Reverb send level (right MIX pot, equal-power LUT in params.c). Scales
 * the audio that's fed INTO the reverb via velvet_reverb_push_sample.
 * The reverb's stereo output is always mixed unscaled into the final
 * output — `reverb_send` only controls how hard the reverb is driven, not
 * how much of its output you hear. */
extern float reverb_send;

/* Dry-path gain applied to the delay/wet mix BEFORE the reverb tail is
 * summed in. Set in params.c so that the top 10% of the right MIX pot
 * fades this to 0, leaving 100% wet reverb at the knob's max. */
extern float reverb_dry_gain;

/* Output filter cutoffs (codec rate, applied post-upsample) */
extern float reverb_lpf_hz;
extern float reverb_hpf_hz;

/* Per-stage window/duration target (driven by the Size macro) */
extern float reverb_t0_window_ms_target;     /* 0..~200 */
extern float reverb_t1_window_ms_target;     /* 0..~666 */
extern float reverb_t2_duration_s_target;    /* 0..~8 */

/* ---- Pre-delay sustain engine (replaces recirculation) ----
 * Two feedback delay lines A/B with one shared loop gain, shared low/high
 * shelves (tone + damping), per-line modulation, ~12 Hz DC block, input
 * ducking, and a dry/wet crossfade into the cascade. */
extern float reverb_feedback;            /* shared loop gain (clamped 0.999) */
extern float reverb_predelay_a_s;        /* line A delay time, seconds */
extern float reverb_predelay_b_s;        /* line B delay time, seconds */
extern float reverb_delay_mix;           /* 0 = dry input, 1 = full pre-delay output */
extern float reverb_fb_low_shelf_hz;
extern float reverb_fb_low_shelf_db;
extern float reverb_fb_high_shelf_hz;
extern float reverb_fb_high_shelf_db;
extern float reverb_fb_mod_rate;         /* Hz */
extern float reverb_duck_amount;         /* 0..1 max feedback reduction */
extern float reverb_duck_release_loops;  /* loop traversals, not seconds */

/* T0 (early-reflection) per-tap Lexicon LFO — golden-angle decorrelated. */
extern float reverb_t0_tap_mod_rate;     /* Hz */

/* ---- API ---- */
void velvet_reverb_init(void);
void velvet_reverb_regenerate_taps(void);

void velvet_reverb_push_sample(int16_t input);            /* mono: both channels */
void velvet_reverb_push_stereo(int16_t left, int16_t right);
/* Output blocks completed so far (a count that only grows). */
uint32_t velvet_reverb_blocks_done(void);
/* Running counts of the reverb falling behind: input blocks dropped because
 * the main loop had not taken the last ones, and output blocks replayed
 * because none was ready. Both stay 0 while it keeps up. */
void velvet_reverb_overruns(uint32_t *drops, uint32_t *misses);
int16_t velvet_reverb_out_left(void);
int16_t velvet_reverb_out_right(void);

void velvet_reverb_poll(void);

/* Count of input blocks discarded because poll had not consumed the previous
 * one — i.e. the main loop was late by more than one block period. Each drop
 * splices REVERB_BLOCK samples out of the reverb's input and is audible as a
 * whoosh, so this is the objective measure of how much main-loop lateness the
 * engine tolerates. Exposed for the host harness. */
extern volatile uint32_t input_drop_count;
/* Peak input-queue occupancy in blocks, out of (REVERB_IN_NBUF - 1) usable. */
extern volatile uint8_t input_queue_max;

/* ---- Macro system ----
 *
 * Three macros — Tone, Density and Size — each take a 0..1 pot position. */
void velvet_reverb_apply_tone_macro   (float v);
/* Density: taps per stage, from a sparse few (each heard as a grain or echo)
 * at 0 to the full set at 1, the default. Loudness is held as taps drop. */
void velvet_reverb_apply_density_macro(float v);
/* Size: the space and its decay. Its lower half lengthens the taps, from a
 * small, tight space to the longest windows; its upper half raises the
 * sustain loop's feedback, the taps' decay slowing with it so the loop's
 * repeats stay hidden, up to full sustain at the top. */
void velvet_reverb_apply_size_macro   (float v);
/* The input delay, 0..1: 0 to INPUT_DELAY_MAX_S on a square law, so the
 * short times get most of the travel. */
void velvet_reverb_set_input_delay(float v);

/* Pitch shift on the reverb feed, taken off the sustain engine's lines so it
 * is shifted once (a fixed offset, not shimmer). set_pitch takes semitones,
 * -12..+12, and may move freely (a 1 V/oct input): it glides with a ~5 ms
 * time constant and never clicks. set_pitch_mix sets the blend, 0 = none,
 * 0.5 = equal, 1 = shifted only, on an equal-power curve; it ramps over up
 * to ~20 ms. set_octave is the DLD's switch: ±1 = an octave at the equal
 * blend, 0 = off. */
void velvet_reverb_set_pitch(float semitones);
void velvet_reverb_set_pitch_mix(float mix);
void velvet_reverb_set_octave(int8_t octave);
/* Freeze: the sustain loop holds what it has, taking no new input and not
 * decaying, while the cascade keeps reverberating it. Ramps over ~20 ms. */
void velvet_reverb_set_freeze(int on);
/* The reverb's modulation, 0..1 (see the .c), gliding; it starts at
 * MOD_AMOUNT_DEFAULT. */
#define MOD_AMOUNT_DEFAULT  0.5f
void velvet_reverb_set_modulation(float amount);
/* Reverse: nonzero reads the pre-delay loop backwards into the reverb (the
 * loop itself keeps running forwards). Fades in and out over 20 ms. */
void velvet_reverb_set_reverse(int on);
void velvet_reverb_recompute_macros   (void);

#ifdef __cplusplus
}
#endif
