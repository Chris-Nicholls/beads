/* wet_fx: see wet_fx.h. */
#include "wet_fx.h"

#include <math.h>
#include <stdint.h>

#define WFX_TWO_PI 6.28318531f

/* Flutter's delay line: a power of two, 42 ms at 48 kHz, well past the
 * deepest excursion below. */
#define FLUTTER_BUF   2048
#define FLUTTER_MASK  (FLUTTER_BUF - 1)
/* Excursions at full amount, in ms of delay, as a worn transport gives, none
 * of them repeating:
 *   wow      a slow swing whose rate (WOW_HZ_LO..HI) and depth (half to all
 *            of WOW_MS) are drawn afresh every cycle
 *   flutter  band-limited noise (FLUT_LO_HZ..FLUT_HI_HZ), the scrape and
 *            capstan wobble of a real tape path, not a single tone
 *   drift    a wander to a new random depth every DRIFT_S_LO..HI seconds
 * Peak pitch deviation stays around 40 cents at full amount. */
#define WOW_HZ_LO     0.35f
#define WOW_HZ_HI     0.9f
#define WOW_MS        6.0f
#define FLUT_LO_HZ    3.0f
#define FLUT_HI_HZ    14.0f
#define FLUTTER_MS    0.35f
#define FLUT_NORM     6.0f        /* brings the noise band to about the old sine's swing */
#define DRIFT_S_LO    1.5f
#define DRIFT_S_HI    5.0f
#define DRIFT_MS      1.0f
#define FLUTTER_MIN_DELAY  1.5f   /* samples: keeps the cubic read behind the write */

/* Saturation: drive up to SAT_DRIVE_DB into a soft clipper, with makeup set
 * so a -20 dBFS signal keeps its level; a small bias adds even harmonics, and
 * the tape's top end closes from 20 kHz to 5 kHz as the amount rises. */
#define SAT_DRIVE_DB  30.0f
#define SAT_BIAS      0.25f
#define SAT_REF       0.1f

/* Sample-rate reduction: new samples are taken at a rate falling
 * exponentially from the full rate to SRR_MIN_HZ, with no anti-aliasing
 * before (that is the sound), and the output glides linearly from each
 * sample to the next rather than holding it. The phase runs 0..0.5 per new
 * sample and keeps its leftover fraction; the rate is smoothed per sample
 * (SRR_SMOOTH, ~17 ms at the 48 kHz codec rate). Coming in from off, the output fades from
 * the dry wet signal over the first 5% of the amount. */
#define SRR_MIN_HZ    480.0f
#define SRR_SMOOTH    0.0012f

static float fs = 48000.0f;
static float target[WET_FX_COUNT];
static float amt[WET_FX_COUNT];

static float buf_l[FLUTTER_BUF], buf_r[FLUTTER_BUF];
static uint32_t wpos;
static float wow_ph, wow_hz, wow_amp, drift_ph, drift_rate, drift_from, drift_to, delay_prev;
static float flut_lo, flut_hi, flut_hi2;
static uint32_t rng = 22222u;

static float lp_l, lp_r;
static float srr_ph, srr_inc_s, prev_l, prev_r, cur_l, cur_r;

void wet_fx_init(float sample_rate)
{
    fs = sample_rate;
    for (int i = 0; i < WET_FX_COUNT; i++) target[i] = amt[i] = 0.0f;
    for (int i = 0; i < FLUTTER_BUF; i++) buf_l[i] = buf_r[i] = 0.0f;
    wpos = 0;
    wow_ph = drift_ph = 0.0f;
    wow_hz = 0.6f; wow_amp = 1.0f;
    drift_rate = 0.3f; drift_from = drift_to = 0.0f;
    flut_lo = flut_hi = flut_hi2 = 0.0f;
    delay_prev = FLUTTER_MIN_DELAY;
    lp_l = lp_r = 0.0f;
    srr_ph = 0.0f; srr_inc_s = 0.5f; prev_l = prev_r = cur_l = cur_r = 0.0f;
}

void wet_fx_set_amount(int fx, float amount)
{
    if (fx < 0 || fx >= WET_FX_COUNT) return;
    if (amount < 0.0f) amount = 0.0f; else if (amount > 1.0f) amount = 1.0f;
    target[fx] = amount;
}

static inline float soft_clip(float x)
{
    if (x > 3.0f) return 1.0f;
    if (x < -3.0f) return -1.0f;
    const float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);   /* tanh, Padé */
}

static inline float rand01(void)
{
    rng = rng * 1664525u + 1013904223u;
    return (float)(rng >> 8) * (1.0f / 16777216.0f);
}

/* Bipolar sine from a 0..1 phase. */
static inline float sine01(float ph) { return sinf(WFX_TWO_PI * ph); }

static inline float hermite(const float *b, float pos)
{
    const int32_t i = (int32_t)floorf(pos);
    const float t = pos - (float)i;
    const float xm1 = b[(uint32_t)(i - 1) & FLUTTER_MASK];
    const float x0  = b[(uint32_t)i & FLUTTER_MASK];
    const float x1  = b[(uint32_t)(i + 1) & FLUTTER_MASK];
    const float x2  = b[(uint32_t)(i + 2) & FLUTTER_MASK];
    const float c = 0.5f * (x1 - xm1);
    const float v = x0 - x1;
    const float w = c + v;
    const float a = w + v + 0.5f * (x2 - x0);
    const float bb = w + a;
    return ((a * t - bb) * t + c) * t + x0;
}

void wet_fx_process(float *l, float *r, int n)
{
    if (n <= 0) return;

    /* Amounts glide with a ~10 ms time constant and land exactly. */
    const float k = 1.0f - expf(-(float)n / (0.01f * fs));
    for (int i = 0; i < WET_FX_COUNT; i++) {
        amt[i] += k * (target[i] - amt[i]);
        if (fabsf(target[i] - amt[i]) < 1e-5f) amt[i] = target[i];
    }

    /* ---- Flutter ---- */
    const float af = amt[WET_FX_FLUTTER];
    const float ms = 0.001f * fs;
    const float dt = (float)n / fs;
    /* Wow: a new rate and depth each cycle, taken at the cycle's start where
     * the swing crosses its centre, so neither change makes a step. */
    wow_ph += wow_hz * dt;
    if (wow_ph >= 1.0f) {
        wow_ph -= 1.0f;
        wow_hz = WOW_HZ_LO + (WOW_HZ_HI - WOW_HZ_LO) * rand01();
        wow_amp = 0.5f + 0.5f * rand01();
    }
    /* Flutter: white noise, band-passed at the call rate, two poles at the
     * top so the delay's slope (the pitch) stays smooth. */
    const float nz = 2.0f * rand01() - 1.0f;
    const float chi = 1.0f - expf(-WFX_TWO_PI * FLUT_HI_HZ * dt);
    flut_hi += chi * (nz - flut_hi);
    flut_hi2 += chi * (flut_hi - flut_hi2);
    flut_lo += (1.0f - expf(-WFX_TWO_PI * FLUT_LO_HZ * dt)) * (flut_hi2 - flut_lo);
    float flut = (flut_hi2 - flut_lo) * FLUT_NORM;
    if (flut > 1.0f) flut = 1.0f; else if (flut < -1.0f) flut = -1.0f;
    /* Drift: smoothstep to a new depth over a random span. */
    drift_ph += drift_rate * dt;
    if (drift_ph >= 1.0f) {
        drift_ph -= 1.0f;
        drift_from = drift_to;
        drift_to = rand01();
        drift_rate = 1.0f / (DRIFT_S_LO + (DRIFT_S_HI - DRIFT_S_LO) * rand01());
    }
    const float ds = drift_ph * drift_ph * (3.0f - 2.0f * drift_ph);
    const float drift = drift_from + (drift_to - drift_from) * ds;
    const float delay_next = FLUTTER_MIN_DELAY + af * ms *
        (WOW_MS * 0.5f * (1.0f + wow_amp * sine01(wow_ph)) +
         FLUTTER_MS * 0.5f * (1.0f + flut) +
         DRIFT_MS * drift);
    /* The delay line always runs, so engaging finds it full; the output fades
     * from the dry wet signal to the delayed one over the first 5% of the
     * amount, so switching in the line's short base delay never clicks. */
    const float fl_mix = (af * 20.0f > 1.0f) ? 1.0f : af * 20.0f;
    const float dstep = (delay_next - delay_prev) / (float)n;

    /* ---- Saturation ---- */
    const float as = amt[WET_FX_SATURATION];
    float g = 1.0f, bias = 0.0f, makeup = 1.0f, lpc = 1.0f;
    if (as > 0.0f) {
        g = powf(10.0f, as * SAT_DRIVE_DB * 0.05f);
        bias = SAT_BIAS * as;
        const float sb = soft_clip(bias);
        makeup = SAT_REF / (soft_clip(SAT_REF * g + bias) - sb);
        const float fc = 20000.0f * powf(0.25f, as);
        lpc = 1.0f - expf(-WFX_TWO_PI * fc / fs);
        if (lpc > 1.0f) lpc = 1.0f;
    }
    const float sat_dc = soft_clip(bias);

    /* ---- Sample-rate reduction ---- */
    const float ar = amt[WET_FX_SAMPLE_RATE];
    const float srr_goal = 0.5f * ((ar > 0.0f) ? powf(SRR_MIN_HZ / fs, ar) : 1.0f);
    const float srr_mix = (ar * 20.0f > 1.0f) ? 1.0f : ar * 20.0f;


    float d = delay_prev;
    for (int i = 0; i < n; i++) {
        float xl = l[i], xr = r[i];

        buf_l[wpos & FLUTTER_MASK] = xl;
        buf_r[wpos & FLUTTER_MASK] = xr;
        if (af > 0.0f) {
            /* Masked index minus delay: negative positions wrap in hermite(). */
            const float pos = (float)(wpos & FLUTTER_MASK) - d;
            const float yl = hermite(buf_l, pos), yr = hermite(buf_r, pos);
            xl += fl_mix * (yl - xl);
            xr += fl_mix * (yr - xr);
        }
        wpos++;
        d += dstep;

        if (as > 0.0f) {
            xl = makeup * (soft_clip(g * xl + bias) - sat_dc);
            xr = makeup * (soft_clip(g * xr + bias) - sat_dc);
            lp_l += lpc * (xl - lp_l);
            lp_r += lpc * (xr - lp_r);
            xl = lp_l; xr = lp_r;
        } else {
            lp_l = xl; lp_r = xr;   /* tracks, so engaging starts settled */
        }

        if (ar > 0.0f) {
            srr_inc_s += SRR_SMOOTH * (srr_goal - srr_inc_s);
            srr_ph += srr_inc_s;
            if (srr_ph >= 0.5f) {
                srr_ph -= 0.5f;
                prev_l = cur_l; prev_r = cur_r;
                cur_l = xl; cur_r = xr;
            }
            const float k = 2.0f * srr_ph;
            const float yl = prev_l + k * (cur_l - prev_l), yr = prev_r + k * (cur_r - prev_r);
            xl += srr_mix * (yl - xl);
            xr += srr_mix * (yr - xr);
        } else {
            /* Off: track, so engaging starts from the signal as it is. */
            srr_ph = 0.0f; srr_inc_s = 0.5f;
            prev_l = cur_l = xl; prev_r = cur_r = xr;
        }

        l[i] = xl; r[i] = xr;
    }
    delay_prev = delay_next;
}
