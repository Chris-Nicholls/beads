/* wet_fx: three character effects on the reverb's wet signal, after its final
 * taps and before its tone filters (velvet_reverb.c runs them at the codec
 * rate, on a +-1 scale). Each has its own amount, 0
 * (off, an exact bypass) to 1, and they run in series:
 *
 *   flutter -> tape saturation -> sample-rate reduction
 *
 * Amounts glide (~10 ms), so a knob or a reset never clicks. */
#ifndef WET_FX_H
#define WET_FX_H

#ifdef __cplusplus
extern "C" {
#endif

enum {
    WET_FX_SAMPLE_RATE = 0,
    WET_FX_SATURATION,
    WET_FX_FLUTTER,
    WET_FX_COUNT
};

void wet_fx_init(float sample_rate);
void wet_fx_set_amount(int fx, float amount);
/* Processes n stereo samples in place. Parameters update once per call, so
 * call it with short blocks (the reverb's output block). */
void wet_fx_process(float *l, float *r, int n);

#ifdef __cplusplus
}
#endif

#endif
