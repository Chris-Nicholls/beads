// Reverb-only firmware for the Beads hardware: the DLD velvet reverb
// (beads/velvet/velvet_reverb.c) in place of the granular engine.
//
// The original granular main is kept as beads_granular.cc.orig.
//
// The reverb is stereo: each input feeds its own side, with a little
// cross-feed so an off-centre sound fills both sides as it decays.
//
// Panel:
//   TIME knob           Pre-delay: the whole reverb comes in up to 0.68 s after
//                       the dry signal (square law, short times get the
//                       travel)
//   SIZE knob           The space and its decay together, as the DLD's Decay
//                       knob: how long the reverb's taps are spread over, how
//                       long they take to fade, and the sustain loop's
//                       feedback; the DLD's default a little past noon
//   SHAPE knob          Tone macro
//   TIME, SIZE, SHAPE and PITCH each have an attenuverter and a CV input, which
//   work as on Beads (RandomizedParameter): with the CV jack patched, clockwise
//   scales the CV and counter-clockwise lets the CV set an amount of random
//   modulation; with it unpatched the knob is an attenurandomizer, a random
//   walk whose depth grows either side of noon, fast clockwise and slow
//   counter-clockwise; PITCH's, clockwise, makes octave jumps instead (see
//   OctaveJumps). Patching is sensed by Beads' normalization probe.
//   DENSITY knob        Density macro (+ DENSITY CV): taps per reverb stage,
//                       from a sparse few to the full set (the DLD's sound)
//                       fully clockwise. Over the bottom quarter the sustain
//                       loop's echoes thin to a single train that repeats
//                       only itself.
//   DRY/WET knob        Reverb send (equal power; the wet output is at the
//                       DLD's level, REVERB_OUT_GAIN). The top 10% fades the dry
//                       signal out, so the knob's end is 100% reverb.
//   PITCH knob          Pitch shift of the reverb feed, on Beads' own detented
//                       interval law (0, 1, 3, 4, 7, 12 semitones either way),
//                       + the PITCH CV input, 1 V/oct with its attenuverter
//                       fully clockwise, together held to ±12
//   REVERB knob         Level of the pitch-shifted feed: 0 none, noon an equal
//                       blend, fully clockwise only the shifted feed
//   FREEZE button/gate  Freezes the sustain loop: no new input, no decay.
//                       New input skips the loop and goes straight to the
//                       reverb's taps (after TIME's pre-delay). The button
//                       latches; the gate freezes while high. The button
//                       lights while frozen.
//   SEED button/gate    Reverse: the sustain loop is read backwards into the
//                       reverb. The button latches; the gate reverses while
//                       high. The button lights while reversed.
//   QUALITY button      Picks what the FEEDBACK knob sets: the reverb's
//                       modulation, then the wet-signal effects sample-rate
//                       reduction, tape saturation and flutter, in turn.
//                       Holding it returns modulation to noon and the
//                       effects to zero.
//   FEEDBACK knob       The selection's amount, once it is moved (a newly
//                       selected one keeps its amount until then). Modulation
//                       sets how far the reverb's delays move: a little at
//                       zero, about the DLD's chorus at noon (where it
//                       starts), deep at the end. The effects
//                       run on the wet signal only, after the reverb's final
//                       taps and before its tone filters (wet_fx.c).
//   Quality LED         The selection: red = modulation, green = sample-rate
//                       reduction, amber = saturation, blue = flutter; brighter
//                       as its amount rises
//   VU LED              Input level
//
// Audio runs at 48 kHz in 12-frame blocks (4 kHz block rate). The reverb runs
// at 24 kHz, in stereo: its input is decimated in velvet_reverb_push_stereo() and its
// 16-sample blocks are processed in the main loop, as on the DLD.
//
// Based on Beads by Emilie Gillet (MIT) and the DLD velvet reverb (MIT).

#include <algorithm>
#include <cmath>
#include <cstring>

#include "beads/cv_reader.h"
#include "beads/drivers/codec.h"
#include "beads/drivers/freeze_input.h"
#include "beads/drivers/leds.h"
#include "beads/drivers/seed_input.h"
#include "beads/drivers/switches.h"
#include "beads/drivers/system.h"
#include "beads/drivers/watchdog.h"
#include "beads/dsp/randomized_parameter.h"
#include "beads/normalization_detector.h"
#include "beads/settings.h"
#include "beads/velvet/velvet_reverb.h"
#include "beads/velvet/wet_fx.h"
#include "stmlib/stmlib.h"

using namespace beads;
using namespace stmlib;

const int kSampleRate = 48000;
const int kBlockRate = 4000;

// The reverb's big buffers, a left and a right of each. These sections are
// NOLOAD (never cleared at start-up); velvet_reverb_init() zeroes them.
//   RAM_D1 (512 KiB): the two T2 rings, 256 KiB each at 24 kHz.
//   RAM_D2 (256 KiB): the four float sustain-loop lines, 64 KiB each.
//   RAM_D3 (64 KiB): the two int16 input delay rings (TIME), 32 KiB each.
// The left T0/T1 rings and all other reverb state are ordinary .bss (DTCM);
// the right T0/T1 rings are in ITCM (see velvet_reverb.c).
int16_t beads_t2_ring_storage[T2_RING_SAMPLES] __attribute__ ((section (".ramd1")));
int16_t beads_t2_ring_r_storage[T2_RING_SAMPLES] __attribute__ ((section (".ramd1")));
float beads_predelay_a_storage[PRE_DELAY_LINE_SAMPLES] __attribute__ ((section (".ramd2")));
float beads_predelay_b_storage[PRE_DELAY_LINE_SAMPLES] __attribute__ ((section (".ramd2")));
float beads_predelay_a_r_storage[PRE_DELAY_LINE_SAMPLES] __attribute__ ((section (".ramd2")));
float beads_predelay_b_r_storage[PRE_DELAY_LINE_SAMPLES] __attribute__ ((section (".ramd2")));
int16_t beads_input_delay_storage[INPUT_DELAY_SAMPLES] __attribute__ ((section (".ramd3")));
int16_t beads_input_delay_r_storage[INPUT_DELAY_SAMPLES] __attribute__ ((section (".ramd3")));

// CPU readout, refreshed once a second, for reading over SWD without
// stopping the module (st-flash --hot-plug read; see the address of
// beads_cpu in build/beads/beads.map). Shares of the CPU in tenths of a
// percent: the reverb's processing in the main loop, the audio interrupt,
// their sum, and the busiest single reverb block as a share of the block's
// time (16 samples at 24 kHz, 667 us). Then the reverb's running counts of
// falling behind (input blocks dropped, output blocks replayed): any
// nonzero count is audible.
struct BeadsCpu {
  uint32_t magic;           // 0xC0FFEE01 once the first second is in
  uint32_t reverb_permille;
  uint32_t audio_permille;
  uint32_t total_permille;
  uint32_t peak_block_permille;
  uint32_t seconds;
  uint32_t input_drops;
  uint32_t output_misses;
};
volatile BeadsCpu beads_cpu __attribute__ ((used)) = { 0, 0, 0, 0, 0, 0, 0, 0 };
volatile uint32_t audio_cycles = 0;   // audio interrupt cycles, running total

Codec codec;
FreezeInput freeze_input;
SeedInput seed_input;
NormalizationDetector normalization_detector;
CvReader cv_reader;
Leds leds;
Settings settings;
Switches switches;

volatile bool ready = false;

// Written by the audio callback, read by the main loop.
volatile float predelay_target = 0.0f;
volatile float size_target = 0.305f;   // the DLD's default, through kSizeLaw
// SIZE's knob law: the macro is the knob position (with its CV) to this
// power, so the top end gets more of the travel: what was noon now sits at
// 9 o'clock (0.2^kSizeLaw = 0.5), and the DLD's default (macro 0.6) at 0.3.
const float kSizeLaw = 0.4307f;
volatile float tone_target = 0.5f;
volatile float density_target = 1.0f;
volatile float pitch_target = 0.0f;
volatile float pitch_mix_target = 0.0f;
volatile bool freeze_latched = false;
volatile bool frozen = false;
volatile bool reverse_latched = false;
volatile bool reversed = false;
volatile float input_peak = 0.0f;

// QUALITY selections: the reverb's modulation, then the wet effects (slot
// 1 + their wet_fx index). The QUALITY button (SysTick) picks the selection
// and bumps fx_select_seq on every pick or reset; Process re-arms the
// FEEDBACK knob's takeover when it sees the sequence change.
const uint32_t kFxHoldMs = 600;
const int kFxModulation = 0;
const int kFxSlots = 1 + WET_FX_COUNT;
volatile uint8_t fx_selected = kFxModulation;
volatile float fx_amount[kFxSlots] = { MOD_AMOUNT_DEFAULT, 0.0f, 0.0f, 0.0f };
volatile uint32_t fx_select_seq = 0;

// Written by the main loop, read by the audio callback.
volatile float send_gain = 0.7f;
volatile float dry_gain = 1.0f;

extern "C" {

void SysTick_Handler() {
  HAL_IncTick();
  Watchdog::Reload();
  if (!ready) {
    return;
  }

  switches.Debounce();
  if (switches.just_pressed(SWITCH_FREEZE)) {
    freeze_latched = !freeze_latched;
  }
  if (switches.just_pressed(SWITCH_SEED)) {
    reverse_latched = !reverse_latched;
  }
  leds.set(LED_FREEZE, frozen ? 255 : 0);
  leds.set(LED_SEED, reversed ? 255 : 0);

  // QUALITY: a short press steps the selection, a hold turns every effect off
  // (and the release that ends it does not also step).
  static uint32_t media_held_ms = 0;
  static bool media_hold_fired = false;
  if (switches.pressed(SWITCH_MEDIA)) {
    if (++media_held_ms == kFxHoldMs) {
      for (int i = 0; i < kFxSlots; ++i) {
        fx_amount[i] = (i == kFxModulation) ? MOD_AMOUNT_DEFAULT : 0.0f;
      }
      ++fx_select_seq;
      media_hold_fired = true;
    }
  } else {
    if (media_held_ms && !media_hold_fired) {
      fx_selected = (fx_selected + 1) % kFxSlots;
      ++fx_select_seq;
    }
    media_held_ms = 0;
    media_hold_fired = false;
  }

  // The selection on the quality LED, brighter with its amount.
  static const uint8_t kFxColor[kFxSlots][3] = {
    { 255, 0, 0 },     // modulation
    { 0, 160, 0 },     // sample-rate reduction
    { 255, 110, 0 },   // saturation
    { 0, 40, 255 },    // flutter
  };
  const uint8_t sel = fx_selected;
  const float level = 0.12f + 0.88f * fx_amount[sel];
  Leds::Color c;
  c.r = uint8_t(kFxColor[sel][0] * level);
  c.g = uint8_t(kFxColor[sel][1] * level);
  c.b = uint8_t(kFxColor[sel][2] * level);
  leds.set_media(c);

  // VU meter: peak hold with a ~150 ms release at 1 kHz.
  float peak = input_peak;
  input_peak = peak * 0.993f;
  leds.set(LED_VU_METER, uint8_t(std::min(peak * 4.0f, 1.0f) * 255.0f));
  leds.Write();
}

}

// The output's soft knee: unchanged up to kOutputKnee of full scale, then
// bending smoothly towards full scale (slope 1 at the knee, so no corner).
// The reverb limits its own output, but the dry signal is added after it, and
// a hot input takes the sum over full scale; rounded off here, the overs are
// a soft saturation rather than hard clips at the DAC, each one a click.
const float kOutputKnee = 0.8f;

static inline float OutputKnee(float x) {
  const float a = fabsf(x);
  if (a <= kOutputKnee) {
    return x;
  }
  const float k = 1.0f - kOutputKnee;
  const float e = a - kOutputKnee;
  const float y = kOutputKnee + k * e / (k + e);
  return x < 0.0f ? -y : y;
}

static inline int16_t SaturateInt16(float x) {
  x = std::max(std::min(x, 32767.0f), -32768.0f);
  return static_cast<int16_t>(x);
}

// The random walks behind the attenurandomizers, one per parameter, at the
// rate Beads' own (its pitch modulation) uses.
RandomWalkModulation time_walk, size_walk, shape_walk, pitch_walk;
const float kWalkFrequency = 0.2f / float(kSampleRate);

// PITCH's attenurandomizer clockwise of noon (jack unpatched): instead of a
// drift, the pitch jumps an octave up or down and comes back. The amount sets
// how often (a mean gap from kJumpGapMaxS down to kJumpGapMinS, random
// between jumps) and for how long (kJumpHoldMinS up to kJumpHoldMaxS, give or
// take a third). A jump that would leave the reverb's +/-12 semitones goes
// the other way.
const float kJumpGapMaxS = 16.0f, kJumpGapMinS = 1.0f;
const float kJumpHoldMinS = 0.12f, kJumpHoldMaxS = 1.5f;

class OctaveJumps {
 public:
  void Init() { wait_ = hold_ = 0; offset_ = 0.0f; }

  // Advances n codec frames; returns the jump offset in semitones.
  float Process(float amount, float base, size_t n) {
    if (amount <= 0.0f) { Init(); return 0.0f; }
    if (hold_ > 0) {
      hold_ -= static_cast<int32_t>(n);
      if (hold_ <= 0) { offset_ = 0.0f; wait_ = DrawGap(amount); }
      return offset_;
    }
    if (wait_ <= 0) wait_ = DrawGap(amount);   // just engaged
    wait_ -= static_cast<int32_t>(n);
    if (wait_ <= 0) {
      offset_ = stmlib::Random::GetFloat() < 0.5f ? 12.0f : -12.0f;
      if (base + offset_ > 12.0f || base + offset_ < -12.0f) offset_ = -offset_;
      const float hold = kJumpHoldMinS * powf(kJumpHoldMaxS / kJumpHoldMinS, amount) *
          (0.67f + 0.66f * stmlib::Random::GetFloat());
      hold_ = static_cast<int32_t>(hold * float(kSampleRate));
    }
    return offset_;
  }

 private:
  static int32_t DrawGap(float amount) {
    const float mean = kJumpGapMaxS * powf(kJumpGapMinS / kJumpGapMaxS, amount);
    const float u = 1.0f - stmlib::Random::GetFloat();   // (0, 1]
    return static_cast<int32_t>(-logf(u) * mean * float(kSampleRate));
  }
  int32_t wait_, hold_;
  float offset_;
};
OctaveJumps octave_jumps;

// The normalization probe is audible on unpatched audio inputs, and the
// detector needs a few decisions to settle, so the output stays muted for a
// moment at start-up, as on Beads (codec_transition_frames).
int startup_mute_blocks = kBlockRate / 4;

void Process(
    const Codec::Event* events,
    const Codec::Frame* in,
    Codec::Frame* out,
    size_t size) {
  const uint32_t process_start = DWT->CYCCNT;
  normalization_detector.Process(
      events[size - 1], in[size - 1], cv_reader.cv_values());
  cv_reader.Read();
  const NormalizationDetector& nd = normalization_detector;

  bool freeze_gate = false;
  for (size_t i = 0; i < size; ++i) {
    freeze_gate |= (events[i] & Codec::EVENT_FREEZE) != 0;
  }
  frozen = freeze_latched || freeze_gate;

  // The SEED jack carries the probe when unpatched, so its gate counts only
  // when something is plugged in.
  bool reverse_gate = false;
  if (!nd.normalized(NORMALIZATION_DETECTOR_CHANNEL_SEED)) {
    for (size_t i = 0; i < size; ++i) {
      reverse_gate |= (events[i] & Codec::EVENT_SEED) != 0;
    }
  }
  reversed = reverse_latched || reverse_gate;

  RandomizedParameter p;
  p.base = cv_reader.pot(POTS_ADC_CHANNEL_TIME);
  p.attenuverter = cv_reader.pot(POTS_ADC_CHANNEL_TIME_AMOUNT);
  p.has_cv = !nd.normalized(NORMALIZATION_DETECTOR_CHANNEL_TIME_CV);
  p.cv = cv_reader.cv(CV_ADC_CHANNEL_TIME);
  predelay_target = p.value(&time_walk, size, 1.0f, true);

  p.base = cv_reader.pot(POTS_ADC_CHANNEL_SIZE);
  p.attenuverter = cv_reader.pot(POTS_ADC_CHANNEL_SIZE_AMOUNT);
  p.has_cv = !nd.normalized(NORMALIZATION_DETECTOR_CHANNEL_SIZE_CV);
  p.cv = cv_reader.cv(CV_ADC_CHANNEL_SIZE);
  size_target = p.value(&size_walk, size, 1.0f, true);

  p.base = cv_reader.pot(POTS_ADC_CHANNEL_SHAPE);
  p.attenuverter = cv_reader.pot(POTS_ADC_CHANNEL_SHAPE_AMOUNT);
  p.has_cv = !nd.normalized(NORMALIZATION_DETECTOR_CHANNEL_SHAPE_CV);
  p.cv = cv_reader.cv(CV_ADC_CHANNEL_SHAPE);
  tone_target = p.value(&shape_walk, size, 1.0f, true);

  // PITCH pot is already in semitones (LAW_DETUNE) and the CV reads in
  // semitones, so the attenuverter fully clockwise is 1 V/oct. The random
  // walk spans the reverb's ±12 (Beads' own spans its ±24); the reverb
  // clamps the sum to ±12.
  p.base = cv_reader.pot(POTS_ADC_CHANNEL_PITCH);
  p.attenuverter = cv_reader.pot(POTS_ADC_CHANNEL_PITCH_AMOUNT);
  p.has_cv = !nd.normalized(NORMALIZATION_DETECTOR_CHANNEL_PITCH_CV);
  p.cv = cv_reader.cv(CV_ADC_CHANNEL_PITCH);
  if (!p.has_cv && p.attenuverter > 0.0f) {
    pitch_target = p.base + octave_jumps.Process(p.attenuverter, p.base, size);
  } else {
    octave_jumps.Process(0.0f, p.base, size);
    pitch_target = p.value(&pitch_walk, size, 12.0f, false);
  }

  // As on Beads, an unpatched DENSITY input is ignored (it carries the probe).
  float density = cv_reader.pot(POTS_ADC_CHANNEL_DENSITY);
  if (!nd.normalized(NORMALIZATION_DETECTOR_CHANNEL_DENSITY_CV)) {
    density += cv_reader.cv(CV_ADC_CHANNEL_DENSITY);
  }
  CONSTRAIN(density, 0.0f, 1.0f);
  density_target = density;
  pitch_mix_target = cv_reader.pot(POTS_ADC_CHANNEL_REVERB);

  // FEEDBACK knob -> the selection's amount, once the knob has moved from
  // where it was when it was picked (or everything was reset).
  static uint32_t fx_seen_seq = 0xffffffff;
  static float fx_knob_ref = 0.0f;
  static bool fx_knob_engaged = false;
  const float fx_knob = cv_reader.pot(POTS_ADC_CHANNEL_FEEDBACK);
  if (fx_select_seq != fx_seen_seq) {
    fx_seen_seq = fx_select_seq;
    fx_knob_ref = fx_knob;
    fx_knob_engaged = false;
  }
  if (!fx_knob_engaged && fabsf(fx_knob - fx_knob_ref) > 0.02f) {
    fx_knob_engaged = true;
  }
  if (fx_knob_engaged) {
    fx_amount[fx_selected] = std::min(std::max(fx_knob, 0.0f), 1.0f);
  }
  velvet_reverb_set_modulation(fx_amount[kFxModulation]);
  for (int i = 0; i < WET_FX_COUNT; ++i) {
    wet_fx_set_amount(i, fx_amount[1 + i]);
  }

  // An unpatched audio input carries the probe: take the other side in its
  // place, or silence with neither patched, as Beads' GainControl does.
  const bool l_open = nd.normalized(NORMALIZATION_DETECTOR_CHANNEL_AUDIO_LEFT);
  const bool r_open = nd.normalized(NORMALIZATION_DETECTOR_CHANNEL_AUDIO_RIGHT);
  const bool mute = startup_mute_blocks > 0;
  if (mute) --startup_mute_blocks;

  const float send = send_gain;
  const float dry = dry_gain;
  float peak = input_peak;

  for (size_t i = 0; i < size; ++i) {
    // Codec frames are 24-bit audio in 32-bit slots; the reverb works in
    // int16 like the DLD, so feed it the top 16 bits of the mono sum.
    float l = static_cast<float>(in[i].l) * (1.0f / 2147483648.0f);
    float r = static_cast<float>(in[i].r) * (1.0f / 2147483648.0f);
    if (mute || (l_open && r_open)) {
      l = r = 0.0f;
    } else if (l_open) {
      l = r;
    } else if (r_open) {
      r = l;
    }
    peak = std::max(peak, std::max(fabsf(l), fabsf(r)));

    // Stereo in: each side feeds its own side of the reverb (a lone patched
    // input has already been copied to both).
    velvet_reverb_push_stereo(SaturateInt16(l * send * 32768.0f),
                              SaturateInt16(r * send * 32768.0f));
    float out_l = OutputKnee(l * dry + velvet_reverb_out_left() * (1.0f / 32768.0f));
    float out_r = OutputKnee(r * dry + velvet_reverb_out_right() * (1.0f / 32768.0f));
    CONSTRAIN(out_l, -1.0f, 0.999999f);
    CONSTRAIN(out_r, -1.0f, 0.999999f);
    out[i].l = static_cast<int32_t>(out_l * 2147483648.0f);
    out[i].r = static_cast<int32_t>(out_r * 2147483648.0f);
  }
  input_peak = peak;
  audio_cycles += DWT->CYCCNT - process_start;
}

// DRY/WET pot -> send and dry gain, as on the DLD's right MIX pot: the send
// follows an equal-power curve, and the last 10% of travel fades the dry path
// out so the end of the knob is 100% reverb.
void UpdateMix() {
  const float t = cv_reader.pot(POTS_ADC_CHANNEL_DRY_WET);
  send_gain = sinf(std::min(std::max(t, 0.0f), 1.0f) * 1.5707963f);
  const float fade_start = 0.9f;
  dry_gain = t <= fade_start ? 1.0f : std::max(0.0f, (1.0f - t) / (1.0f - fade_start));
}

void Init() {
  System sys;
  sys.Init(true);

  Watchdog::Start();
  codec.Init();
  leds.Init();
  switches.Init();
  freeze_input.Init();
  seed_input.Init();

  // Read the factory CV calibration. Nothing here ever writes settings back.
  settings.Init(false);
  cv_reader.Init(&settings);
  // Prescaler 2 at 48 kHz, as Beads sets it.
  normalization_detector.Init(2);
  time_walk.Init(kWalkFrequency);
  size_walk.Init(kWalkFrequency);
  shape_walk.Init(kWalkFrequency);
  pitch_walk.Init(kWalkFrequency);
  octave_jumps.Init();

  wet_fx_init(float(kSampleRate));
  velvet_reverb_init();

  // The cycle counter, for the CPU readout.
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->LAR = 0xC5ACCE55;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

  ready = true;
  codec.Start(kSampleRate, kSampleRate / kBlockRate, &Process);
}

// Times one pass of velvet_reverb_poll() and folds it into beads_cpu.
uint32_t cpu_window_start = 0, cpu_reverb_cycles = 0, cpu_audio_at_start = 0;
uint32_t cpu_peak_cycles = 0;

void PollAndMeasure() {
  uint32_t& window_start = cpu_window_start;
  uint32_t& reverb_cycles = cpu_reverb_cycles;
  uint32_t& audio_at_start = cpu_audio_at_start;
  uint32_t& peak_cycles = cpu_peak_cycles;
  const uint32_t blocks = velvet_reverb_blocks_done();
  const uint32_t audio0 = audio_cycles;
  const uint32_t t0 = DWT->CYCCNT;
  velvet_reverb_poll();
  const uint32_t elapsed = DWT->CYCCNT - t0;
  if (velvet_reverb_blocks_done() != blocks) {
    const uint32_t own = elapsed - (audio_cycles - audio0);
    reverb_cycles += own;
    if (own > peak_cycles) peak_cycles = own;
  }
  const uint32_t window = DWT->CYCCNT - window_start;
  if (window >= SystemCoreClock) {
    const uint32_t audio = audio_cycles - audio_at_start;
    const float per = 1000.0f / float(window);
    const float block_cycles = float(SystemCoreClock) * float(REVERB_BLOCK) / float(REVERB_FS_HZ);
    beads_cpu.reverb_permille = uint32_t(float(reverb_cycles) * per);
    beads_cpu.audio_permille = uint32_t(float(audio) * per);
    beads_cpu.total_permille = beads_cpu.reverb_permille + beads_cpu.audio_permille;
    beads_cpu.peak_block_permille = uint32_t(float(peak_cycles) * 1000.0f / block_cycles);
    beads_cpu.seconds = beads_cpu.seconds + 1;
    uint32_t drops, misses;
    velvet_reverb_overruns(&drops, &misses);
    beads_cpu.input_drops = drops;
    beads_cpu.output_misses = misses;
    beads_cpu.magic = 0xC0FFEE01u;
    window_start = DWT->CYCCNT;
    reverb_cycles = 0;
    audio_at_start = audio_cycles;
    peak_cycles = 0;
  }
}

int main(void) {
  Init();

  while (1) {
    // The macro setters only store a target (with a dead-band); the reverb
    // picks the change up inside poll, so calling them every pass is cheap.
    const float size_macro = powf(size_target, kSizeLaw);
    velvet_reverb_apply_decay_macro(size_macro);
    velvet_reverb_apply_size_macro(size_macro);
    velvet_reverb_set_input_delay(predelay_target);
    velvet_reverb_apply_tone_macro(tone_target);
    velvet_reverb_apply_density_macro(density_target);
    velvet_reverb_set_pitch(pitch_target);
    velvet_reverb_set_pitch_mix(pitch_mix_target);
    velvet_reverb_set_freeze(frozen);
    velvet_reverb_set_reverse(reversed);
    UpdateMix();
    PollAndMeasure();
  }
}
