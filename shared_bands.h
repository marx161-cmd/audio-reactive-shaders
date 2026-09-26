// Shared-memory contract between the audio analysis writer (currently
// live_tap_writer.py; eventually a native EasyEffects effect module) and
// the shader renderer (renderer.py). POSIX shm, name below, fixed layout,
// packed to guarantee byte-identical size regardless of compiler padding
// so the Python struct format and this header can never silently drift.
//
// Torn-read protocol: writer increments `generation` (odd = mid-write),
// writes the payload, increments `generation` again (even = stable).
// Reader: read generation, read payload, read generation again; retry if
// either read was odd or the two values differ.

#pragma once

#include <stdint.h>

#define SHADER_BANDS_SHM_NAME "/shader_bands"
// 2026-09-06 (yet later): 120 bands, 27.5Hz-28160Hz (was 128, 55Hz-16kHz).
// tonnetz_pins.mp's discrete grid is ROWS(10) x COLS(12) = 120 real,
// distinct cells -- this band count/range is chosen to match that grid
// exactly (~11.78 bands/octave, just under the 12 semitone columns, so
// collisions are rare instead of the old ~15.64-vs-12 oversubscription).
// Low edge 27.5Hz = A0 (lowest standard piano note) -- previously excluded
// sub-bass had NO visible representation anywhere in the shader (the
// "handled separately by named bands" claim was stale, those uniforms
// aren't read by tonnetz_pins at all). High edge 28160Hz keeps the grid's
// 10-octaves-both-ends-extended structure exact even though the top
// octave (16-32kHz) is mostly above human hearing and will read silent --
// a deliberate structural choice over an awkward non-round range.
#define SHADER_BANDS_N_BANDS 120

#pragma pack(push, 1)
typedef struct {
  double timestamp;    // CLOCK_MONOTONIC seconds at write time
  uint32_t generation; // torn-read guard, see above
  uint32_t n_bands;    // = SHADER_BANDS_N_BANDS, sanity check for readers

  float bands[SHADER_BANDS_N_BANDS]; // log-spaced band energies, 0..1, smoothed

  // Derived scalar features, same scheme as shader-musicvideo/features.py
  float rms;
  float sub;
  float bass;
  float lowmid;
  float mid;
  float highmid;
  float presence;
  float brilliance;
  float centroid;
  float flux;
  float onset;
} shader_bands_t;
#pragma pack(pop)
