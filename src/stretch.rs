//! Playback speed and pitch for local playback.
//!
//! Speed and pitch are independent by default. Speed changes how fast the
//! song plays without changing its key; pitch moves the key in semitones
//! without changing its speed. Locked together, they move as a record does
//! played faster or slower: 1.5× plays a fifth higher. Both apply to this
//! computer's output only, like the equalizer; a Connect device or Cast
//! speaker plays at normal speed.
//!
//! Two stages do the work, both pure Rust and run on the player's thread:
//!
//! 1. WSOLA (waveform-similarity overlap-add) changes the duration. Each
//!    40 ms Hann-windowed grain is taken from near where the speed says it
//!    should come from, nudged up to 12 ms either way to wherever it best
//!    continues the previous grain, so the waveform stays coherent.
//! 2. A windowed-sinc resampler with a variable ratio moves the pitch, and
//!    the first stage is asked for the matching extra stretch so the speed
//!    is unchanged. Its 128-tap Kaiser kernel passes everything up to about
//!    20 kHz and rejects what would alias by about 100 dB.
//!
//! Locked, the first stage has nothing to do and passes the audio through
//! sample for sample: the resampler alone plays the song faster or slower,
//! which leaves none of the time stretch's artefacts.
//!
//! At normal speed and pitch the processor is bypassed and the samples go
//! through untouched. Once engaged, it keeps running (at normal settings it
//! reproduces its input exactly) until the next seek or skip, so returning
//! to normal never drops the audio it was holding.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{LazyLock, Mutex, PoisonError};
use std::time::{Duration, Instant};

use librespot_playback::{NUM_CHANNELS, SAMPLE_RATE};

/// The slowest and fastest speeds, as factors of normal.
pub const TEMPO_RANGE: std::ops::RangeInclusive<f32> = 0.5..=2.0;
/// How far pitch goes either way, in semitones.
pub const PITCH_RANGE: f32 = 12.0;
/// Speed snaps to steps this size.
pub const TEMPO_STEP: f32 = 0.05;

const CHANNELS: usize = NUM_CHANNELS as usize;
/// One grain, 40 ms. Long enough for a bass note's period several times
/// over, short enough that a transient isn't heard twice.
const GRAIN: usize = SAMPLE_RATE as usize / 25;
/// Grains overlap by half, so a Hann window sums to one.
const HOP: usize = GRAIN / 2;
/// How far from its nominal place a grain may be taken, 12 ms.
const SEEK: usize = SAMPLE_RATE as usize * 12 / 1000;
/// The coarse search looks at every fourth frame, then refines.
const COARSE: usize = 4;
/// Input frames either side of each resampled frame.
const TAPS_HALF: usize = 64;
/// Kernel table entries per input frame; linear interpolation between them
/// stays more than 100 dB under the signal.
const KERNEL_STEPS: usize = 512;
/// Half the resampler's transition band, as a fraction of the Nyquist
/// frequency (1.1 kHz): the passband ends this far below the cutoff and the
/// stopband starts this far above it.
const TRANSITION: f64 = 0.05;
/// The Kaiser window's shape for about 100 dB of stopband rejection.
const KAISER_BETA: f64 = 10.0;
/// Tempo changes remembered to work out how far a song has moved.
const HISTORY: usize = 32;

/// The speed and pitch the player follows, shared by the controls, the
/// player's thread and the position shown under the song.
pub struct Speed {
    tempo: AtomicU32,
    semitones: AtomicU32,
    /// Pitch follows speed, as on a record player.
    locked: AtomicBool,
    /// When each recent tempo took effect, so a position observed before a
    /// change still extrapolates correctly after it.
    changes: Mutex<Timeline>,
}

struct Timeline {
    /// The tempo before the oldest remembered change.
    before: f64,
    changes: VecDeque<(Instant, f64)>,
}

impl Default for Speed {
    fn default() -> Self {
        Self {
            tempo: AtomicU32::new(1.0f32.to_bits()),
            semitones: AtomicU32::new(0.0f32.to_bits()),
            locked: AtomicBool::new(false),
            changes: Mutex::new(Timeline {
                before: 1.0,
                changes: VecDeque::new(),
            }),
        }
    }
}

static SPEED: LazyLock<Speed> = LazyLock::new(Speed::default);

/// The speed and pitch of local playback.
pub fn speed() -> &'static Speed {
    &SPEED
}

impl Speed {
    /// Speed as a factor of normal.
    pub fn tempo(&self) -> f32 {
        f32::from_bits(self.tempo.load(Ordering::Relaxed))
    }

    /// The pitch set on its own, in semitones from the original. While
    /// locked, [`Self::pitch`] is what plays and this waits for unlocking.
    pub fn semitones(&self) -> f32 {
        f32::from_bits(self.semitones.load(Ordering::Relaxed))
    }

    /// Whether pitch follows speed.
    pub fn locked(&self) -> bool {
        self.locked.load(Ordering::Relaxed)
    }

    pub fn set_locked(&self, locked: bool) {
        self.locked.store(locked, Ordering::Relaxed);
    }

    /// The pitch that plays, in semitones: the speed's own while locked.
    pub fn pitch(&self) -> f32 {
        if self.locked() {
            locked_semitones(self.tempo())
        } else {
            self.semitones()
        }
    }

    pub fn is_normal(&self) -> bool {
        is_normal(self.tempo(), self.semitones(), self.locked())
    }

    /// Sets both, kept within range and snapped to their steps.
    pub fn set(&self, tempo: f32, semitones: f32) {
        self.set_at(tempo, semitones, Instant::now());
    }

    pub fn reset(&self) {
        self.set(1.0, 0.0);
    }

    fn set_at(&self, tempo: f32, semitones: f32, at: Instant) {
        let tempo = snap_tempo(tempo);
        let semitones = snap_semitones(semitones);
        self.semitones.store(semitones.to_bits(), Ordering::Relaxed);
        let previous = f32::from_bits(self.tempo.swap(tempo.to_bits(), Ordering::Relaxed));
        if previous.to_bits() == tempo.to_bits() {
            return;
        }
        let mut timeline = self.changes.lock().unwrap_or_else(PoisonError::into_inner);
        timeline.changes.push_back((at, f64::from(tempo)));
        while timeline.changes.len() > HISTORY {
            if let Some((_, tempo)) = timeline.changes.pop_front() {
                timeline.before = tempo;
            }
        }
    }

    /// How much of the song played between `since` and now, which differs
    /// from the time that passed whenever the speed isn't normal.
    pub fn played_since(&self, since: Instant) -> Duration {
        self.played_between(since, Instant::now())
    }

    fn played_between(&self, since: Instant, now: Instant) -> Duration {
        let timeline = self.changes.lock().unwrap_or_else(PoisonError::into_inner);
        let mut rate = timeline.before;
        for (when, tempo) in &timeline.changes {
            if *when <= since {
                rate = *tempo;
            }
        }
        let mut from = since;
        let mut played = 0.0;
        for (when, tempo) in timeline.changes.iter().filter(|(when, _)| *when > since) {
            let until = (*when).min(now);
            played += until.saturating_duration_since(from).as_secs_f64() * rate;
            from = until;
            rate = *tempo;
        }
        played += now.saturating_duration_since(from).as_secs_f64() * rate;
        Duration::from_secs_f64(played)
    }
}

fn is_normal(tempo: f32, semitones: f32, locked: bool) -> bool {
    (tempo - 1.0).abs() < f32::EPSILON && (locked || semitones.abs() < f32::EPSILON)
}

/// How far a speed moves the pitch when the two are locked: 1.5× is about
/// seven semitones up, 0.5× an octave down.
pub fn locked_semitones(tempo: f32) -> f32 {
    12.0 * snap_tempo(tempo).log2()
}

pub fn snap_tempo(tempo: f32) -> f32 {
    if !tempo.is_finite() {
        return 1.0;
    }
    let snapped = (tempo / TEMPO_STEP).round() * TEMPO_STEP;
    // Two decimals exactly, so 1.0 compares equal to 1.0.
    ((snapped * 100.0).round() / 100.0).clamp(*TEMPO_RANGE.start(), *TEMPO_RANGE.end())
}

pub fn snap_semitones(semitones: f32) -> f32 {
    if !semitones.is_finite() {
        return 0.0;
    }
    semitones.round().clamp(-PITCH_RANGE, PITCH_RANGE)
}

/// Changes speed and pitch on the player's thread.
pub struct Processor {
    engaged: bool,
    stretch: Wsola,
    shift: Shift,
}

impl Default for Processor {
    fn default() -> Self {
        Self::new()
    }
}

impl Processor {
    pub fn new() -> Self {
        Self {
            engaged: false,
            stretch: Wsola::new(),
            shift: Shift::new(),
        }
    }

    /// Processes one packet at the shared speed and pitch.
    pub fn process(&mut self, samples: Vec<f64>) -> Vec<f64> {
        let speed = speed();
        self.process_with(samples, speed.tempo(), speed.semitones(), speed.locked())
    }

    /// Processes one packet at the given speed and pitch, or with pitch
    /// following speed when `locked`. The output may be shorter or longer
    /// than the input, and empty while it fills.
    pub fn process_with(
        &mut self,
        samples: Vec<f64>,
        tempo: f32,
        semitones: f32,
        locked: bool,
    ) -> Vec<f64> {
        if !self.engaged {
            if is_normal(tempo, semitones, locked) {
                return samples;
            }
            *self = Self::new();
            self.engaged = true;
        }
        let tempo = f64::from(snap_tempo(tempo));
        // Locked, the resampler alone changes the speed and the stretch
        // only passes the audio on. It keeps running, so locking or
        // unlocking mid-song never drops the audio it holds.
        let (stretch, ratio) = if locked {
            (1.0, tempo)
        } else {
            let ratio = 2f64.powf(f64::from(snap_semitones(semitones)) / 12.0);
            (tempo / ratio, ratio)
        };
        let stretched = self.stretch.process(&samples, stretch);
        self.shift.process(&stretched, ratio)
    }

    /// Forgets held audio after a seek or skip. Normal settings go back to
    /// the bypass.
    pub fn reset(&mut self) {
        *self = Self::new();
        self.engaged = !speed().is_normal();
    }

    #[cfg(test)]
    fn reset_with(&mut self, tempo: f32, semitones: f32) {
        *self = Self::new();
        self.engaged = !is_normal(tempo, semitones, false);
    }
}

/// Time-scale modification by waveform-similarity overlap-add.
struct Wsola {
    window: Vec<f64>,
    /// Interleaved input frames still in reach.
    input: Vec<f64>,
    /// The absolute frame `input` starts at.
    base: usize,
    /// Where the next grain would be taken without any nudge.
    nominal: f64,
    /// Where the previous grain was taken.
    previous: Option<usize>,
    /// The previous grain's second half, windowed, waiting to be added.
    tail: Vec<f64>,
}

impl Wsola {
    fn new() -> Self {
        // Periodic Hann: halves that overlap by `HOP` sum to exactly one.
        let window = (0..GRAIN)
            .map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / GRAIN as f64).cos())
            .collect();
        Self {
            window,
            input: Vec::new(),
            base: 0,
            nominal: 0.0,
            previous: None,
            tail: vec![0.0; HOP * CHANNELS],
        }
    }

    fn end(&self) -> usize {
        self.base + self.input.len() / CHANNELS
    }

    fn frame(&self, at: usize) -> &[f64] {
        let start = (at - self.base) * CHANNELS;
        &self.input[start..start + CHANNELS]
    }

    fn mono(&self, at: usize) -> f64 {
        self.frame(at).iter().sum::<f64>() / CHANNELS as f64
    }

    /// `factor` is input played per output: 2 halves the duration.
    fn process(&mut self, samples: &[f64], factor: f64) -> Vec<f64> {
        self.input.extend_from_slice(samples);
        let mut out = Vec::with_capacity((samples.len() as f64 / factor) as usize + HOP * CHANNELS);
        loop {
            let nominal = self.nominal.round() as usize;
            let reach = match self.previous {
                Some(previous) => (nominal + SEEK).max(previous + HOP) + GRAIN,
                None => nominal + GRAIN,
            };
            if reach > self.end() {
                break;
            }
            let at = match self.previous {
                // With nothing to stretch, the audio that followed the last
                // grain continues it perfectly: the halves sum back to the
                // input, sample for sample.
                Some(previous) if factor == 1.0 && (previous + HOP).abs_diff(nominal) <= SEEK => {
                    previous + HOP
                }
                Some(previous) => self.best_match(previous + HOP, nominal),
                None => {
                    // The first grain's first half plays as it is: the tail
                    // is set to what the window leaves out of it.
                    for frame in 0..HOP {
                        let gain = self.window[HOP + frame];
                        for channel in 0..CHANNELS {
                            self.tail[frame * CHANNELS + channel] =
                                self.frame(nominal + frame)[channel] * gain;
                        }
                    }
                    nominal
                }
            };
            for frame in 0..HOP {
                let gain = self.window[frame];
                for channel in 0..CHANNELS {
                    out.push(
                        self.tail[frame * CHANNELS + channel]
                            + self.frame(at + frame)[channel] * gain,
                    );
                }
            }
            for frame in 0..HOP {
                let gain = self.window[HOP + frame];
                for channel in 0..CHANNELS {
                    self.tail[frame * CHANNELS + channel] =
                        self.frame(at + HOP + frame)[channel] * gain;
                }
            }
            self.previous = Some(at);
            self.nominal += HOP as f64 * factor;
            let keep = (self.nominal.floor() as usize)
                .saturating_sub(SEEK)
                .min(at + HOP)
                .max(self.base);
            self.input.drain(..(keep - self.base) * CHANNELS);
            self.base = keep;
        }
        out
    }

    /// The grain start near `nominal` whose first half best continues the
    /// audio that followed the previous grain, from `natural`.
    fn best_match(&self, natural: usize, nominal: usize) -> usize {
        let low = nominal.saturating_sub(SEEK).max(self.base);
        let high = nominal + SEEK;
        let reference: Vec<f64> = (0..HOP).map(|i| self.mono(natural + i)).collect();
        let score = |at: usize, step: usize| {
            let mut product = 0.0;
            let mut energy = 0.0;
            for i in (0..HOP).step_by(step) {
                let sample = self.mono(at + i);
                product += reference[i] * sample;
                energy += sample * sample;
            }
            product / (energy + 1e-12).sqrt()
        };
        // The audio that followed the last grain is the perfect
        // continuation, so it starts as the best when within reach; the
        // nominal place otherwise. Either wins ties.
        let mut best = if (low..=high).contains(&natural) {
            natural
        } else {
            nominal.max(low)
        };
        let mut best_score = score(best, COARSE);
        let mut candidate = nominal % COARSE + low - low % COARSE;
        while candidate <= high {
            if candidate >= low {
                let candidate_score = score(candidate, COARSE);
                if candidate_score > best_score {
                    best = candidate;
                    best_score = candidate_score;
                }
            }
            candidate += COARSE;
        }
        let coarse = best;
        let mut best_score = score(best, 1);
        for candidate in
            coarse.saturating_sub(COARSE - 1).max(low)..=(coarse + COARSE - 1).min(high)
        {
            let candidate_score = score(candidate, 1);
            if candidate_score > best_score {
                best = candidate;
                best_score = candidate_score;
            }
        }
        best
    }
}

/// Variable-ratio windowed-sinc resampler: plays its input `ratio` times
/// faster, which raises the pitch by that factor.
struct Shift {
    /// Interleaved input frames still in reach.
    input: Vec<f64>,
    /// Where the next output frame sits in `input`, in frames.
    time: f64,
    /// The kernel at `KERNEL_STEPS` points per frame, for `cutoff`.
    kernel: Vec<f64>,
    cutoff: f64,
}

impl Shift {
    fn new() -> Self {
        Self {
            input: vec![0.0; (TAPS_HALF - 1) * CHANNELS],
            time: (TAPS_HALF - 1) as f64,
            kernel: Vec::new(),
            cutoff: 0.0,
        }
    }

    fn process(&mut self, samples: &[f64], ratio: f64) -> Vec<f64> {
        // Reading faster squeezes the spectrum up: everything above the
        // output's Nyquist frequency once squeezed is cut before it can fold
        // back. Reading slower, the images above the input's Nyquist are cut.
        // At exactly normal speed every tap but one falls on a zero of the
        // full-band kernel, so the samples pass through unchanged.
        let cutoff = if ratio == 1.0 {
            1.0
        } else {
            (1.0 / ratio).min(1.0) - TRANSITION
        };
        if (cutoff - self.cutoff).abs() > f64::EPSILON || self.kernel.is_empty() {
            self.kernel = kernel(cutoff);
            self.cutoff = cutoff;
        }
        self.input.extend_from_slice(samples);
        let frames = self.input.len() / CHANNELS;
        let mut out = Vec::with_capacity((samples.len() as f64 / ratio) as usize + CHANNELS * 2);
        while (self.time.floor() as usize) + TAPS_HALF < frames {
            let whole = self.time.floor() as usize;
            let fraction = self.time - whole as f64;
            let mut sums = [0.0; CHANNELS];
            for tap in 0..2 * TAPS_HALF {
                // From `1 - TAPS_HALF` frames before to `TAPS_HALF` after.
                let offset = tap as f64 + 1.0 - TAPS_HALF as f64;
                let gain = self.tap(offset - fraction);
                let frame = (whole + tap + 1 - TAPS_HALF) * CHANNELS;
                for (channel, sum) in sums.iter_mut().enumerate() {
                    *sum += self.input[frame + channel] * gain;
                }
            }
            out.extend_from_slice(&sums);
            self.time += ratio;
        }
        let used = (self.time.floor() as usize).saturating_sub(TAPS_HALF - 1);
        let used = used.min(frames);
        self.input.drain(..used * CHANNELS);
        self.time -= used as f64;
        out
    }

    /// The kernel at `x` frames from the output's position.
    fn tap(&self, x: f64) -> f64 {
        let position = (x + TAPS_HALF as f64) * KERNEL_STEPS as f64;
        if position <= 0.0 {
            return 0.0;
        }
        let index = position.floor() as usize;
        let Some(&next) = self.kernel.get(index + 1) else {
            return self.kernel.get(index).copied().unwrap_or(0.0);
        };
        let here = self.kernel[index];
        here + (next - here) * (position - index as f64)
    }
}

/// A Kaiser-windowed sinc low-pass at `cutoff` (1 is the input's Nyquist
/// frequency), sampled across the taps.
fn kernel(cutoff: f64) -> Vec<f64> {
    let span = TAPS_HALF as f64;
    (0..=2 * TAPS_HALF * KERNEL_STEPS)
        .map(|step| {
            let x = step as f64 / KERNEL_STEPS as f64 - span;
            let sinc = if x.abs() < 1e-12 {
                1.0
            } else {
                let arg = std::f64::consts::PI * cutoff * x;
                arg.sin() / arg
            };
            let edge = x / span;
            let window = bessel_i0(KAISER_BETA * (1.0 - edge * edge).max(0.0).sqrt())
                / bessel_i0(KAISER_BETA);
            cutoff * sinc * window
        })
        .collect()
}

/// The zeroth-order modified Bessel function of the first kind, which
/// shapes the Kaiser window, by its power series.
fn bessel_i0(x: f64) -> f64 {
    let half = x / 2.0;
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1..100 {
        term *= half / k as f64;
        let squared = term * term;
        sum += squared;
        if squared < sum * 1e-17 {
            break;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: f64 = SAMPLE_RATE as f64;

    fn tone(hz: f64, seconds: f64) -> Vec<f64> {
        let frames = (RATE * seconds) as usize;
        (0..frames)
            .flat_map(|i| {
                let sample = (std::f64::consts::TAU * hz * i as f64 / RATE).sin() * 0.5;
                [sample, sample]
            })
            .collect()
    }

    /// Feeds `input` in librespot-sized packets.
    fn run(processor: &mut Processor, input: &[f64], tempo: f32, semitones: f32) -> Vec<f64> {
        feed(processor, input, tempo, semitones, false)
    }

    /// The same with pitch following speed.
    fn run_locked(processor: &mut Processor, input: &[f64], tempo: f32) -> Vec<f64> {
        feed(processor, input, tempo, 0.0, true)
    }

    fn feed(
        processor: &mut Processor,
        input: &[f64],
        tempo: f32,
        semitones: f32,
        locked: bool,
    ) -> Vec<f64> {
        input
            .chunks(1024 * CHANNELS)
            .flat_map(|chunk| processor.process_with(chunk.to_vec(), tempo, semitones, locked))
            .collect()
    }

    /// The middle half of the left channel.
    fn middle(samples: &[f64]) -> Vec<f64> {
        let left: Vec<f64> = samples.iter().step_by(CHANNELS).copied().collect();
        left[left.len() / 4..left.len() * 3 / 4].to_vec()
    }

    fn rms(samples: &[f64]) -> f64 {
        (samples.iter().map(|s| s * s).sum::<f64>() / samples.len() as f64).sqrt()
    }

    /// How far everything else is below the tones at `hz`, in decibels
    /// (signal to noise and distortion). The tones are fitted by least
    /// squares in each 93 ms block of the middle of the left channel, so a
    /// stretch's slow drift in phase isn't counted, only what it adds.
    fn sinad_db(samples: &[f64], hz: &[f64]) -> f64 {
        const BLOCK: usize = 4096;
        let signal = middle(samples);
        let (mut residual, mut fitted) = (0.0, 0.0);
        for (index, block) in signal.as_chunks::<BLOCK>().0.iter().enumerate() {
            let (r, f) = fit_block(block, index * BLOCK, hz);
            residual += r;
            fitted += f;
        }
        10.0 * (fitted / residual.max(1e-300)).log10()
    }

    /// The energy a block leaves after fitting the tones, and theirs.
    fn fit_block(block: &[f64], offset: usize, hz: &[f64]) -> (f64, f64) {
        let size = hz.len() * 2 + 1;
        let basis = |i: usize| {
            let mut row: Vec<f64> = hz
                .iter()
                .flat_map(|hz| {
                    let phase = std::f64::consts::TAU * hz * (offset + i) as f64 / RATE;
                    [phase.sin(), phase.cos()]
                })
                .collect();
            row.push(1.0);
            row
        };
        // The normal equations, solved by Gaussian elimination.
        let mut gram = vec![vec![0.0; size]; size];
        let mut fit = vec![0.0; size];
        for (i, sample) in block.iter().enumerate() {
            let b = basis(i);
            for row in 0..size {
                fit[row] += b[row] * sample;
                for column in 0..size {
                    gram[row][column] += b[row] * b[column];
                }
            }
        }
        for pivot in 0..size {
            for row in pivot + 1..size {
                let factor = gram[row][pivot] / gram[pivot][pivot];
                let pivot_row = gram[pivot].clone();
                for (cell, above) in gram[row].iter_mut().zip(&pivot_row).skip(pivot) {
                    *cell -= factor * above;
                }
                fit[row] -= factor * fit[pivot];
            }
        }
        for row in (0..size).rev() {
            let known: f64 = (row + 1..size)
                .map(|column| gram[row][column] * fit[column])
                .sum();
            fit[row] = (fit[row] - known) / gram[row][row];
        }
        let mut residual = 0.0;
        let mut fitted = 0.0;
        for (i, sample) in block.iter().enumerate() {
            let b = basis(i);
            let tones: f64 = (0..size - 1).map(|k| fit[k] * b[k]).sum();
            residual += (sample - tones - fit[size - 1]).powi(2);
            fitted += tones * tones;
        }
        (residual, fitted)
    }

    /// The dominant frequency of the left channel, by counting rising zero
    /// crossings over the middle of the signal.
    fn frequency(samples: &[f64]) -> f64 {
        let left: Vec<f64> = samples.iter().step_by(CHANNELS).copied().collect();
        let middle = &left[left.len() / 4..left.len() * 3 / 4];
        let crossings = middle
            .windows(2)
            .filter(|pair| pair[0] < 0.0 && pair[1] >= 0.0)
            .count();
        crossings as f64 * RATE / middle.len() as f64
    }

    fn frames(samples: &[f64]) -> f64 {
        (samples.len() / CHANNELS) as f64
    }

    #[test]
    fn normal_speed_and_pitch_pass_samples_through() {
        let input = tone(440.0, 1.0);
        let mut processor = Processor::new();
        assert_eq!(run(&mut processor, &input, 1.0, 0.0), input);
    }

    #[test]
    fn speed_changes_duration_and_keeps_pitch() {
        let input = tone(440.0, 4.0);
        for tempo in [0.5, 0.75, 1.5, 2.0] {
            let output = run(&mut Processor::new(), &input, tempo, 0.0);
            let expected = frames(&input) / f64::from(tempo);
            let length = frames(&output);
            // Short only by the grain or two still held at the end, which
            // plays for longer the slower the speed.
            let held = (GRAIN + SEEK + HOP) as f64 / f64::from(tempo);
            assert!(
                (length - expected).abs() < held,
                "tempo {tempo}: {length} frames, expected about {expected}"
            );
            let hz = frequency(&output);
            assert!((hz - 440.0).abs() < 8.0, "tempo {tempo}: {hz} Hz");
        }
    }

    #[test]
    fn pitch_changes_frequency_and_keeps_duration() {
        let input = tone(440.0, 4.0);
        for semitones in [-12.0, -5.0, 7.0, 12.0] {
            let output = run(&mut Processor::new(), &input, 1.0, semitones);
            let length = frames(&output);
            assert!(
                (length - frames(&input)).abs() < RATE * 0.1,
                "{semitones} semitones: {length} frames"
            );
            let expected = 440.0 * 2f64.powf(f64::from(semitones) / 12.0);
            let hz = frequency(&output);
            assert!(
                (hz - expected).abs() < expected * 0.02,
                "{semitones} semitones: {hz} Hz, expected {expected}"
            );
        }
    }

    #[test]
    fn speed_and_pitch_combine() {
        let input = tone(330.0, 4.0);
        let output = run(&mut Processor::new(), &input, 1.25, 3.0);
        let expected = frames(&input) / 1.25;
        assert!((frames(&output) - expected).abs() < RATE * 0.1);
        let hz = frequency(&output);
        let pitched = 330.0 * 2f64.powf(0.25);
        assert!((hz - pitched).abs() < pitched * 0.02, "{hz} Hz");
    }

    #[test]
    fn the_level_holds_through_a_stretch() {
        let input = tone(440.0, 2.0);
        let output = run(&mut Processor::new(), &input, 0.8, 2.0);
        let middle = &output[output.len() / 4..output.len() * 3 / 4];
        let peak = middle.iter().fold(0.0f64, |peak, s| peak.max(s.abs()));
        assert!((0.4..0.6).contains(&peak), "peak {peak}");
    }

    #[test]
    fn going_back_to_normal_keeps_every_sample_in_order() {
        // Engaged, then normal: the processor keeps running and plays
        // exactly what it is given, after its own short delay.
        let input = tone(220.0, 1.0);
        let mut processor = Processor::new();
        let mut output = run(&mut processor, &input[..input.len() / 2], 1.0, 0.0);
        assert_eq!(output.len(), input.len() / 2);
        processor.reset_with(1.0, 2.0);
        output.extend(run(&mut processor, &input[input.len() / 2..], 1.0, 0.0));
        let produced = &output[input.len() / 2..];
        let held = input.len() / 2 - produced.len();
        assert!(held < (GRAIN + TAPS_HALF) * CHANNELS * 2, "held {held}");
        for (out, expected) in produced.iter().zip(&input[input.len() / 2..]) {
            assert!((out - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn locked_pitch_follows_speed() {
        let input = tone(440.0, 4.0);
        for tempo in [0.5, 0.8, 1.25, 1.5, 2.0] {
            let output = run_locked(&mut Processor::new(), &input, tempo);
            let expected = frames(&input) / f64::from(tempo);
            let length = frames(&output);
            // Only what the stages still hold is missing, and it would
            // play for longer the slower the speed.
            let held = (GRAIN + SEEK + TAPS_HALF) as f64 / f64::from(tempo);
            assert!(
                (length - expected).abs() < held,
                "tempo {tempo}: {length} frames, expected about {expected}"
            );
            let pitched = 440.0 * f64::from(tempo);
            let hz = frequency(&output);
            assert!(
                (hz - pitched).abs() < pitched * 0.01,
                "tempo {tempo}: {hz} Hz, expected {pitched}"
            );
        }
    }

    #[test]
    fn locked_playback_is_clean() {
        // A pure tone through the locked chain: everything that isn't the
        // tone, from filtering or interpolation, is over 110 dB under it.
        let input = tone(1000.0, 3.0);
        for tempo in [0.5f32, 0.8, 1.25, 1.5, 2.0] {
            let output = run_locked(&mut Processor::new(), &input, tempo);
            let clean = sinad_db(&output, &[1000.0 * f64::from(tempo)]);
            eprintln!("1 kHz locked at {tempo}x: {clean:.1} dB");
            assert!(clean > 110.0, "tempo {tempo}: {clean:.1} dB");
        }
    }

    #[test]
    fn locking_avoids_the_stretchs_artefacts() {
        // A chord, two notes a major third apart, is where a time stretch
        // has to compromise: no one grain lines up with both. Locked, the
        // chord comes through as cleanly as a single tone.
        let (low, high) = (440.0, 440.0 * 2f64.powf(4.0 / 12.0));
        let frames = (RATE * 3.0) as usize;
        let input: Vec<f64> = (0..frames)
            .flat_map(|i| {
                let t = i as f64 / RATE;
                let sample = 0.25 * (std::f64::consts::TAU * low * t).sin()
                    + 0.25 * (std::f64::consts::TAU * high * t).sin();
                [sample, sample]
            })
            .collect();
        for tempo in [0.8f32, 1.25, 1.5] {
            let ratio = f64::from(tempo);
            let locked = sinad_db(
                &run_locked(&mut Processor::new(), &input, tempo),
                &[low * ratio, high * ratio],
            );
            let stretched = sinad_db(
                &run(&mut Processor::new(), &input, tempo, 0.0),
                &[low, high],
            );
            eprintln!("chord at {tempo}x: locked {locked:.1} dB, pitch kept {stretched:.1} dB");
            assert!(locked > 110.0, "tempo {tempo}: {locked:.1} dB");
            assert!(locked > stretched + 40.0);
        }
    }

    #[test]
    fn locked_speed_up_removes_what_would_alias() {
        // At 2× a 15 kHz tone would play at 30 kHz, above what the output
        // can hold; instead of folding back as a false tone, it is gone.
        let input = tone(15_000.0, 2.0);
        let output = run_locked(&mut Processor::new(), &input, 2.0);
        let level = 20.0 * (rms(&middle(&output)) / rms(&middle(&input))).log10();
        eprintln!("15 kHz at 2x: {level:.1} dB");
        assert!(level < -90.0, "{level:.1} dB");
    }

    #[test]
    fn locked_slow_down_keeps_the_top_octave() {
        // 18 kHz at 0.8× plays at 14.4 kHz, at the level it had.
        let input = tone(18_000.0, 2.0);
        let output = run_locked(&mut Processor::new(), &input, 0.8);
        let level = 20.0 * (rms(&middle(&output)) / rms(&middle(&input))).log10();
        assert!(level.abs() < 0.05, "{level:.3} dB");
        // 0.8 as the speed holds it, which is a hair over 0.8.
        let clean = sinad_db(&output, &[18_000.0 * f64::from(0.8f32)]);
        eprintln!("18 kHz at 0.8x: {clean:.1} dB");
        assert!(clean > 110.0, "{clean:.1} dB");
    }

    #[test]
    fn locking_mid_song_keeps_every_moment() {
        // Speed and pitch kept for a second, then locked at the same speed:
        // the song carries on with nothing dropped or repeated.
        let input = tone(440.0, 4.0);
        let half = input.len() / 2;
        let mut processor = Processor::new();
        let mut output = run(&mut processor, &input[..half], 1.25, 0.0);
        output.extend(run_locked(&mut processor, &input[half..], 1.25));
        let expected = frames(&input) / 1.25;
        assert!(
            (frames(&output) - expected).abs() < (GRAIN + SEEK + HOP + TAPS_HALF) as f64,
            "{} frames, expected about {expected}",
            frames(&output)
        );
        let after = &output[output.len() * 3 / 4..];
        let hz = frequency(after);
        assert!((hz - 550.0).abs() < 6.0, "{hz} Hz");
    }

    #[test]
    fn a_locked_speed_sets_the_pitch() {
        assert!((locked_semitones(1.5) - 7.02).abs() < 0.01);
        assert!((locked_semitones(0.5) + 12.0).abs() < 1e-6);
        assert_eq!(locked_semitones(1.0), 0.0);
        let speed = Speed::default();
        speed.set(1.0, 5.0);
        speed.set_locked(true);
        // Locked at normal speed plays at normal pitch, whatever the pitch
        // slider holds for later.
        assert!(speed.is_normal());
        assert_eq!(speed.pitch(), 0.0);
        speed.set_locked(false);
        assert_eq!(speed.pitch(), 5.0);
        assert!(!speed.is_normal());
    }

    #[test]
    fn values_snap_to_their_steps_and_ranges() {
        assert_eq!(snap_tempo(1.02), 1.0);
        assert_eq!(snap_tempo(1.13), 1.15);
        assert_eq!(snap_tempo(5.0), 2.0);
        assert_eq!(snap_tempo(f32::NAN), 1.0);
        assert_eq!(snap_semitones(3.4), 3.0);
        assert_eq!(snap_semitones(-30.0), -12.0);
    }

    #[test]
    fn played_time_follows_each_tempo_in_turn() {
        let speed = Speed::default();
        let start = Instant::now();
        speed.set_at(2.0, 0.0, start + Duration::from_secs(10));
        speed.set_at(0.5, 0.0, start + Duration::from_secs(20));
        let played = speed.played_between(start, start + Duration::from_secs(30));
        // 10 s at 1x, 10 s at 2x, 10 s at 0.5x.
        assert_eq!(played.as_millis(), 35_000);
        let later = speed.played_between(
            start + Duration::from_secs(15),
            start + Duration::from_secs(30),
        );
        assert_eq!(later.as_millis(), 15_000);
        // Pitch alone never moves the clock.
        speed.set_at(0.5, 7.0, start + Duration::from_secs(25));
        let pitched = speed.played_between(start, start + Duration::from_secs(30));
        assert_eq!(pitched, played);
    }
}
