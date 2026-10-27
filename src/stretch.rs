//! Playback speed and pitch for local playback.
//!
//! Speed and pitch are independent. Speed changes how fast the song plays
//! without changing its key; pitch moves the key in semitones without
//! changing its speed. Both apply to this computer's output only, like the
//! equalizer; a Connect device or Cast speaker plays at normal speed.
//!
//! Two stages do the work, both pure Rust and run on the player's thread:
//!
//! 1. WSOLA (waveform-similarity overlap-add) changes the duration. Each
//!    40 ms Hann-windowed grain is taken from near where the speed says it
//!    should come from, nudged up to 12 ms either way to wherever it best
//!    continues the previous grain, so the waveform stays coherent.
//! 2. A windowed-sinc resampler with a variable ratio moves the pitch, and
//!    the first stage is asked for the matching extra stretch so the speed
//!    is unchanged. Pitch up lowers the cutoff first, so it doesn't alias.
//!
//! At normal speed and pitch the processor is bypassed and the samples go
//! through untouched. Once engaged, it keeps running (at normal settings it
//! reproduces its input exactly) until the next seek or skip, so returning
//! to normal never drops the audio it was holding.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, Ordering};
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
const TAPS_HALF: usize = 8;
/// Kernel table entries per input frame.
const KERNEL_STEPS: usize = 256;
/// Tempo changes remembered to work out how far a song has moved.
const HISTORY: usize = 32;

/// The speed and pitch the player follows, shared by the controls, the
/// player's thread and the position shown under the song.
pub struct Speed {
    tempo: AtomicU32,
    semitones: AtomicU32,
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

    /// Pitch in semitones from the original.
    pub fn semitones(&self) -> f32 {
        f32::from_bits(self.semitones.load(Ordering::Relaxed))
    }

    pub fn is_normal(&self) -> bool {
        is_normal(self.tempo(), self.semitones())
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

fn is_normal(tempo: f32, semitones: f32) -> bool {
    (tempo - 1.0).abs() < f32::EPSILON && semitones.abs() < f32::EPSILON
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
        self.process_with(samples, speed.tempo(), speed.semitones())
    }

    /// Processes one packet at the given speed and pitch. The output may be
    /// shorter or longer than the input, and empty while it fills.
    pub fn process_with(&mut self, samples: Vec<f64>, tempo: f32, semitones: f32) -> Vec<f64> {
        if !self.engaged {
            if is_normal(tempo, semitones) {
                return samples;
            }
            *self = Self::new();
            self.engaged = true;
        }
        let tempo = f64::from(snap_tempo(tempo));
        let ratio = 2f64.powf(f64::from(snap_semitones(semitones)) / 12.0);
        let stretched = self.stretch.process(&samples, tempo / ratio);
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
        self.engaged = !is_normal(tempo, semitones);
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
        // The nominal place wins ties, so steady input at normal speed
        // comes out unchanged.
        let mut best = nominal.max(low);
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
        // Raising the pitch squeezes the spectrum up: cut what would fold
        // back past the output's Nyquist frequency.
        let cutoff = if ratio > 1.0 { 0.97 / ratio } else { 1.0 };
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

/// A Blackman-windowed sinc low-pass at `cutoff` (1 is the input's
/// Nyquist frequency), sampled across the taps.
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
            let phase = std::f64::consts::PI * (x / span + 1.0);
            let window = 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos();
            cutoff * sinc * window
        })
        .collect()
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
        input
            .chunks(1024 * CHANNELS)
            .flat_map(|chunk| processor.process_with(chunk.to_vec(), tempo, semitones))
            .collect()
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
