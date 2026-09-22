use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use vapi_core::SamplingParams;

use crate::backend::{CandidateNeed, GumbelDraw, RowCandidates};

/// Per-sequence sampling state.
///
/// The RNG lives here, per sequence, rather than in a shared engine-level
/// generator. With a shared RNG a seeded request's output would depend on what
/// else happened to be in the batch alongside it, which makes `seed`
/// meaningless and breaks the tier-2 response cache's determinism guarantee.
#[derive(Debug)]
pub struct SamplerState {
    rng: ChaCha8Rng,
    /// Occurrence counts for the penalties, indexed by token id.
    counts: std::collections::HashMap<u32, u32>,
    /// Seed for device-side draws: the request's, or a random one.
    device_seed: u64,
    /// Number of device-side draws made so far; each uses fresh noise.
    device_draws: u64,
}

impl SamplerState {
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            rng: match seed {
                Some(s) => ChaCha8Rng::seed_from_u64(s),
                None => ChaCha8Rng::from_rng(&mut rand::rng()),
            },
            counts: std::collections::HashMap::new(),
            device_seed: seed.unwrap_or_else(|| rand::rng().random::<u64>()),
            device_draws: 0,
        }
    }

    /// Account for a token the backend drew on the device.
    pub fn observe_device_draw(&mut self, token: u32) {
        self.device_draws += 1;
        self.observe(token);
    }

    /// Whether these parameters are nothing but temperature, which a
    /// Gumbel-max draw on the device samples exactly.
    pub fn device_drawable(params: &SamplingParams) -> bool {
        !params.is_greedy() && params.top_k.is_none() && params.top_p >= 1.0 && params.min_p <= 0.0
    }

    /// Record a token for the repetition/frequency/presence penalties.
    pub fn observe(&mut self, token: u32) {
        *self.counts.entry(token).or_insert(0) += 1;
    }

    pub fn observe_all(&mut self, tokens: &[u32]) {
        for &t in tokens {
            self.observe(t);
        }
    }

    /// Number of distinct tokens the penalties can touch.
    pub fn distinct_observed(&self) -> usize {
        self.counts.len()
    }

    /// What a device-side candidate selection must return for this row to
    /// sample exactly as [`Sampler::sample`] would from the full logits.
    ///
    /// Penalties here only ever lower a logit, so any token outside the raw
    /// top `k + distinct_observed` cannot enter the post-penalty top `k`.
    /// Greedy is top-1. Without a top-k (or with a penalty that can raise a
    /// logit) only the complete candidate window will do, so `needed` is 0.
    pub fn need(&self, params: &SamplingParams) -> CandidateNeed {
        if params.logprobs.is_some() {
            return CandidateNeed {
                inv_temperature: 1.0,
                needed: 0,
                gumbel: None,
                full: true,
            };
        }
        let raising = params.repetition_penalty < 1.0
            || params.frequency_penalty < 0.0
            || params.presence_penalty < 0.0;
        let k = if params.is_greedy() {
            Some(1)
        } else {
            params.top_k
        };
        CandidateNeed {
            inv_temperature: if params.is_greedy() {
                1.0
            } else {
                1.0 / params.temperature.max(1e-5)
            },
            needed: match (k, raising) {
                (Some(k), false) => k.max(1) + self.distinct_observed(),
                _ => 0,
            },
            gumbel: Self::device_drawable(params).then(|| GumbelDraw {
                seed: self.device_seed,
                draw: self.device_draws,
                observed: self.counts.iter().map(|(&t, &c)| (t, c)).collect(),
                repetition_penalty: params.repetition_penalty,
                frequency_penalty: params.frequency_penalty,
                presence_penalty: params.presence_penalty,
            }),
            full: false,
        }
    }
}

/// Log-probabilities of a row under the model's own distribution.
#[derive(Clone, Debug, PartialEq)]
pub struct RowLogprobs {
    /// `log Σ exp(l)` over the row, so `logprob(t) = l_t - lse`.
    pub lse: f32,
    /// The `k` largest logits as `(token, logprob)`, largest first.
    pub top: Vec<(u32, f32)>,
}

impl RowLogprobs {
    /// Two passes over the row plus a partial sort of `k` entries. Done on
    /// the raw logits, before temperature and penalties: what the client
    /// asked about is the model, not the sampler.
    pub fn of(logits: &[f32], k: usize) -> Self {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
        let lse = max + sum.ln();
        let mut top: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
        for (i, &l) in logits.iter().enumerate() {
            if top.len() < k {
                top.push((i as u32, l));
                top.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            } else if let Some(last) = top.last()
                && l > last.1
            {
                top.pop();
                let pos = top.partition_point(|x| x.1 >= l);
                top.insert(pos, (i as u32, l));
            }
        }
        for t in top.iter_mut() {
            t.1 -= lse;
        }
        Self { lse, top }
    }

    pub fn logprob(&self, raw_logit: f32) -> f32 {
        raw_logit - self.lse
    }
}

pub struct Sampler;

impl Sampler {
    /// Pick one token from a row of logits.
    ///
    /// Operates on a mutable slice so penalties are applied in place and no
    /// vocab-sized allocation happens per token.
    pub fn sample(logits: &mut [f32], params: &SamplingParams, state: &mut SamplerState) -> u32 {
        Self::apply_penalties(logits, params, state);

        if params.is_greedy() {
            return argmax(logits);
        }

        let mut candidates = candidates(logits, params.temperature);
        if let Some(k) = params.top_k {
            truncate_top_k(&mut candidates, k);
        }
        if params.min_p > 0.0 {
            truncate_min_p(&mut candidates, params.min_p);
        }
        if params.top_p < 1.0 {
            truncate_top_p(&mut candidates, params.top_p);
        }
        weighted_choice(&candidates, &mut state.rng)
    }

    /// [`Self::sample`] over a row's device-selected candidates.
    ///
    /// The candidates are the tokens above a threshold, so the penalties
    /// apply to those present (an observed token below the threshold can
    /// only sink further, and the backend guaranteed the row is covered).
    /// Probabilities are normalised by the row's full-vocabulary sum,
    /// corrected for what the penalties moved, so `top_p` sees the same
    /// numbers it would from the full logits.
    pub fn sample_candidates(
        row: &mut RowCandidates,
        params: &SamplingParams,
        state: &mut SamplerState,
    ) -> u32 {
        let inv_t = if params.is_greedy() {
            1.0
        } else {
            1.0 / params.temperature.max(1e-5)
        };
        let raw_max = row.max;
        let mut sum = row.sum;
        let penalised = params.repetition_penalty != 1.0
            || params.frequency_penalty != 0.0
            || params.presence_penalty != 0.0;
        if penalised {
            for (tok, l) in row.entries.iter_mut() {
                if let Some(&count) = state.counts.get(tok) {
                    let before = ((*l - raw_max) * inv_t).exp();
                    Self::penalise(l, count, params);
                    let after = ((*l - raw_max) * inv_t).exp();
                    sum += after - before;
                }
            }
        }

        if params.is_greedy() {
            // First index wins ties in `argmax`, i.e. the lowest id.
            let mut best: Option<(u32, f32)> = None;
            for &(t, l) in &row.entries {
                best = match best {
                    Some((bt, bl)) if l < bl || (l == bl && t > bt) => Some((bt, bl)),
                    _ => Some((t, l)),
                };
            }
            return best.map(|(t, _)| t).unwrap_or(0);
        }

        // Same arithmetic as `candidates`: weights relative to the row max
        // (which cannot rise under a lowering penalty; if one raised it the
        // backend fell back to the full logits), scaled by 1 / temperature.
        let mut cands: Vec<(u32, f32)> = row
            .entries
            .iter()
            .map(|&(t, l)| (t, ((l - raw_max) * inv_t).exp()))
            .collect();
        let local: f32 = cands.iter().map(|c| c.1).sum();
        let denom = if sum > 0.0 { sum.max(local) } else { local };
        if denom > 0.0 {
            for c in cands.iter_mut() {
                c.1 /= denom;
            }
        }
        cands.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        if let Some(k) = params.top_k {
            truncate_top_k(&mut cands, k);
        }
        if params.min_p > 0.0 {
            truncate_min_p(&mut cands, params.min_p);
        }
        if params.top_p < 1.0 {
            truncate_top_p(&mut cands, params.top_p);
        }
        weighted_choice(&cands, &mut state.rng)
    }

    fn penalise(l: &mut f32, count: u32, params: &SamplingParams) {
        let rep = params.repetition_penalty;
        // Repetition penalty is multiplicative and sign-aware: dividing a
        // negative logit would *increase* it.
        if rep != 1.0 {
            *l = if *l > 0.0 { *l / rep } else { *l * rep };
        }
        *l -= params.frequency_penalty * count as f32;
        if count > 0 {
            *l -= params.presence_penalty;
        }
    }

    fn apply_penalties(logits: &mut [f32], params: &SamplingParams, state: &SamplerState) {
        let rep = params.repetition_penalty;
        let freq = params.frequency_penalty;
        let pres = params.presence_penalty;
        if rep == 1.0 && freq == 0.0 && pres == 0.0 {
            return;
        }
        for (&tok, &count) in &state.counts {
            let Some(l) = logits.get_mut(tok as usize) else {
                continue;
            };
            Self::penalise(l, count, params);
        }
    }
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// Tokens further than this below the row maximum (after temperature)
/// get `exp(-30) ≈ 9e-14` of the top token's mass. Over a 128K vocabulary
/// that is at most ~1e-8 of the total, below f32 resolution of the sum, so
/// leaving them out changes nothing a caller can observe while cutting the
/// work per row from "sort the vocabulary" to "sort the plausible tokens".
const CANDIDATE_CUTOFF: f32 = 30.0;

/// `(token_id, probability)` for every token that can carry mass, sorted
/// by descending probability, ties by id.
///
/// Two O(vocab) passes with no allocation (max, then exp over the tokens
/// above the cutoff) and a sort of only the survivors. The old
/// implementation built and sorted all 128K pairs per row, which was ~4 ms
/// a row and dominated the step at batch 64.
fn candidates(logits: &[f32], temperature: f32) -> Vec<(u32, f32)> {
    let inv_t = 1.0 / temperature.max(1e-5);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) * inv_t;
    let floor = max - CANDIDATE_CUTOFF;
    let mut out: Vec<(u32, f32)> = Vec::with_capacity(256);
    let mut sum = 0.0f32;
    for (i, &l) in logits.iter().enumerate() {
        let l = l * inv_t;
        if l >= floor {
            let e = (l - max).exp();
            sum += e;
            out.push((i as u32, e));
        }
    }
    if sum > 0.0 {
        for c in out.iter_mut() {
            c.1 /= sum;
        }
    }
    out.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    out
}

fn truncate_top_k(c: &mut Vec<(u32, f32)>, k: usize) {
    c.truncate(k.max(1));
}

/// Keep tokens with probability at least `min_p × p_max`. Scales with
/// confidence, unlike a fixed top-p.
fn truncate_min_p(c: &mut Vec<(u32, f32)>, min_p: f32) {
    let Some(&(_, top)) = c.first() else { return };
    let floor = min_p * top;
    let keep = c.iter().take_while(|(_, p)| *p >= floor).count().max(1);
    c.truncate(keep);
}

/// Keep the smallest prefix whose cumulative probability reaches `top_p`.
fn truncate_top_p(c: &mut Vec<(u32, f32)>, top_p: f32) {
    let mut acc = 0.0f32;
    let mut keep = 0usize;
    for (_, p) in c.iter() {
        acc += *p;
        keep += 1;
        if acc >= top_p {
            break;
        }
    }
    c.truncate(keep.max(1));
}

fn weighted_choice(c: &[(u32, f32)], rng: &mut ChaCha8Rng) -> u32 {
    let total: f32 = c.iter().map(|(_, p)| *p).sum();
    if total <= 0.0 {
        return c.first().map(|(t, _)| *t).unwrap_or(0);
    }
    let mut r = rng.random::<f32>() * total;
    for &(t, p) in c {
        r -= p;
        if r <= 0.0 {
            return t;
        }
    }
    // Floating-point drift can exhaust the loop; the last candidate is the
    // correct fallback, not token 0.
    c.last().map(|(t, _)| *t).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn greedy() -> SamplingParams {
        SamplingParams {
            temperature: 0.0,
            ..Default::default()
        }
    }

    #[test]
    fn greedy_takes_the_argmax() {
        let mut logits = vec![0.1, 5.0, 0.3];
        let mut st = SamplerState::new(None);
        assert_eq!(Sampler::sample(&mut logits, &greedy(), &mut st), 1);
    }

    #[test]
    fn greedy_is_reproducible_without_a_seed() {
        // Greedy must not touch the RNG at all, or "temperature 0" would not
        // be deterministic across processes.
        for _ in 0..20 {
            let mut logits = vec![1.0, 2.0, 1.5];
            let mut st = SamplerState::new(None);
            assert_eq!(Sampler::sample(&mut logits, &greedy(), &mut st), 1);
        }
    }

    #[test]
    fn the_same_seed_gives_the_same_token() {
        let params = SamplingParams {
            temperature: 1.0,
            seed: Some(7),
            ..Default::default()
        };
        let run = || {
            let mut logits = vec![1.0, 1.1, 0.9, 1.05];
            let mut st = SamplerState::new(params.seed);
            Sampler::sample(&mut logits, &params, &mut st)
        };
        let first = run();
        for _ in 0..10 {
            assert_eq!(run(), first, "a seeded request must be reproducible");
        }
    }

    #[test]
    fn different_seeds_explore_different_tokens() {
        let sample_with = |seed| {
            let params = SamplingParams {
                temperature: 2.0,
                seed: Some(seed),
                ..Default::default()
            };
            let mut logits = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
            let mut st = SamplerState::new(params.seed);
            Sampler::sample(&mut logits, &params, &mut st)
        };
        let seen: std::collections::HashSet<u32> = (0..40).map(sample_with).collect();
        assert!(
            seen.len() > 1,
            "a uniform distribution should not collapse to one token"
        );
    }

    /// The implementation this replaced: full softmax over the vocabulary,
    /// then sort everything. Kept as the oracle.
    fn reference_candidates(logits: &[f32], temperature: f32) -> Vec<(u32, f32)> {
        let t = temperature.max(1e-5);
        let scaled: Vec<f32> = logits.iter().map(|l| l / t).collect();
        let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        let mut out: Vec<(u32, f32)> = scaled
            .iter()
            .enumerate()
            .map(|(i, &l)| {
                let e = (l - max).exp();
                sum += e;
                (i as u32, e)
            })
            .collect();
        for c in out.iter_mut() {
            c.1 /= sum;
        }
        out.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out
    }

    #[test]
    fn the_candidate_cutoff_changes_nothing_observable() {
        // Random logits with a realistic spread, at several temperatures:
        // the surviving prefix (everything a truncation could keep) must
        // match the full-vocabulary reference in order and in probability.
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        for &t in &[0.1f32, 0.7, 1.0, 2.0] {
            let logits: Vec<f32> = (0..20_000)
                .map(|_| rng.random::<f32>() * 40.0 - 20.0)
                .collect();
            let fast = candidates(&logits, t);
            let full = reference_candidates(&logits, t);
            assert!(fast.len() <= full.len());
            for (a, b) in fast.iter().zip(&full) {
                assert_eq!(a.0, b.0, "order differs at temperature {t}");
                // Scaling by 1/t rather than dividing by t rounds differently,
                // and exp amplifies that; 1e-4 relative is far below anything
                // a truncation or a draw could notice.
                assert!(
                    (a.1 - b.1).abs() <= 1e-4 * b.1 + 1e-9,
                    "probability differs at temperature {t}: {} vs {}",
                    a.1,
                    b.1
                );
            }
            let mass: f32 = fast.iter().map(|c| c.1).sum();
            assert!((mass - 1.0).abs() < 1e-5, "mass {mass} at temperature {t}");
        }
    }

    /// A host model of the device selection: the row's tokens above
    /// `max - window / temperature`, or its top `count` tokens, plus the
    /// full softmax denominator.
    fn select(logits: &[f32], inv_t: f32, keep: impl Fn(usize, f32) -> bool) -> RowCandidates {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = logits.iter().map(|&l| ((l - max) * inv_t).exp()).sum();
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
        let entries = order
            .iter()
            .enumerate()
            .filter(|&(rank, &i)| keep(rank, logits[i]))
            .map(|(_, &i)| (i as u32, logits[i]))
            // Scramble the order: the device compaction has none.
            .rev()
            .collect();
        RowCandidates { entries, max, sum }
    }

    #[test]
    fn candidates_sample_exactly_like_the_full_logits() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let logits: Vec<f32> = (0..5_000)
            .map(|_| rng.random::<f32>() * 30.0 - 15.0)
            .collect();
        let seen: Vec<u32> = (0..40).map(|_| rng.random_range(0..5_000u32)).collect();
        let cases = [
            // complete window, penalties, top-p
            SamplingParams {
                temperature: 0.7,
                top_p: 0.9,
                repetition_penalty: 1.3,
                presence_penalty: 0.5,
                ..Default::default()
            },
            // top-k with penalties: only top k + observed are needed
            SamplingParams {
                temperature: 1.0,
                top_k: Some(50),
                repetition_penalty: 1.1,
                frequency_penalty: 0.2,
                ..Default::default()
            },
            // greedy with a penalty
            SamplingParams {
                temperature: 0.0,
                repetition_penalty: 2.0,
                ..Default::default()
            },
            SamplingParams {
                temperature: 0.3,
                top_k: Some(5),
                min_p: 0.1,
                ..Default::default()
            },
        ];
        for params in cases {
            let mut full_state = SamplerState::new(Some(3));
            full_state.observe_all(&seen);
            let mut cand_state = SamplerState::new(Some(3));
            cand_state.observe_all(&seen);
            let need = cand_state.need(&params);
            let mut row = if need.needed > 0 {
                select(&logits, need.inv_temperature, |rank, _| rank < need.needed)
            } else {
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let floor = max - CANDIDATE_CUTOFF / need.inv_temperature;
                select(&logits, need.inv_temperature, |_, l| l >= floor)
            };
            for _ in 0..200 {
                let mut copy = logits.clone();
                let want = Sampler::sample(&mut copy, &params, &mut full_state);
                let mut r = row.clone();
                let got = Sampler::sample_candidates(&mut r, &params, &mut cand_state);
                assert_eq!(got, want, "params {params:?}");
                full_state.observe(want);
                cand_state.observe(got);
                row = if need.needed > 0 {
                    let n = cand_state.need(&params).needed;
                    select(&logits, need.inv_temperature, |rank, _| rank < n)
                } else {
                    row
                };
            }
        }
    }

    #[test]
    fn row_logprobs_are_a_log_softmax() {
        let logits = [1.0f32, 3.0, 2.0, -1.0, 3.0];
        let lp = RowLogprobs::of(&logits, 3);
        let z: f32 = logits.iter().map(|l| l.exp()).sum();
        for (i, &l) in logits.iter().enumerate() {
            let want = (l.exp() / z).ln();
            assert!((lp.logprob(l) - want).abs() < 1e-5, "token {i}");
        }
        // Ties by id, largest first.
        let ids: Vec<u32> = lp.top.iter().map(|t| t.0).collect();
        assert_eq!(ids, vec![1, 4, 2]);
        let mass: f32 = lp.top.iter().map(|t| t.1.exp()).sum();
        assert!(mass < 1.0 && mass > 0.9);
    }

    #[test]
    fn top_k_of_one_is_equivalent_to_greedy() {
        let params = SamplingParams {
            temperature: 1.0,
            top_k: Some(1),
            seed: Some(1),
            ..Default::default()
        };
        let mut logits = vec![0.5, 9.0, 0.2, 3.0];
        let mut st = SamplerState::new(params.seed);
        assert_eq!(Sampler::sample(&mut logits, &params, &mut st), 1);
    }

    #[test]
    fn top_p_excludes_the_tail() {
        // One token holds ~all the mass; top_p 0.5 must never reach the rest.
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 0.5,
            seed: Some(3),
            ..Default::default()
        };
        for _ in 0..50 {
            let mut logits = vec![20.0, 0.0, 0.0, 0.0];
            let mut st = SamplerState::new(None);
            assert_eq!(Sampler::sample(&mut logits, &params, &mut st), 0);
        }
    }

    #[test]
    fn repetition_penalty_pushes_away_from_seen_tokens() {
        let params = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 100.0,
            ..Default::default()
        };
        let mut st = SamplerState::new(None);
        st.observe(1);
        // Token 1 leads before the penalty; penalising it must hand the win
        // to token 2.
        let mut logits = vec![0.0, 5.0, 4.0];
        assert_eq!(Sampler::sample(&mut logits, &params, &mut st), 2);
    }

    #[test]
    fn repetition_penalty_does_not_reward_negative_logits() {
        // Dividing a negative logit by a >1 penalty would raise it toward
        // zero, i.e. make a penalised token *more* likely.
        let params = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 2.0,
            ..Default::default()
        };
        let mut st = SamplerState::new(None);
        st.observe(0);
        let mut logits = vec![-1.0, -1.5];
        assert_eq!(Sampler::sample(&mut logits, &params, &mut st), 1);
    }

    #[test]
    fn frequency_penalty_scales_with_count() {
        let params = SamplingParams {
            temperature: 0.0,
            frequency_penalty: 1.0,
            ..Default::default()
        };
        let mut st = SamplerState::new(None);
        st.observe_all(&[0, 0, 0]);
        let mut logits = vec![2.0, 0.0];
        assert_eq!(
            Sampler::sample(&mut logits, &params, &mut st),
            1,
            "3 × 1.0 penalty beats the 2.0 lead"
        );
    }

    #[test]
    fn sampling_never_returns_an_out_of_range_token() {
        for seed in 0..50u64 {
            let params = SamplingParams {
                temperature: 1.5,
                top_p: 0.9,
                seed: Some(seed),
                ..Default::default()
            };
            let mut logits = vec![0.3, 0.1, 0.9, 0.2, 0.5];
            let mut st = SamplerState::new(params.seed);
            let t = Sampler::sample(&mut logits, &params, &mut st);
            assert!((t as usize) < 5, "token {t} is outside the vocab");
        }
    }
}
