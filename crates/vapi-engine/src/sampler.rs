use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use vapi_core::SamplingParams;

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
}

impl SamplerState {
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            rng: match seed {
                Some(s) => ChaCha8Rng::seed_from_u64(s),
                None => ChaCha8Rng::from_rng(&mut rand::rng()),
            },
            counts: std::collections::HashMap::new(),
        }
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

        if params.temperature != 1.0 {
            let t = params.temperature.max(1e-5);
            for l in logits.iter_mut() {
                *l /= t;
            }
        }

        let mut candidates = softmax_candidates(logits);
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
            // Repetition penalty is multiplicative and sign-aware: dividing a
            // negative logit would *increase* it.
            if rep != 1.0 {
                *l = if *l > 0.0 { *l / rep } else { *l * rep };
            }
            *l -= freq * count as f32;
            if count > 0 {
                *l -= pres;
            }
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

/// `(token_id, probability)` sorted by descending probability.
fn softmax_candidates(logits: &[f32]) -> Vec<(u32, f32)> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    let mut out: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| {
            let e = (l - max).exp();
            sum += e;
            (i as u32, e)
        })
        .collect();
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
