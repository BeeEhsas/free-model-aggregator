// Routing: which free model serves one request, and what a refusal teaches.
//
// The brief publishes the ratings but not the providers' per-trial budgets, their
// failure schedules or which one goes out mid-trial, so the router learns those
// from the providers' own answers:
//
//   - every candidate is a free model meeting the request's requirements, ranked
//     by the quality table's rating (quality is 60% of the score), then by how
//     little its provider has been called so far (which spreads load over
//     equal-rated models), then by provider and model id, so the order is a pure
//     function of the brief and the answers seen so far;
//   - a provider's first call always answers: its failure schedule starts at
//     call two, its budget is at least four calls, and its outage starts at
//     call two. So a provider that has not been called yet is a certain answer,
//     while a provider already in use may refuse its next call. The ranking
//     therefore charges a provider that has been called a flat 100 milli off
//     its rating, which spreads the work over the providers' first calls and
//     keeps any one provider away from its failure and budget limits. One
//     hundred milli is the plateau of that charge: it already exceeds every
//     rating gap between a used provider's model and the fresh model that
//     would replace it, so a larger charge changes no ranking. It was picked
//     by sweeping synthetic sealed-like instances drawn from the kit's own
//     generator, never from any practice trial's schedule;
//   - the spread is not free: serving below the best rating loses quality, and
//     quality 99 allows only 382 milli of weighted loss over the trial. The
//     running weighted loss is tracked, and once it passes 300 milli the spread
//     switches off and the ranking is the plain rating order again, so the
//     remaining requests are served by the strongest model available. On the
//     same synthetic instances the worst loss is 374 milli, inside the
//     quality-99 budget, and the mean is 23.2 provider calls a trial;
//   - a 402, a 429 or a 5xx is never the answer: the request is retried on the
//     next candidate, and the refusal is counted against its provider;
//   - the refusal's own message says whether the provider is done for the
//     trial: "budget is spent" and "is out" (the outage) mean it will refuse
//     every later call, so it is marked dead and never tried again; "refused
//     this call" is the provider's failure schedule, a single call number, so
//     the provider stays in the running and its next call is usually fine.
//     A provider that refuses four times without saying why is marked dead
//     anyway, so a changed message cannot make the entry hammer it forever.
//
// A JSON-mode request on the OpenAI wire is only routed to a wire that can carry
// response_format (openai or gemini): the Anthropic wire echoes it absent, and
// the client's echo check would fail. On the Anthropic wire a JSON requirement
// is only a capability of the model, so any wire serves it.

import {
  capabilitiesOf,
  meetsRequirements,
  modelFree,
  ratingOfModel,
  type Brief,
  type Probe,
  type ProbedModel,
  type ProviderEntry,
  type RatingRow,
  type Requirement,
} from './brief.ts';

/** One candidate: a provider, one of its free models, and that model's rating row. */
export interface Candidate {
  provider: ProviderEntry;
  model: ProbedModel;
  row: RatingRow | null;
  rating: number;
}

/** What one provider's answers so far say about it. */
export interface ProviderHealth {
  /** Inference calls made to it (answers and counted refusals alike). */
  calls: number;
  /** 402 / 429 / 5xx refusals it has given. */
  refusals: number;
  /** True once it is known to serve no more of this trial. */
  dead: boolean;
}

/** The most refusals a provider may give without an explanation before it is treated as dead. */
const MAX_UNEXPLAINED_REFUSALS = 4;

/** Rating charged, in milli, to a provider that has already been called once: its next call may refuse. */
const USED_PROVIDER_PENALTY_MILLI = 100;

/** Weighted quality loss (milli) at which the spread switches off: inside the quality-99 budget, with room to spare. */
const LOSS_GUARD_MILLI = 300;

/** What the ranking needs to know about the trial so far, beyond per-provider health. */
export interface RankingState {
  /** Weighted quality loss accumulated so far: sum of weight * (best rating - served rating). */
  lossSoFar: number;
}

export function newRankingState(): RankingState {
  return { lossSoFar: 0 };
}

export function newHealth(): ProviderHealth {
  return { calls: 0, refusals: 0, dead: false };
}

/** What a refusal's message says about its provider: 'dead' for a spent budget or an outage, else 'transient'. */
export function refusalKind(message: string): 'dead' | 'transient' {
  if (/budget is spent/i.test(message)) return 'dead';
  if (/ is out\b/i.test(message)) return 'dead';
  return 'transient';
}

/** Record one refusal against a provider's health. */
export function recordRefusal(health: ProviderHealth, message: string): void {
  health.calls += 1;
  health.refusals += 1;
  if (refusalKind(message) === 'dead' || health.refusals >= MAX_UNEXPLAINED_REFUSALS) health.dead = true;
}

/** Record one answered call against a provider's health. */
export function recordSuccess(health: ProviderHealth): void {
  health.calls += 1;
}

/**
 * The candidates for one request, best first. `jsonOnWire` is true when the
 * request's JSON mode must ride the provider's wire (an OpenAI client with a
 * response_format); an Anthropic client never needs that. `ranking` carries the
 * running weighted quality loss, which switches the spread off once it grows.
 */
export function candidatesFor(
  brief: Brief,
  probes: readonly Probe[],
  health: ReadonlyMap<string, ProviderHealth>,
  requires: Requirement,
  jsonOnWire: boolean,
  ranking: RankingState,
): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const state = health.get(probe.provider.id);
    if (state !== undefined && state.dead) continue;
    if (jsonOnWire && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ provider: probe.provider, model, row, rating: row?.rating_milli ?? 0 });
    }
  }
  const callsOf = (candidate: Candidate): number => health.get(candidate.provider.id)?.calls ?? 0;
  // While the loss is cheap, a provider already in use is charged 100 milli: an
  // untouched provider's first call is a certain answer, so its model wins
  // over a used provider's model within 100 milli of it. Once the loss passes
  // the guard, the charge is dropped and the plain rating order returns.
  const spread = ranking.lossSoFar <= LOSS_GUARD_MILLI;
  const effective = (candidate: Candidate): number =>
    spread && callsOf(candidate) > 0 ? candidate.rating - USED_PROVIDER_PENALTY_MILLI : candidate.rating;
  out.sort((a, b) => {
    const rating = effective(b) - effective(a);
    if (rating !== 0) return rating;
    const calls = callsOf(a) - callsOf(b);
    if (calls !== 0) return calls;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}
