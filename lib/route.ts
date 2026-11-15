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
 * response_format); an Anthropic client never needs that.
 */
export function candidatesFor(
  brief: Brief,
  probes: readonly Probe[],
  health: ReadonlyMap<string, ProviderHealth>,
  requires: Requirement,
  jsonOnWire: boolean,
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
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    const calls = callsOf(a) - callsOf(b);
    if (calls !== 0) return calls;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}
