// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// The trial gives this server a brief and an env, and runs it as `node /bot/main.ts`
// from the repository's root, with Node 22's standard library alone (nothing is
// installed; TypeScript runs directly, and a relative import needs its .ts
// extension). The brief is at the path in $MAKEFUN_PROMPT_FILE. It carries three
// JSON documents:
//   - the provider directory: ten simulated free-tier providers, each with its id,
//     wire (openai, anthropic or gemini), auth scheme, key environment variable,
//     the key the service takes, and the base path you join to $MAKEFUN_ENV_URL;
//   - the quality table: every model's public rating and what it can do (tools,
//     vision, JSON mode, context length), and whether it is free;
//   - the workload manifest: the twenty requests the client will send, in order,
//     with each one's api, whether it streams, its weight and the requirements a
//     model must meet to serve it.
// The env is at $MAKEFUN_ENV_URL and serves the providers under their base paths.
// This server reads the brief, probes each provider's catalog route in the
// provider's own wire, builds one catalog, then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//      x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//      /v1/messages, each with a JSON body or with server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//      process still running when the wall ends scores zero, so never wait for anything else.
// `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
// body's catalog must be that same document. Build it once, as this file does, and use the one object.
//
// ROUTING (lib/route.ts): every request is served by the strongest free model
// meeting its requirements, ranked by the quality table's rating, then by how
// little its provider has been called so far. A 402, a 429 or a 5xx is never the
// answer: the request is retried on the next candidate, and the refusal's own
// message says whether the provider is done for the trial (a spent budget, an
// outage) or only refused that one call (its failure schedule).
//
// WIRES (lib/wire.ts): each request is translated into the provider's wire and
// the answer back into the client's, in JSON and in SSE, carrying the system
// prompt, the user text (images included), the tools, the stop sequences,
// max_tokens and JSON mode untouched. A JSON-mode request on the OpenAI wire is
// only routed to a wire that can carry response_format. A streamed OpenAI call
// asks for stream_options.include_usage, because the client never sends it and
// without it the provider's stream carries no usage chunk.
//
// DETERMINISM: no clock, no unseeded randomness. Every choice is a function of
// the brief, the requests and the providers' answers so far, so a replay leaves
// the same state export byte for byte.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/brief.ts       the brief's types and parser, the auth headers, the catalog probe.
//   lib/route.ts       candidate ranking and what a refusal teaches about a provider.
//   lib/wire.ts        the six request translations and the four stream translations.
//   LICENSE            MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  isRecord,
  joinBase,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/brief.ts';
import {
  anthropicAnswerToOpenAI,
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIAnswerToAnthropic,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  openAIToAnthropic,
  openAIToGemini,
  providerErrorMessage,
  wantsJson,
  type ServedModel,
} from './lib/wire.ts';
import {
  candidatesFor,
  newHealth,
  recordRefusal,
  recordSuccess,
  type Candidate,
  type ProviderHealth,
} from './lib/route.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const PROVIDER_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** What the providers' answers so far say about each of them, by provider id. */
const healthByProvider = new Map<string, ProviderHealth>();
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function healthOf(providerId: string): ProviderHealth {
  let health = healthByProvider.get(providerId);
  if (health === undefined) {
    health = newHealth();
    healthByProvider.set(providerId, health);
  }
  return health;
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: wantsJson(body) };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendRaw(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One provider's answer: its status, its parsed JSON (non-stream), its raw text, or its stream. */
interface UpstreamResult {
  status: number;
  json: Json | null;
  rawText: string;
  stream: ReadableStream<Uint8Array> | null;
  errorMessage: string;
}

/** An OpenAI client body for an OpenAI provider: the model replaced, and usage asked for on a stream. */
function openAIRequest(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  if (body.stream === true) out.stream_options = { include_usage: true };
  return out;
}

/** Whether a provider's 200 JSON answer has the shape its wire promises. */
function validJsonShape(wire: 'openai' | 'anthropic' | 'gemini', json: Json): boolean {
  if (wire === 'openai') {
    const choices = Array.isArray(json.choices) ? json.choices : [];
    return choices.length > 0 && isRecord(choices[0]) && isRecord(choices[0].message);
  }
  if (wire === 'anthropic') return json.type === 'message' && Array.isArray(json.content);
  const candidates = Array.isArray(json.candidates) ? json.candidates : [];
  return candidates.length > 0 && isRecord(candidates[0]) && isRecord(candidates[0].content);
}

/** One inference call to the routed provider, in its own wire. */
async function callProvider(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Promise<UpstreamResult> {
  const provider = candidate.provider;
  const base = joinBase(ENV_URL, provider.base);
  const stream = body.stream === true;
  let url: string;
  let payload: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    payload = clientApi === 'openai' ? openAIRequest(body, candidate.model.id) : anthropicToOpenAI(body, candidate.model.id);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    payload = clientApi === 'openai' ? openAIToAnthropic(body, candidate.model.id) : { ...body, model: candidate.model.id };
  } else {
    const method = stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${method}${stream ? '?alt=sse' : ''}`;
    payload = clientApi === 'openai' ? openAIToGemini(body, candidate.model.id) : anthropicToGemini(body, candidate.model.id);
  }
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (stream) headers.accept = 'text/event-stream';
  const res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(payload), signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS) });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    let message = '';
    try {
      message = providerErrorMessage(JSON.parse(text));
    } catch {
      message = text.slice(0, 200);
    }
    return { status: res.status, json: null, rawText: '', stream: null, errorMessage: message };
  }
  if (stream) return { status: 200, json: null, rawText: '', stream: res.body, errorMessage: '' };
  const text = await res.text();
  let json: Json | null = null;
  try {
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed)) json = parsed;
  } catch {
    // Not JSON: json stays null and the caller treats the call as failed.
  }
  return { status: 200, json, rawText: text, stream: null, errorMessage: '' };
}

/** A provider's JSON answer rewritten into the client's wire (same-wire answers pass through untouched). */
function translateJson(clientApi: 'openai' | 'anthropic', providerWire: 'openai' | 'anthropic' | 'gemini', json: Json, model: string): Json {
  if (clientApi === 'openai' && providerWire === 'anthropic') return anthropicAnswerToOpenAI(json, model);
  if (clientApi === 'openai' && providerWire === 'gemini') return geminiToOpenAI(json, model);
  if (clientApi === 'anthropic' && providerWire === 'openai') return openAIAnswerToAnthropic(json, model);
  if (clientApi === 'anthropic' && providerWire === 'gemini') return geminiToAnthropic(json, model);
  return json;
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** Write translated events to the client as they arrive. */
async function pipeEvents(events: AsyncGenerator<string>, res: ServerResponse): Promise<void> {
  for await (const event of events) res.write(event);
  res.end();
}

/** What one candidate's attempts came to: an answer, a fatal error, or move on to the next candidate. */
type AttemptOutcome =
  | { kind: 'ok'; result: UpstreamResult }
  | { kind: 'fatal'; status: number; message: string }
  | { kind: 'next' };

/**
 * Try one candidate, with one immediate retry of the same provider after a
 * transient refusal: a scheduled refusal is one call number, so the provider's
 * next call is usually fine, and retrying it keeps the better model instead of
 * falling back. A refusal that says the provider is done for the trial earns
 * no retry, and neither does a 400 or a 401 (the request itself was wrong).
 */
async function tryCandidate(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Promise<AttemptOutcome> {
  const health = healthOf(candidate.provider.id);
  const stream = body.stream === true;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    let result: UpstreamResult;
    try {
      result = await callProvider(candidate, clientApi, body);
    } catch {
      // The provider could not be reached at all: a refusal with nothing said.
      recordRefusal(health, '');
      if (attempt === 0) continue;
      return { kind: 'next' };
    }
    if (result.status === 200) {
      const usable = stream ? result.stream !== null : result.json !== null && validJsonShape(candidate.provider.wire, result.json);
      if (usable) {
        recordSuccess(health);
        return { kind: 'ok', result };
      }
      // A 200 that is not a usable answer: a refusal with nothing said.
      recordRefusal(health, '');
      if (attempt === 0) continue;
      return { kind: 'next' };
    }
    if (result.status === 402 || result.status === 429 || result.status >= 500) {
      recordRefusal(health, result.errorMessage);
      if (health.dead) return { kind: 'next' };
      if (attempt === 0) continue;
      return { kind: 'next' };
    }
    return { kind: 'fatal', status: result.status, message: result.errorMessage };
  }
  return { kind: 'next' };
}

/**
 * One client request, served by the best free model that meets it, failing over
 * to the next candidate whenever a provider refuses. A refusal that says the
 * provider is done for the trial marks it dead; a scheduled refusal does not.
 */
async function answerRequest(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const requires = requirementsFor(body);
  const jsonOnWire = clientApi === 'openai' && wantsJson(body);
  const candidates = candidatesFor(brief, probes, healthByProvider, requires, jsonOnWire);
  if (candidates.length === 0) {
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  const stream = body.stream === true;
  for (const candidate of candidates) {
    const outcome = await tryCandidate(candidate, clientApi, body);
    if (outcome.kind === 'next') continue;
    if (outcome.kind === 'fatal') {
      // A 400 or a 401 says the request itself was wrong: another provider would answer the same way.
      return sendError(res, clientApi, outcome.status, clientApi === 'anthropic' ? 'api_error' : 'invalid_request_error', `the provider answered HTTP ${outcome.status}: ${outcome.message.slice(0, 200)}`);
    }
    const result = outcome.result;
    if (stream) {
      // tryCandidate only returns ok for a stream with a body.
      const upstream = result.stream as ReadableStream<Uint8Array>;
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      const wire = candidate.provider.wire;
      if (clientApi === 'openai' && wire === 'openai') await pipeStream(upstream, res);
      else if (clientApi === 'openai' && wire === 'anthropic') await pipeEvents(openAIEventsFromAnthropic(upstream, candidate.model.id), res);
      else if (clientApi === 'openai') await pipeEvents(openAIEventsFromGemini(upstream, candidate.model.id), res);
      else if (clientApi === 'anthropic' && wire === 'openai') await pipeEvents(anthropicEventsFromOpenAI(upstream, candidate.model.id), res);
      else if (clientApi === 'anthropic' && wire === 'anthropic') await pipeStream(upstream, res);
      else await pipeEvents(anthropicEventsFromGemini(upstream, candidate.model.id), res);
      return;
    }
    // tryCandidate only returns ok for JSON that parsed and has its wire's shape.
    const json = result.json as Json;
    if (clientApi === 'openai' && candidate.provider.wire === 'openai') {
      sendRaw(res, 200, 'application/json', result.rawText);
      return;
    }
    if (clientApi === 'anthropic' && candidate.provider.wire === 'anthropic') {
      sendRaw(res, 200, 'application/json', result.rawText);
      return;
    }
    sendJson(res, 200, translateJson(clientApi, candidate.provider.wire, json, candidate.model.id));
    return;
  }
  return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'every free model that meets this request refused it');
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await answerRequest('openai', body, res);
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await answerRequest('anthropic', body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
