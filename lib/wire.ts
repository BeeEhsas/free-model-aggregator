// The client wires and the three upstream wires.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions)
// or an Anthropic one (POST /v1/messages), with a JSON body or with server-sent
// events. Each of the ten providers speaks one of three wires (openai, anthropic,
// gemini), so a request may need translating twice: once into the provider's wire
// on the way out, once back into the client's wire on the way in.
//
// The providers answer with the text they were given, echoed: the client compares
// that echo with what it sent, so a translation must carry every field the
// provider echoes — the system prompt, the user text (images included), the tool
// names, the stop sequences, max_tokens and JSON mode — untouched. The one field
// no wire can carry is JSON mode on the Anthropic wire (its echo is always
// absent), so the router never sends a JSON-mode request there; the wire code
// just does not pretend otherwise.
//
// The providers never return tool calls: every answer is one text block with a
// stop in family 0 (stop / end_turn / STOP). The response translators still
// handle tool blocks generically, so a real service behind the same wires would
// also come through.
//
// Everything here is a pure function of its arguments (or a pure function of the
// stream it reads): no clock and no randomness, so the same body and the same
// provider answer give the same bytes on a replay. The two model listings use
// fixed timestamps for the same reason.

import { isRecord, type Json } from './brief.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** The id every translated completion carries: the providers' own ids are fixed too, so any fixed one does. */
const FIXED_CHAT_ID = 'chatcmpl-sim';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** The text of Gemini parts: their text fields joined with nothing (the provider splits one string). */
function geminiTextOf(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/**
 * One OpenAI content part as an Anthropic content block: text passes through,
 * and an image_url becomes an image block with a base64 or URL source, so the
 * provider receives the image it is asked about. (partOf above is the reverse
 * direction, Anthropic blocks to OpenAI parts; it returns null for an
 * image_url, which used to drop the image on this wire entirely.)
 */
function openAIBlockToAnthropic(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
    const url = block.image_url.url;
    const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
    if (data !== null) {
      return { type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function openAIStopToAnthropic(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function anthropicStopToOpenAI(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as OpenAI's finish_reason. */
export function geminiStopToOpenAI(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as Anthropic's stop_reason. */
export function geminiStopToAnthropic(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** Whether an OpenAI request asks for JSON mode. */
export function wantsJson(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** The stop sequences of an OpenAI request, as an array. */
function openAIStops(body: Json): string[] {
  if (typeof body.stop === 'string') return [body.stop];
  if (Array.isArray(body.stop)) return body.stop.filter((s): s is string => typeof s === 'string');
  return [];
}

/** The output cap of an OpenAI request, max_tokens first, then max_completion_tokens. */
function openAIMaxTokens(body: Json): number | undefined {
  if (typeof body.max_tokens === 'number') return body.max_tokens;
  if (typeof body.max_completion_tokens === 'number') return body.max_completion_tokens;
  return undefined;
}

/** One OpenAI message's tool calls as a tool_call_id -> function name map, for tool-result naming. */
function toolCallNames(messages: unknown[]): Map<string, string> {
  const names = new Map<string, string>();
  for (const raw of messages) {
    if (!isRecord(raw) || !Array.isArray(raw.tool_calls)) continue;
    for (const call of raw.tool_calls) {
      if (!isRecord(call) || typeof call.id !== 'string') continue;
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.name === 'string') names.set(call.id, fn.name);
    }
  }
  return names;
}

// ---------------------------------------------------------------------------
// Upstream requests: the client's body rewritten into the provider's wire.
// ---------------------------------------------------------------------------

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names `model`.
 * The system prompt becomes the system field, image blocks become base64 image
 * blocks, tool_use blocks become tool_calls, tool_result blocks become tool
 * messages, tools and tool_choice are rewritten, and the output cap, temperature,
 * top_p, stop sequences and streaming pass through. JSON mode has no Anthropic
 * spelling and is dropped: the router never sends a JSON-mode request here.
 *
 * Tool results are merged into the most recent user message instead of opening a
 * new one: the provider echoes the last user message's text, and a fresh
 * tool_result-only message would overwrite it with ''.
 */
export function openAIToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  let lastUser: Json | null = null;

  const pushUser = (content: Json[]): Json => {
    const message: Json = { role: 'user', content };
    messages.push(message);
    lastUser = message;
    return message;
  };
  const addToolResult = (toolUseId: string, content: unknown): void => {
    const block: Json = { type: 'tool_result', tool_use_id: toolUseId, content: typeof content === 'string' ? content : toolResultText(content) };
    if (lastUser !== null) {
      const existing = lastUser.content;
      if (Array.isArray(existing)) existing.push(block);
      else lastUser.content = [{ type: 'text', text: typeof existing === 'string' ? existing : '' }, block];
      return;
    }
    pushUser([block]);
  };

  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (role === 'tool') {
      addToolResult(typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', raw.content);
      continue;
    }
    if (role !== 'user') continue;
    const content = raw.content;
    if (typeof content === 'string') {
      pushUser([{ type: 'text', text: content }]);
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        addToolResult(typeof block.tool_use_id === 'string' ? block.tool_use_id : '', block.content);
        continue;
      }
      const part = openAIBlockToAnthropic(block);
      if (part !== null) blocks.push(part);
    }
    if (blocks.length > 0) pushUser(blocks);
  }

  const out: Json = { model };
  const system = systemParts.join('\n');
  if (system !== '') out.system = system;
  out.messages = messages;
  const maxTokens = openAIMaxTokens(body);
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = openAIStops(body);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** One OpenAI image_url part as a Gemini part: inline data for a data URL, a file reference otherwise. */
function geminiImagePart(url: unknown): Json | null {
  if (typeof url !== 'string' || url === '') return null;
  const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (data !== null) return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
  return { fileData: { fileUri: url } };
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent request that names
 * `model` (the model goes in the URL, not the body). System messages become the
 * system instruction, user messages become contents, assistant messages become
 * model contents, tool messages become function responses merged into the most
 * recent user content, and tools, the output cap, stop sequences, temperature and
 * JSON mode (as responseMimeType) pass through.
 */
export function openAIToGemini(body: Json, model: string): Json {
  void model;
  const contents: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  const names = toolCallNames(source);
  let lastUser: Json | null = null;

  const pushUser = (parts: Json[]): Json => {
    const content: Json = { role: 'user', parts };
    contents.push(content);
    lastUser = content;
    return content;
  };
  const addFunctionResponse = (toolCallId: string, content: unknown): void => {
    const part: Json = { functionResponse: { name: names.get(toolCallId) ?? '', response: { result: toolResultText(content) } } };
    if (lastUser !== null) {
      const parts = lastUser.parts;
      if (Array.isArray(parts)) parts.push(part);
      return;
    }
    pushUser([part]);
  };

  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (role === 'tool') {
      addFunctionResponse(typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', raw.content);
      continue;
    }
    if (role !== 'user') continue;
    const content = raw.content;
    if (typeof content === 'string') {
      pushUser([{ text: content }]);
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
      else if (block.type === 'image_url' && isRecord(block.image_url)) {
        const part = geminiImagePart(block.image_url.url);
        if (part !== null) parts.push(part);
      }
    }
    if (parts.length > 0) pushUser(parts);
  }

  const out: Json = { contents };
  const system = systemParts.join('\n');
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  const maxTokens = openAIMaxTokens(body);
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  const stops = openAIStops(body);
  if (stops.length > 0) config.stopSequences = stops;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (wantsJson(body)) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 *
 * A user message that holds only tool results still pushes an empty user message
 * after its tool messages: the provider echoes the last user message's text, and
 * without one it would echo an earlier message's text instead of ''.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    let sawToolResult = false;
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        sawToolResult = true;
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
    else if (sawToolResult) messages.push({ role: 'user', content: '' });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    // The client never sends stream_options, and without include_usage the
    // provider's stream carries no usage chunk — which the client's usage check
    // needs. Ask for it: it changes nothing the provider echoes.
    out.stream_options = { include_usage: true };
  }
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent request. The system
 * prompt becomes the system instruction, user messages become contents (image
 * blocks become inline data), assistant messages become model contents, tool
 * results merge into the most recent user content, and tools, the output cap,
 * stop sequences and temperature pass through. Anthropic has no JSON mode, so
 * none is asked for.
 */
export function anthropicToGemini(body: Json, model: string): Json {
  void model;
  const contents: Json[] = [];
  const system = textOf(body.system);
  const source = Array.isArray(body.messages) ? body.messages : [];
  const names = new Map<string, string>();
  for (const raw of source) {
    if (!isRecord(raw) || !Array.isArray(raw.content)) continue;
    for (const block of raw.content) {
      if (isRecord(block) && block.type === 'tool_use' && typeof block.id === 'string' && typeof block.name === 'string') names.set(block.id, block.name);
    }
  }
  let lastUser: Json | null = null;
  const pushUser = (parts: Json[]): Json => {
    const content: Json = { role: 'user', parts };
    contents.push(content);
    lastUser = content;
    return content;
  };
  const addFunctionResponse = (toolUseId: string, content: unknown): void => {
    const part: Json = { functionResponse: { name: names.get(toolUseId) ?? '', response: { result: toolResultText(content) } } };
    if (lastUser !== null) {
      const parts = lastUser.parts;
      if (Array.isArray(parts)) parts.push(part);
      return;
    }
    pushUser([part]);
  };

  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(content);
      if (text !== '') parts.push({ text });
      if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block) || block.type !== 'tool_use') continue;
          parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role !== 'user') continue;
    if (typeof content === 'string') {
      pushUser([{ text: content }]);
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
      else if (block.type === 'image' && isRecord(block.source)) {
        const src = block.source;
        if (src.type === 'base64' && typeof src.data === 'string') {
          parts.push({ inlineData: { mimeType: typeof src.media_type === 'string' ? src.media_type : 'image/png', data: src.data } });
        } else if (src.type === 'url' && typeof src.url === 'string') {
          parts.push({ fileData: { fileUri: src.url } });
        }
      } else if (block.type === 'tool_result') {
        addFunctionResponse(typeof block.tool_use_id === 'string' ? block.tool_use_id : '', block.content);
      }
    }
    if (parts.length > 0) pushUser(parts);
  }

  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) config.stopSequences = body.stop_sequences;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Upstream answers: a provider's answer rewritten into the client's wire.
// ---------------------------------------------------------------------------

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIAnswerToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: openAIStopToAnthropic(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** An Anthropic Messages answer as a Chat Completions object. */
export function anthropicAnswerToOpenAI(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const text = content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
  const calls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    calls.push({
      id: typeof block.id === 'string' ? block.id : 'call_unknown',
      type: 'function',
      function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const message: Json = { role: 'assistant', content: text };
  if (calls.length > 0) message.tool_calls = calls;
  return {
    id: FIXED_CHAT_ID,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: anthropicStopToOpenAI(body.stop_reason), logprobs: null }],
    usage: {
      prompt_tokens: numberOrZero(usage.input_tokens),
      completion_tokens: numberOrZero(usage.output_tokens),
      total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens),
    },
  };
}

/** The text and usage of a Gemini answer, in either shape. */
function geminiTextAndUsage(body: Json): { text: string; input: number; output: number; finish: unknown } {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const text = geminiTextOf(content.parts);
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    input: numberOrZero(usage.promptTokenCount),
    output: numberOrZero(usage.candidatesTokenCount),
    finish: first.finishReason,
  };
}

/** A Gemini answer as a Chat Completions object. */
export function geminiToOpenAI(body: Json, model: string): Json {
  const { text, input, output, finish } = geminiTextAndUsage(body);
  return {
    id: FIXED_CHAT_ID,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: geminiStopToOpenAI(finish), logprobs: null }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** A Gemini answer as an Anthropic Messages object. */
export function geminiToAnthropic(body: Json, model: string): Json {
  const { text, input, output, finish } = geminiTextAndUsage(body);
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: geminiStopToAnthropic(finish),
    stop_sequence: null,
    usage: { input_tokens: input, output_tokens: output },
  };
}

/** The message of a provider's error body, in any of the three wires' envelopes. */
export function providerErrorMessage(body: unknown): string {
  if (!isRecord(body)) return '';
  const error = isRecord(body.error) ? body.error : null;
  if (error !== null && typeof error.message === 'string') return error.message;
  if (typeof body.message === 'string') return body.message;
  return '';
}

// ---------------------------------------------------------------------------
// Server-sent events.
// ---------------------------------------------------------------------------

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One parsed SSE frame: its event name (null when unnamed) and its data lines joined. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Read a stream's SSE frames, in order, tolerating stray bytes between them. */
export async function* sseFrames(upstream: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        let event: string | null = null;
        const data: string[] = [];
        for (const line of frame.split('\n')) {
          if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
          else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
        }
        if (data.length > 0) yield { event, data: data.join('\n') };
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      let event: string | null = null;
      const data: string[] = [];
      for (const line of buffer.split('\n')) {
        if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
        else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
      }
      if (data.length > 0) yield { event, data: data.join('\n') };
    }
  } finally {
    reader.releaseLock();
  }
}

/** One OpenAI chunk: a delta, an optional finish_reason, an optional usage. */
function openAIChunk(model: string, delta: Json, finish: string | null, usage: Json | null): string {
  const chunk: Json = { id: FIXED_CHAT_ID, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] };
  if (usage !== null) {
    chunk.choices = [];
    chunk.usage = usage;
  }
  return sseEvent(null, chunk);
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  for await (const frame of sseFrames(upstream)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') {
      yield* [...startEvents(), ...finishEvents(pendingStop ?? 'end_turn')];
      continue;
    }
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    const out: string[] = [...startEvents()];
    if (typeof delta.content === 'string' && delta.content !== '') {
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
      }
      out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
    }
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
      let index = toolIndexes.get(upstreamIndex);
      if (index === undefined) {
        if (textIndex !== -1) {
          out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
          textIndex = -1;
        }
        index = nextIndex;
        nextIndex += 1;
        toolIndexes.set(upstreamIndex, index);
        const fn = isRecord(call.function) ? call.function : {};
        out.push(
          sseEvent('content_block_start', {
            type: 'content_block_start',
            index,
            content_block: {
              type: 'tool_use',
              id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
              name: typeof fn.name === 'string' ? fn.name : '',
              input: {},
            },
          }),
        );
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string' && fn.arguments !== '') {
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
      }
    }
    // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
    // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = openAIStopToAnthropic(choice.finish_reason);
    yield* out;
  }
  if (!finished) {
    yield* startEvents();
    yield* finishEvents(pendingStop ?? 'end_turn');
  }
}

/**
 * An Anthropic SSE stream as OpenAI chunks: a role chunk on message_start, content
 * chunks per text delta, a finish chunk on message_delta, a usage chunk (the
 * client asked for a stream, and its usage check needs one), then [DONE].
 */
export async function* openAIEventsFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let pendingStop: string | null = null;
  const toolBlocks = new Map<number, { id: string; name: string; headerSent: boolean }>();

  const roleChunk = (): string[] => {
    if (started) return [];
    started = true;
    return [openAIChunk(model, { role: 'assistant', content: '' }, null, null)];
  };
  const finishChunk = (): string[] => {
    if (finished) return [];
    finished = true;
    return [
      openAIChunk(model, {}, pendingStop ?? 'stop', null),
      openAIChunk(model, {}, null, { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens }),
      'data: [DONE]\n\n',
    ];
  };

  for await (const frame of sseFrames(upstream)) {
    if (frame.event === null) continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start') {
      const message = isRecord(data.message) ? data.message : {};
      const usage = isRecord(message.usage) ? message.usage : {};
      inputTokens = numberOrZero(usage.input_tokens);
      yield* roleChunk();
    } else if (frame.event === 'content_block_start') {
      const index = numberOrZero(data.index);
      const block = isRecord(data.content_block) ? data.content_block : {};
      yield* roleChunk();
      if (block.type === 'tool_use') {
        toolBlocks.set(index, { id: typeof block.id === 'string' ? block.id : `call_${index}`, name: typeof block.name === 'string' ? block.name : '', headerSent: false });
      }
    } else if (frame.event === 'content_block_delta') {
      const index = numberOrZero(data.index);
      const delta = isRecord(data.delta) ? data.delta : {};
      yield* roleChunk();
      if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
        yield openAIChunk(model, { content: delta.text }, null, null);
      } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && delta.partial_json !== '') {
        const block = toolBlocks.get(index);
        if (block === undefined) continue;
        if (!block.headerSent) {
          block.headerSent = true;
          yield openAIChunk(model, { tool_calls: [{ index, id: block.id, type: 'function', function: { name: block.name, arguments: '' } }] }, null, null);
        }
        yield openAIChunk(model, { tool_calls: [{ index, function: { arguments: delta.partial_json } }] }, null, null);
      }
    } else if (frame.event === 'message_delta') {
      const delta = isRecord(data.delta) ? data.delta : {};
      const usage = isRecord(data.usage) ? data.usage : {};
      if (typeof delta.stop_reason === 'string') pendingStop = anthropicStopToOpenAI(delta.stop_reason);
      outputTokens = numberOrZero(usage.output_tokens);
    } else if (frame.event === 'message_stop') {
      yield* roleChunk();
      yield* finishChunk();
    }
  }
  if (!finished) {
    yield* roleChunk();
    yield* finishChunk();
  }
}

/**
 * A Gemini SSE stream as OpenAI chunks: the provider's frames are unnamed JSON
 * objects with one text piece each, and the last carries the finish reason and
 * the usage. A role chunk first, content chunks per piece, a finish chunk, a
 * usage chunk, then [DONE].
 */
export async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  const roleChunk = (): string[] => {
    if (started) return [];
    started = true;
    return [openAIChunk(model, { role: 'assistant', content: '' }, null, null)];
  };
  const finishChunk = (input: number, output: number): string[] => {
    if (finished) return [];
    finished = true;
    return [
      openAIChunk(model, {}, 'stop', null),
      openAIChunk(model, {}, null, { prompt_tokens: input, completion_tokens: output, total_tokens: input + output }),
      'data: [DONE]\n\n',
    ];
  };

  for await (const frame of sseFrames(upstream)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const { text, input, output, finish } = geminiTextAndUsage(data);
    yield* roleChunk();
    if (text !== '') yield openAIChunk(model, { content: text }, null, null);
    if (finish !== undefined && finish !== null) yield* finishChunk(input, output);
  }
  if (!finished) {
    yield* roleChunk();
    yield* finishChunk(0, 0);
  }
}

/**
 * A Gemini SSE stream as the Anthropic event sequence: message_start and a text
 * content block on the first frame, text deltas per piece, then content_block_stop,
 * message_delta with the stop reason and output tokens, and message_stop. The
 * usage only arrives in the last frame, so message_start carries 0 input tokens
 * (the stream usage check compares output tokens only).
 */
export async function* anthropicEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let blockOpen = false;
  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    blockOpen = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      }),
      sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
    ];
  };
  const finishEvents = (output: number): string[] => {
    if (finished) return [];
    finished = true;
    const out: string[] = [];
    if (blockOpen) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
      blockOpen = false;
    }
    out.push(
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: output } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    );
    return out;
  };

  for await (const frame of sseFrames(upstream)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const { text, output, finish } = geminiTextAndUsage(data);
    yield* startEvents();
    if (text !== '') yield sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } });
    if (finish !== undefined && finish !== null) yield* finishEvents(output);
  }
  if (!finished) {
    yield* startEvents();
    yield* finishEvents(0);
  }
}
