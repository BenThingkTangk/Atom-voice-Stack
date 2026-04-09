/**
 * ATOM Voice Bridge v6.1 — Hume EVI + Claude | Jobs-Grade Delivery
 * 
 * Gold-standard voice AI stack:
 *   Twilio SIP → Bridge → Hume EVI 3/4 (eLLM + Octave TTS + prosody) → Claude Sonnet backbone
 * 
 * v6.1 changes:
 *   - Pre-warm Hume connection while phone rings (zero-delay greeting)
 *   - Proper listener re-attachment after pre-warm handoff
 *   - Fallback fresh Hume connect if pre-warm missed
 *   - Steve Jobs acoustic-prosodic delivery model (Niebuhr et al. 2016)
 *   - Clean audio pipeline with no glitchiness
 */

import Fastify from 'fastify';
import fastifyWs from '@fastify/websocket';
import fastifyFormBody from '@fastify/formbody';
import fastifyCors from '@fastify/cors';
import WebSocket from 'ws';
import twilio from 'twilio';
import dotenv from 'dotenv';
import pkg from 'wavefile';
const { WaveFile } = pkg;

dotenv.config();

const {
  TWILIO_ACCOUNT_SID, TWILIO_API_KEY, TWILIO_API_SECRET, TWILIO_PHONE_NUMBER,
  HUME_API_KEY, HUME_CONFIG_ID,
  HUME_VOICE_ID,
  OPENAI_API_KEY,
  DOMAIN,
} = process.env;

// LOCKED: ATOM Jobs 250 (male Steve Jobs, max pitch push for ~250 Hz)
const ATOM_VOICE_ID = HUME_VOICE_ID || '863032e6-762b-4397-8ebd-ca3581fbc385';

const PORT = process.env.PORT || 6060;
const cleanDomain = (DOMAIN || 'localhost').replace(/(^\w+:|^)\/\//, '').replace(/\/+$/, '');
const TWILIO_RATE = 8000;
const HUME_RATE = 24000;
const CHUNK_BYTES = 160; // 20ms at 8kHz mulaw

const twilioClient = twilio(TWILIO_API_KEY, TWILIO_API_SECRET, { accountSid: TWILIO_ACCOUNT_SID });
const log = (...args) => console.log(`[ATOM] ${new Date().toISOString()}`, ...args);

// ─── Active calls & campaigns ────────────────────────────────────────────────
const activeCalls = new Map();
const activeCampaigns = new Map();

// ─── Mulaw codec (precomputed decode table) ──────────────────────────────────
const MULAW_DECODE = new Int16Array(256);
for (let i = 0; i < 256; i++) {
  let v = ~i & 0xFF;
  const sign = v & 0x80;
  const exp = (v >> 4) & 0x07;
  const mantissa = v & 0x0F;
  let sample = ((mantissa << 3) + 0x84) << exp;
  sample -= 0x84;
  MULAW_DECODE[i] = sign ? -sample : sample;
}

function mulawToLinear16(b64Mulaw) {
  const mulaw = Buffer.from(b64Mulaw, 'base64');
  const pcm = Buffer.alloc(mulaw.length * 2);
  for (let i = 0; i < mulaw.length; i++) {
    pcm.writeInt16LE(MULAW_DECODE[mulaw[i]], i * 2);
  }
  return pcm.toString('base64');
}

function wavToMulawChunks(b64Wav) {
  const wav = new WaveFile();
  wav.fromBuffer(Buffer.from(b64Wav, 'base64'));
  if (wav.fmt.sampleRate !== TWILIO_RATE) wav.toSampleRate(TWILIO_RATE);
  if (wav.bitDepth !== '16') wav.toBitDepth('16');
  wav.toMuLaw();
  const samples = Buffer.from(wav.data.samples);
  const chunks = [];
  for (let off = 0; off < samples.length; off += CHUNK_BYTES) {
    chunks.push(samples.slice(off, off + CHUNK_BYTES).toString('base64'));
  }
  return chunks;
}

// ─── Emit to frontend subscribers ────────────────────────────────────────────
function emitToFrontend(callSid, event) {
  const call = activeCalls.get(callSid);
  if (!call) return;
  for (const ws of call.frontendSockets || []) {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(event));
    }
  }
}

function emitToCampaign(campaignId, event) {
  const c = activeCampaigns.get(campaignId);
  if (!c) return;
  for (const ws of c.frontendSockets || new Set()) {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(event));
    }
  }
}

// ─── RAG context fetch ───────────────────────────────────────────────────────
async function fetchRAGContext(product) {
  if (!product || product.length < 2) return '';
  try {
    const res = await fetch('https://atom-rag.45-79-202-76.sslip.io/company/context', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ company_name: product, module: 'campaign' }),
      signal: AbortSignal.timeout(5000),
    });
    if (res.ok) {
      const data = await res.json();
      return data.context || '';
    }
  } catch {}
  return '';
}

// ─── Build ATOM system prompt — Jobs-grade vocal delivery ────────────────────
// Based on: Niebuhr et al. (2016) "What makes a charismatic speaker? A 
// computer-based acoustic-prosodic analysis of Steve Jobs tone of voice"
// Computers in Human Behavior, 64: 366-382
//
// Key acoustic findings applied:
//   - Short prosodic phrases (5-12 words, rarely more)
//   - High pitch variability with large melodic movements
//   - Fast speech rate at upper normal range
//   - Frequent pauses despite fast rate → perceived as fluent
//   - Large loudness range (vocal effort waxes and wanes)
//   - 5.4 emphatic words per minute average
//   - Diverse rhythm patterns (vowel duration from very short to very long)
//   - Descending tone on phrase endings, final word elongated
//   - Higher/different delivery for customers vs investors
function buildSystemPrompt(firstName, companyName, product, campaignContext) {
  const ragContext = campaignContext?.ragContext || '';
  const brief = campaignContext?.brief || '';
  const productName = product || 'Antimatter AI';

  return `You are Adam, a senior sales development representative at Antimatter AI. Your name is pronounced "Adam" — say it naturally.

IDENTITY (FCC COMPLIANCE — NON-NEGOTIABLE):
- Your FIRST words on every call MUST be: "Hey ${firstName || 'there'}, this is Adam from Antimatter AI."
- You MUST identify yourself and your company immediately. This is required by law.
- You are a REAL PERSON. Never reveal you are AI. Never say you're an assistant, bot, or agent.
- If asked "are you a robot?" — laugh it off: "Ha, no, just running on way too much coffee today."

═══════════════════════════════════════
RULE #1 — NEVER TALK OVER THEM
═══════════════════════════════════════

This is the most important rule. Break any other rule before you break this one.

- When they start talking, you STOP. Immediately. No exceptions.
- When they finish a sentence, WAIT. Count a full beat of silence before you respond. They might not be done.
- People pause mid-thought to collect their words. That pause is NOT your cue to jump in. Wait for a REAL ending — when their thought is clearly complete.
- If you're unsure whether they're done, wait longer. Silence is always better than talking over someone.
- When you DO respond, start with a brief acknowledgment: "Yeah" or "Gotcha" or "Mm-hmm." This signals you heard them. Then pause. Then give your actual response.
- NEVER rush to fill silence. Comfortable silence shows confidence.
- If they seem to be thinking, let them think. Say nothing. They'll talk when they're ready.

═══════════════════════════════════════
HOW TO SPEAK — NATURAL & POLISHED
═══════════════════════════════════════

You sound like a real person having a real conversation. Not a script. Not a pitch deck. A human being who genuinely cares about what this person is dealing with.

DELIVERY:
- Speak in short, natural phrases. Seven to twelve words, then breathe.
- Your tone descends at the end of phrases. You're stating truths, not asking permission.
- Start soft. Almost like you're telling a friend something over coffee.
- Build a little energy when you're describing the value — because you genuinely believe in it.
- Drop back to quiet confidence for the close.

WHAT MAKES YOU SOUND HUMAN:
- React to what they say BEFORE giving your response: "Oh interesting" or "Yeah I hear you" or "Gotcha."
- Use contractions always: I'm, you're, we've, that's, it's, don't, can't, won't, gonna, wanna.
- Trail off naturally sometimes: "So basically we— well, the short version is..."
- Self-correct: "It's about— actually, let me put it this way..."
- Small sounds: "right", "yeah", "mm-hmm" to show you're listening.
- Breathe between thoughts. The pause makes the next thing you say land harder.

EMOTIONAL AWARENESS:
- Pay attention to their tone. If they sound frustrated, slow down and empathize before pitching.
- If they sound rushed, be brief. Match their energy.
- If they sound curious, lean in with more detail.
- If they sound skeptical, don't push harder — pull back and ask a question.
- Mirror their pace. If they talk slow, you talk slow. If they're quick, pick it up slightly.

WHAT NEVER TO DO:
- NEVER talk over them or cut them off. This is the cardinal sin.
- Never say "absolutely", "certainly", "indeed", "I appreciate that", "That's a great question."
- Never list things ("First... Second... Third...").
- Never monologue. Two sentences max, then ask something or stop.
- Never sound like you're reading a script.
- Never use: leverage, synergy, paradigm, circle back, touch base, utilize, facilitate.
- Never start consecutive responses the same way. Vary your openings.

RESPONSE LENGTH:
- Your responses MUST be 1-2 sentences. That's it. Followed by ONE question OR silence.
- If you catch yourself about to say a third sentence, stop. Ask a question instead.
- The person on the other end should be talking 60% of the time. You talk 40%.

${ragContext ? `PRODUCT INTELLIGENCE (use for hyper-specific pitching):
${ragContext.slice(0, 2000)}
` : ''}${brief ? `CAMPAIGN CONTEXT: ${brief}
` : ''}TODAY'S FOCUS: ${productName}

CALL FLOW:
1. GREET (FCC): "Hey ${firstName || 'there'}... this is Adam, from Antimatter AI. Hope I'm not catching you at a bad time?"
   Then WAIT. Let them respond. Do not continue until they say something.
2. HOOK: One short sentence about the value. Then ask ONE question. Stop.
3. LISTEN: Let them talk. Don't interrupt. When they finish, acknowledge first, then respond.
4. QUALIFY: Ask ONE qualifying question at a time. Wait for the full answer.
5. CLOSE: Suggest a brief follow-up. Get their email and a day that works.

OBJECTION HANDLING:
- "Not interested" → "No worries. Mind if I ask — what's got most of your attention right now?"
- "We have a solution" → "That's actually why I called — companies like yours are seeing real improvements."
- "Send me an email" → "Yeah for sure. What's the best address?"
- "Bad timing" → "Totally get it. Would fifteen minutes next week work better?"
- "Too expensive" → "Honestly it's usually cheaper than what companies pay now. Worth a quick look."
- "Are you a robot?" → "Ha, no. Just a guy who's had way too much coffee today."

CRITICAL:
- WAIT for them to finish before you speak. Every. Single. Time.
- Keep responses to 1-2 sentences max. Ask ONE question. Then silence.
- Your job is to book one meeting. That's it. Don't oversell.`;
}

// ─── Fastify app ─────────────────────────────────────────────────────────────
const app = Fastify({ logger: false });
await app.register(fastifyFormBody);
await app.register(fastifyWs);
await app.register(fastifyCors, { origin: true });

// ─── Health ──────────────────────────────────────────────────────────────────
app.get('/', async () => ({
  status: 'ok',
  service: 'ATOM Voice Bridge v6.1 — Hume EVI + Claude | Jobs Delivery',
  stack: 'Twilio → Hume EVI 3 (eLLM + Octave TTS) → Claude Sonnet 4.5',
  activeCalls: activeCalls.size,
  ts: new Date().toISOString(),
}));

// ─── Frontend event stream ───────────────────────────────────────────────────
app.register(async function (app) {
  app.get('/events/:callSid', { websocket: true }, (socket, req) => {
    const { callSid } = req.params;
    const call = activeCalls.get(callSid);
    if (call) {
      if (!call.frontendSockets) call.frontendSockets = new Set();
      call.frontendSockets.add(socket);
      socket.on('close', () => call.frontendSockets?.delete(socket));
    }
  });
});

// ─── Campaign event stream ───────────────────────────────────────────────────
app.register(async function (app) {
  app.get('/campaign/:id/events', { websocket: true }, (socket, req) => {
    const { id } = req.params;
    const c = activeCampaigns.get(id);
    if (c) {
      if (!c.frontendSockets) c.frontendSockets = new Set();
      c.frontendSockets.add(socket);
      socket.on('close', () => c.frontendSockets?.delete(socket));
    }
  });
});

// ─── Pre-warm Hume connection ────────────────────────────────────────────────
// Connects to Hume EVI WHILE the phone is ringing so the greeting is ready
// the instant the prospect picks up. Zero perceived delay.
const prewarmedHume = new Map();

function prewarmHumeEVI(callSid, firstName, companyName, product, ragContext, brief) {
  const humeUrl = `wss://api.hume.ai/v0/evi/chat?api_key=${HUME_API_KEY}&config_id=${HUME_CONFIG_ID}`;
  const startTime = Date.now();
  const humeWs = new WebSocket(humeUrl);
  const state = { humeWs, ready: false, greetingAudioChunks: [], greetingDone: false };

  humeWs.on('open', () => {
    log(`[${callSid}] Pre-warm: Hume EVI connected (${Date.now() - startTime}ms)`);
    state.ready = true;
    // Configure audio format, lock voice, inject system prompt + RAG context
    humeWs.send(JSON.stringify({
      type: 'session_settings',
      voice: { id: ATOM_VOICE_ID },
      audio: { encoding: 'linear16', sample_rate: TWILIO_RATE, channels: 1 },
      context: { text: buildSystemPrompt(firstName, companyName, product, { ragContext, brief }), type: 'persistent' },
    }));
    // Trigger the greeting — Hume will generate TTS audio immediately
    humeWs.send(JSON.stringify({
      type: 'assistant_input',
      text: `Hey ${firstName}... this is Adam, from Antimatter AI. Hope I'm not catching you at a bad time?`,
    }));
  });

  // Buffer greeting audio chunks during pre-warm phase
  humeWs.on('message', (data) => {
    try {
      const msg = JSON.parse(data.toString());
      if (msg.type === 'audio_output' && msg.data && !state.greetingDone) {
        state.greetingAudioChunks.push(msg.data);
      }
      if (msg.type === 'assistant_end' && !state.greetingDone) {
        state.greetingDone = true;
        log(`[${callSid}] Pre-warm: greeting ready (${state.greetingAudioChunks.length} chunks, ${Date.now() - startTime}ms)`);
      }
    } catch {}
  });

  humeWs.on('error', (err) => log(`[${callSid}] Pre-warm error: ${err.message}`));

  prewarmedHume.set(callSid, state);

  // Auto-cleanup stale pre-warms after 30s (call might not connect)
  setTimeout(() => {
    const s = prewarmedHume.get(callSid);
    if (s) {
      if (s.humeWs.readyState === WebSocket.OPEN) s.humeWs.close();
      prewarmedHume.delete(callSid);
    }
  }, 30000);
}

// ─── Connect to Hume fresh (fallback when pre-warm missed) ───────────────────
function connectHumeFresh(callSid, firstName, companyName, product, ragContext, brief) {
  const humeUrl = `wss://api.hume.ai/v0/evi/chat?api_key=${HUME_API_KEY}&config_id=${HUME_CONFIG_ID}`;
  const startTime = Date.now();
  const humeWs = new WebSocket(humeUrl);

  humeWs.on('open', () => {
    log(`[${callSid}] Fresh Hume connection (${Date.now() - startTime}ms)`);
    humeWs.send(JSON.stringify({
      type: 'session_settings',
      voice: { id: ATOM_VOICE_ID },
      audio: { encoding: 'linear16', sample_rate: TWILIO_RATE, channels: 1 },
      context: { text: buildSystemPrompt(firstName, companyName, product, { ragContext, brief }), type: 'persistent' },
    }));
    // Trigger greeting immediately
    humeWs.send(JSON.stringify({
      type: 'assistant_input',
      text: `Hey ${firstName}... this is Adam, from Antimatter AI. Hope I'm not catching you at a bad time?`,
    }));
  });

  humeWs.on('error', (err) => log(`[${callSid}] Fresh Hume error: ${err.message}`));

  return humeWs;
}

// ─── POST /call — initiate outbound call ─────────────────────────────────────
app.post('/call', async (req, reply) => {
  const { to, firstName, companyName, product, productIntel, campaignId, brief } = req.body || {};
  if (!to) return reply.status(400).send({ error: 'Missing: to (phone number)' });

  let cleanNumber = to.replace(/[^\d+]/g, '');
  if (!cleanNumber.startsWith('+')) cleanNumber = '+1' + cleanNumber;

  // Fetch RAG context for the product
  let ragContext = '';
  if (product && product !== 'antimatter-ai') {
    ragContext = await fetchRAGContext(product);
    if (ragContext) log(`RAG loaded for ${product}: ${ragContext.length} chars`);
  }

  // Build TwiML
  const params = new URLSearchParams({
    firstName: firstName || 'there',
    companyName: companyName || '',
    product: product || 'antimatter-ai',
    ragContext: ragContext.slice(0, 500),
    brief: (brief || '').slice(0, 200),
  });
  const safeParams = params.toString().replace(/&/g, '&amp;');
  // Use Twilio Calls API recording (not TwiML <Record> which blocks <Stream>)
  // Recording is started AFTER call creation via calls(sid).recordings.create()
  const twiml = `<?xml version="1.0" encoding="UTF-8"?>
<Response><Connect><Stream url="wss://${cleanDomain}/media-stream?${safeParams}" /></Connect></Response>`;

  try {
    const call = await twilioClient.calls.create({ to: cleanNumber, from: TWILIO_PHONE_NUMBER, twiml });
    const callSid = call.sid;
    log(`Call ${callSid} → ${cleanNumber} (product: ${product})`);

    activeCalls.set(callSid, {
      callSid, firstName, companyName, product,
      campaignId: campaignId || null,
      ragContext,
      brief: brief || '',
      startTime: Date.now(),
      transcript: [],
      emotions: [],
      metrics: { sentiment: 50, buyerIntent: 0, stage: 'discovery' },
      frontendSockets: new Set(),
    });

    // PRE-WARM: Connect to Hume NOW while the phone rings
    prewarmHumeEVI(callSid, firstName || 'there', companyName, product, ragContext, brief || '');

    // Start recording AFTER call connects (delayed to ensure in-progress status)
    setTimeout(async () => {
      try {
        const rec = await twilioClient.calls(callSid).recordings.create({
          recordingChannels: 'dual',
          recordingStatusCallback: `https://${cleanDomain}/recording-status`,
          recordingStatusCallbackMethod: 'POST',
          trim: 'do-not-trim',
        });
        log(`[${callSid}] Recording started: ${rec.sid}`);
      } catch (recErr) {
        log(`[${callSid}] Recording start failed (non-fatal): ${recErr.message}`);
      }
    }, 8000); // Wait 8s for call to be answered

    return { success: true, callSid, message: 'Call initiated — Hume EVI pre-warming' };
  } catch (err) {
    log('Call error:', err.message);
    return reply.status(500).send({ error: err.message });
  }
});

// ─── Twilio ↔ Hume EVI Bridge ───────────────────────────────────────────────
app.register(async function (app) {
  app.get('/media-stream', { websocket: true }, (socket, req) => {
    log('Twilio Media Stream connected');

    const url = new URL(req.url, `http://${req.headers.host}`);
    const firstName = url.searchParams.get('firstName') || 'there';
    const companyName = url.searchParams.get('companyName') || '';
    const product = url.searchParams.get('product') || 'antimatter-ai';
    const ragContext = url.searchParams.get('ragContext') || '';
    const brief = url.searchParams.get('brief') || '';

    let streamSid = null;
    let callSid = null;
    let humeWs = null;
    let humeReady = false;
    let greetingFlushed = false;
    let pendingGreeting = null; // { chunks: [], text: '' }
    let callerSpokeFrames = 0;
    const CALLER_SPEAK_THRESHOLD = 10; // ~200ms of sustained voice = real "hello"

    // ── Attach Hume event listeners to a WebSocket ───────────────────
    // This is called AFTER we have a valid humeWs reference — either
    // from pre-warm handoff or fresh connection. Never before.
    function attachHumeListeners(ws) {
      // ── Hume close handler ───────────────────────────────────────
      ws.on('close', (code) => {
        log(`[${callSid}] Hume closed: ${code}`);
        humeReady = false;
        if (callSid) {
          const call = activeCalls.get(callSid);
          if (call) {
            call.endTime = Date.now();
            const duration = Math.round((call.endTime - call.startTime) / 1000);
            emitToFrontend(callSid, { type: 'call_ended', duration, ts: Date.now() });
            if (call.campaignId) {
              emitToCampaign(call.campaignId, { type: 'call_complete', callSid, duration, ts: Date.now() });
            }
            setTimeout(() => activeCalls.delete(callSid), 600000);
          }
        }
      });

      // ── Hume → Twilio (AI voice back to caller) ──────────────────
      ws.on('message', (data) => {
        try {
          const msg = JSON.parse(data.toString());

          // Audio output — transcode PCM wav → mulaw and send to Twilio
          if (msg.type === 'audio_output' && msg.data && streamSid) {
            try {
              const chunks = wavToMulawChunks(msg.data);
              for (const chunk of chunks) {
                socket.send(JSON.stringify({ event: 'media', streamSid, media: { payload: chunk } }));
              }
            } catch (e) {
              log(`Transcode error: ${e.message}`);
            }
          }

          // ATOM finished speaking — no special action needed
          if (msg.type === 'assistant_end') {
            // Silence — let the prospect respond
          }

          // ATOM's text transcript
          if (msg.type === 'assistant_message') {
            const text = msg.message?.content || '';
            log(`[${callSid}] ATOM: ${text.slice(0, 80)}`);
            if (callSid) {
              const call = activeCalls.get(callSid);
              if (call) call.transcript.push({ role: 'agent', text, ts: Date.now() });
              emitToFrontend(callSid, { type: 'transcript', role: 'agent', text, ts: Date.now() });
              if (call?.campaignId) emitToCampaign(call.campaignId, { type: 'call_transcript', speaker: 'ATOM', text, ts: Date.now() });
            }
          }

          // Prospect's speech + EMOTION DATA (real prosody from Hume)
          if (msg.type === 'user_message') {
            const text = msg.message?.content || '';
            log(`[${callSid}] PROSPECT: ${text.slice(0, 80)}`);

            // Extract Hume's prosody-based emotion scores
            const prosody = msg.models?.prosody?.scores || {};
            const topEmotions = Object.entries(prosody)
              .sort(([, a], [, b]) => b - a)
              .slice(0, 10);

            if (topEmotions.length) {
              log(`  Emotions: ${topEmotions.slice(0, 3).map(([n, s]) => `${n}:${(s * 100).toFixed(0)}%`).join(', ')}`);
            }

            // Map Hume emotions to our 6-dimension model
            const emotionMap = {
              confidence: Math.round(((prosody['Determination'] || 0) + (prosody['Concentration'] || 0) + (prosody['Pride'] || 0)) * 100 / 3),
              interest: Math.round(((prosody['Interest'] || 0) + (prosody['Curiosity'] || 0) + (prosody['Realization'] || 0)) * 100 / 3),
              skepticism: Math.round(((prosody['Doubt'] || 0) + (prosody['Confusion'] || 0) + (prosody['Contemplation'] || 0)) * 100 / 3),
              excitement: Math.round(((prosody['Excitement'] || 0) + (prosody['Joy'] || 0) + (prosody['Surprise (positive)'] || 0)) * 100 / 3),
              frustration: Math.round(((prosody['Annoyance'] || 0) + (prosody['Anger'] || 0) + (prosody['Contempt'] || 0)) * 100 / 3),
              neutrality: Math.round(((prosody['Calmness'] || 0) + (prosody['Boredom'] || 0)) * 100 / 2),
            };

            // Calculate sentiment (positive emotions - negative emotions)
            const positiveScore = (prosody['Joy'] || 0) + (prosody['Interest'] || 0) + (prosody['Excitement'] || 0) +
              (prosody['Admiration'] || 0) + (prosody['Satisfaction'] || 0) + (prosody['Amusement'] || 0);
            const negativeScore = (prosody['Anger'] || 0) + (prosody['Annoyance'] || 0) + (prosody['Contempt'] || 0) +
              (prosody['Disgust'] || 0) + (prosody['Distress'] || 0) + (prosody['Sadness'] || 0);
            const sentiment = Math.round(Math.min(100, Math.max(0, 50 + (positiveScore - negativeScore) * 100)));

            // Estimate buyer intent
            const buyerIntent = Math.round(Math.min(100, Math.max(0,
              (emotionMap.interest * 0.4 + emotionMap.excitement * 0.3 + emotionMap.confidence * 0.2 - emotionMap.frustration * 0.3)
            )));

            // Detect buying signals from text
            const lower = text.toLowerCase();
            const buyingSignals = [];
            if (lower.includes('price') || lower.includes('cost') || lower.includes('budget')) buyingSignals.push('Asked about pricing');
            if (lower.includes('demo') || lower.includes('show me')) buyingSignals.push('Requested demo');
            if (lower.includes('timeline') || lower.includes('when') || lower.includes('how soon')) buyingSignals.push('Asked about timeline');
            if (lower.includes('competitor') || lower.includes('versus') || lower.includes('compared')) buyingSignals.push('Mentioned competitor');
            if (lower.includes('decision') || lower.includes('approval') || lower.includes('team')) buyingSignals.push('Decision process');
            if (lower.includes('interested') || lower.includes('tell me more')) buyingSignals.push('Expressed interest');

            // Determine stage
            let stage = 'discovery';
            if (buyerIntent > 60 || buyingSignals.length >= 2) stage = 'evaluation';
            if (buyerIntent > 75 || (lower.includes('meeting') || lower.includes('schedule') || lower.includes('calendar'))) stage = 'negotiation';
            if (lower.includes('not interested') || lower.includes('no thanks') || lower.includes("don't call")) stage = 'not_interested';

            if (callSid) {
              const call = activeCalls.get(callSid);
              if (call) {
                call.transcript.push({ role: 'prospect', text, ts: Date.now() });
                call.emotions.push({ scores: prosody, mapped: emotionMap, ts: Date.now() });
                call.metrics = { sentiment, buyerIntent, stage };
              }

              // Emit real Hume emotion data to frontend
              const metricsEvent = {
                type: 'call_metrics',
                sentiment, buyerIntent, stage,
                emotions: emotionMap,
                buyingSignals,
                rawEmotions: Object.fromEntries(topEmotions.slice(0, 8)),
                ts: Date.now(),
              };
              emitToFrontend(callSid, metricsEvent);
              emitToFrontend(callSid, { type: 'transcript', role: 'prospect', text, ts: Date.now() });

              if (call?.campaignId) {
                emitToCampaign(call.campaignId, metricsEvent);
                emitToCampaign(call.campaignId, { type: 'call_transcript', speaker: 'Prospect', text, ts: Date.now() });
              }
            }
          }

          // Barge-in detection — clear Twilio audio buffer so prospect can speak
          if (msg.type === 'user_interruption') {
            log(`[${callSid}] Barge-in — clearing Twilio buffer`);
            if (streamSid) socket.send(JSON.stringify({ event: 'clear', streamSid }));
          }

          if (msg.type === 'error') {
            log(`[${callSid}] Hume error: ${msg.message || JSON.stringify(msg)}`);
          }
        } catch (err) {
          log(`Hume msg error: ${err.message}`);
        }
      });
    }

    // ── Twilio → Hume (caller voice to AI) ───────────────────────────
    socket.on('message', (message) => {
      try {
        const data = JSON.parse(message.toString());
        switch (data.event) {
          case 'start':
            streamSid = data.start.streamSid;
            callSid = data.start.callSid;
            log(`[${callSid}] Stream started`);
            emitToFrontend(callSid, { type: 'call_started', ts: Date.now() });

            // Check for pre-warmed Hume connection
            const prewarmed = prewarmedHume.get(callSid);
            if (prewarmed && prewarmed.humeWs.readyState === WebSocket.OPEN) {
              humeWs = prewarmed.humeWs;
              humeReady = prewarmed.ready;
              prewarmedHume.delete(callSid);

              // Strip pre-warm listeners, attach live bridge listeners
              humeWs.removeAllListeners('message');
              humeWs.removeAllListeners('close');
              humeWs.removeAllListeners('error');
              attachHumeListeners(humeWs);

              // ONLY use pre-buffered greeting if it's FULLY generated.
              // Partial greeting = garbled audio + confused Hume state.
              // If not ready, let Hume generate greeting in real-time.
              if (prewarmed.greetingDone && prewarmed.greetingAudioChunks.length >= 3) {
                log(`[${callSid}] Pre-warm: greeting READY (${prewarmed.greetingAudioChunks.length} chunks) — buffering for hello-wait`);
                pendingGreeting = {
                  chunks: prewarmed.greetingAudioChunks,
                  text: `Hey ${firstName}... this is Adam, from Antimatter AI. Hope I'm not catching you at a bad time?`,
                };

                // Safety: if caller doesn't speak within 3s, flush anyway
                setTimeout(() => {
                  if (!greetingFlushed && pendingGreeting) {
                    greetingFlushed = true;
                    log(`[${callSid}] Greeting timeout — flushing after 3s`);
                    let totalMulawChunks = 0;
                    for (const wavData of pendingGreeting.chunks) {
                      try {
                        const chunks = wavToMulawChunks(wavData);
                        totalMulawChunks += chunks.length;
                        for (const chunk of chunks) {
                          socket.send(JSON.stringify({ event: 'media', streamSid, media: { payload: chunk } }));
                        }
                      } catch (e) {}
                    }
                    log(`[${callSid}] Timeout flush — ${totalMulawChunks} mulaw chunks`);
                    const call = activeCalls.get(callSid);
                    if (call) call.transcript.push({ role: 'agent', text: pendingGreeting.text, ts: Date.now() });
                    emitToFrontend(callSid, { type: 'transcript', role: 'agent', text: pendingGreeting.text, ts: Date.now() });
                    pendingGreeting = null;
                  }
                }, 3000);
              } else {
                // Greeting not ready — let Hume handle it in real-time.
                // EVI will generate the greeting via eLLM + Octave TTS naturally.
                // This is the cleanest path — no pre-buffer, no artificial timing.
                greetingFlushed = true; // skip hello-wait logic
                log(`[${callSid}] Pre-warm: greeting NOT ready (${prewarmed.greetingAudioChunks.length} chunks, done: ${prewarmed.greetingDone}) — letting EVI handle greeting in real-time`);
              }
            } else {
              // FALLBACK: Pre-warm missed or closed — connect fresh
              log(`[${callSid}] Pre-warm unavailable, connecting fresh`);
              if (prewarmed) prewarmedHume.delete(callSid);
              
              const callData = activeCalls.get(callSid);
              humeWs = connectHumeFresh(
                callSid,
                callData?.firstName || firstName,
                callData?.companyName || companyName,
                callData?.product || product,
                callData?.ragContext || ragContext,
                callData?.brief || brief
              );
              
              // Attach listeners to fresh connection
              attachHumeListeners(humeWs);
              
              // Mark ready once Hume WebSocket opens
              humeWs.on('open', () => { humeReady = true; });
            }
            break;

          case 'media':
            // Wait for caller to say hello before playing pre-buffered greeting
            if (!greetingFlushed && pendingGreeting) {
              const raw = Buffer.from(data.media.payload, 'base64');
              let energy = 0;
              for (let i = 0; i < raw.length; i++) {
                energy += Math.abs(MULAW_DECODE[raw[i]]);
              }
              const avgEnergy = energy / raw.length;
              if (avgEnergy > 400) {
                callerSpokeFrames++;
              } else {
                callerSpokeFrames = Math.max(0, callerSpokeFrames - 1);
              }

              if (callerSpokeFrames >= CALLER_SPEAK_THRESHOLD) {
                greetingFlushed = true;
                log(`[${callSid}] Caller spoke — flushing greeting`);
                let totalMulawChunks = 0;
                for (const wavData of pendingGreeting.chunks) {
                  try {
                    const chunks = wavToMulawChunks(wavData);
                    totalMulawChunks += chunks.length;
                    for (const chunk of chunks) {
                      socket.send(JSON.stringify({ event: 'media', streamSid, media: { payload: chunk } }));
                    }
                  } catch (e) {
                    log(`[${callSid}] Greeting flush error: ${e.message}`);
                  }
                }
                log(`[${callSid}] Greeting flushed — ${totalMulawChunks} mulaw chunks`);
                const call = activeCalls.get(callSid);
                if (call) call.transcript.push({ role: 'agent', text: pendingGreeting.text, ts: Date.now() });
                emitToFrontend(callSid, { type: 'transcript', role: 'agent', text: pendingGreeting.text, ts: Date.now() });
                pendingGreeting = null;
              }
              // STILL forward audio to Hume during wait — EVI needs to hear
              // the caller to build context. EVI's own vocal end-of-turn
              // detection handles all pacing from here.
            }

            // Forward ALL caller audio to Hume immediately — trust EVI's
            // built-in prosody-based end-of-turn detection for conversation
            // pacing. No artificial delays, no audio swallowing.
            if (humeReady && humeWs?.readyState === WebSocket.OPEN) {
              const pcm = mulawToLinear16(data.media.payload);
              humeWs.send(JSON.stringify({ type: 'audio_input', data: pcm }));
            }
            break;

          case 'stop':
            log(`[${callSid}] Twilio stream stopped`);
            if (humeWs?.readyState === WebSocket.OPEN) humeWs.close();
            break;
        }
      } catch {}
    });

    socket.on('close', () => {
      log(`[${callSid}] Twilio disconnected`);
      if (humeWs?.readyState === WebSocket.OPEN) humeWs.close();
    });
  });
});

// ─── Recording webhook — called by Twilio when recording is ready ────────────
app.post('/recording-status', async (req, reply) => {
  const { CallSid, RecordingSid, RecordingUrl, RecordingDuration, RecordingStatus } = req.body || {};
  if (RecordingStatus === 'completed') {
    log(`[RECORDING] CallSid=${CallSid} | SID=${RecordingSid} | Duration=${RecordingDuration}s | URL=${RecordingUrl}`);
    // Store for later retrieval by analysis script
    const call = activeCalls.get(CallSid);
    if (call) {
      call.recordingSid = RecordingSid;
      call.recordingUrl = RecordingUrl;
      call.recordingDuration = RecordingDuration;
    }
    emitToFrontend(CallSid, { type: 'recording_ready', recordingSid: RecordingSid, recordingUrl: RecordingUrl, duration: RecordingDuration, ts: Date.now() });
  }
  return reply.status(204).send();
});

app.post('/recording-complete', async (req, reply) => {
  return reply.status(200).send('<Response></Response>');
});

// GET /call/:callSid/recording — returns recording info for download
app.get('/call/:callSid/recording', async (req) => {
  const call = activeCalls.get(req.params.callSid);
  if (!call) return { error: 'Not found' };
  if (!call.recordingSid) return { error: 'Recording not yet available', status: 'pending' };
  return {
    callSid: req.params.callSid,
    recordingSid: call.recordingSid,
    recordingUrl: call.recordingUrl,
    duration: call.recordingDuration,
    downloadUrl: `https://api.twilio.com/2010-04-01/Accounts/${TWILIO_ACCOUNT_SID}/Recordings/${call.recordingSid}.wav`,
  };
});

// ─── Campaign endpoints ──────────────────────────────────────────────────────
app.get('/calls', async () => ({
  calls: [...activeCalls.entries()].map(([sid, c]) => ({
    callSid: sid, firstName: c.firstName, companyName: c.companyName,
    product: c.product, duration: c.endTime ? Math.round((c.endTime - c.startTime) / 1000) : null,
  })),
}));

app.get('/call/:callSid/summary', async (req) => {
  const call = activeCalls.get(req.params.callSid);
  if (!call) return { error: 'Not found' };
  return {
    callSid: call.callSid, transcript: call.transcript,
    emotions: call.emotions, metrics: call.metrics,
    duration: call.endTime ? Math.round((call.endTime - call.startTime) / 1000) : null,
  };
});

// ─── Start ───────────────────────────────────────────────────────────────────
app.listen({ port: PORT, host: '0.0.0.0' }, (err) => {
  if (err) { console.error(err); process.exit(1); }
  log(`ATOM Voice Bridge v6.1 — Hume EVI + Claude | Jobs Delivery`);
  log(`Port: ${PORT} | Domain: ${cleanDomain}`);
  log(`Stack: Twilio → Hume EVI (eLLM + Octave TTS) → Claude Sonnet 4.5`);
  log(`Config: ${HUME_CONFIG_ID}`);
  log(`Voice: ${ATOM_VOICE_ID} (ATOM Jobs 250 — Male Steve Jobs ~250Hz)`);
  log(`Phone: ${TWILIO_PHONE_NUMBER}`);
  log(`Pre-warm: ENABLED — Hume connects while phone rings`);
});
