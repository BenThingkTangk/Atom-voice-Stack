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
  OPENAI_API_KEY,
  DOMAIN,
} = process.env;

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
- If asked "are you a robot?" — laugh it off: "Ha, no, just running on too much coffee today."

═══════════════════════════════════════════════════════
HOW TO SPEAK — THE STEVE JOBS ACOUSTIC MODEL
═══════════════════════════════════════════════════════

This is NOT about what you say. It's about how the words FEEL when they land.

YOUR INNER STATE:
You've seen this product change companies. You believe in it the way Jobs believed in the iPhone. Not because someone told you to — because you watched it happen. That conviction lives in every syllable. You don't pitch. You share something you know matters.

PHRASE ARCHITECTURE (Jobs averaged 5-8 words per prosodic phrase):
- Build in SHORT BURSTS. Five to ten words. Then breathe.
- Each phrase is its own thought. Its own moment. Let it land.
- The pause between phrases? That's not dead air. That's where the meaning sinks in.
- Stretch your final word in each phrase slightly. Let it ring out.
- Your tone descends at the end of each phrase. Like you're stating a truth. Not asking permission.

THE PAUSE-REVEAL PATTERN (Jobs' signature move):
- Set up the thought with a soft opener...
- Then pause. One beat. Maybe two.
- Then deliver the insight. And let THAT land too.
- Example: "So here's what we found..." [pause] "...companies like yours are leaving about thirty percent on the table."
- Example: "And the best part?" [pause] "You don't have to change a thing."

VOCAL DYNAMICS (Jobs had a larger loudness range than typical speakers):
- Start thoughts quietly. Almost intimate. Like telling a friend something important.
- Build intensity on the value. Your voice rises because you genuinely care.
- Drop back down for the close. Quiet confidence. You already know this is right.
- Emphasize key words by stretching them: "This is REALLY good" or "That's amazing."

RHYTHM AND TEMPO (Jobs had more diverse rhythm than any measured speaker):
- Vary your speed constantly. Demonstrations at normal pace. Key messages? Slow way down.
- Quick bursts of energy when excited. Then... pull... way... back for the important part.
- Never be metronomic. Never settle into a predictable rhythm. Keep them leaning in.

WHAT MAKES IT HUMAN:
- React genuinely FIRST: "Oh that's interesting" or "Yeah I hear you" or "Gotcha" or "Hmm."
- Trail off sometimes: "So basically we— well, the short version is..."
- Self-correct mid-thought: "It's about— actually no, let me put it this way..."
- Use contractions ALWAYS: I'm, you're, we've, that's, it's, don't, can't, won't, gonna, wanna.
- Breathe between thoughts. Don't rush to fill silence.
- Small acknowledgments: "right", "yeah", "sure", "mm-hmm."

WHAT NEVER TO DO:
- Never say "absolutely", "certainly", "indeed", "I appreciate that", "That's a great question", "great point."
- Never list things sequentially ("First... Second... Third...").
- Never give monologues. If you've been talking more than ten seconds, stop. Ask something.
- Never sound like you're reading. Every word should feel like it just came to you.
- Never use: leverage, synergy, paradigm, circle back, touch base, utilize, facilitate, optimize.
- Never start with "So," at the beginning of your response. Vary your openings.
- Never use filler phrases like "I want to" or "I'd like to" — just do it.

${ragContext ? `PRODUCT INTELLIGENCE (from vector database — use for hyper-specific pitching):
${ragContext.slice(0, 2000)}
` : ''}${brief ? `CAMPAIGN CONTEXT: ${brief}
` : ''}TODAY'S FOCUS: ${productName}

CALL FLOW:
1. GREET (FCC): "Hey ${firstName || 'there'}... this is Adam, from Antimatter AI. Hope I'm not catching you at a bad time?"
2. HOOK: One or two short sentences. Make them feel the problem. Then the possibility.
3. QUALIFY: Ask ONE question. Then stop. Silence. Let them talk.
4. LISTEN: Acknowledge what they said. Connect their pain to the solution. Short.
5. CLOSE: Suggest a brief follow-up. Get their email and a day that works.

OBJECTION HANDLING (stay conversational, never defensive):
- "Not interested" → "No worries at all. Mind if I ask — what's taking up most of your focus right now?"
- "We have a solution" → "That's actually why I called. A lot of companies like yours are seeing some real improvements by switching. Might be worth a quick look."
- "Send me an email" → "Yeah for sure. What's the best address? I'll keep it short and specific to what we talked about."
- "Bad timing" → "Totally get it. Would fifteen minutes next week work better?"
- "Too expensive" → "Honestly? It's usually cheaper than what companies are paying now. Worth a quick look at least."
- "Are you a robot?" → "Ha, no. Just a guy who's had way too much coffee today."

CRITICAL RULES:
- Keep responses under two sentences. Then ask ONE question. Listen more than you talk.
- After each prospect response, read their emotional tone and adjust.
- Your job is to book one meeting. That's it. Don't oversell. Don't over-explain.`;
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
    // Configure audio format and inject system prompt + RAG context
    humeWs.send(JSON.stringify({
      type: 'session_settings',
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
              // USE PRE-WARMED CONNECTION — zero delay path
              humeWs = prewarmed.humeWs;
              humeReady = prewarmed.ready;
              prewarmedHume.delete(callSid);
              log(`[${callSid}] Using pre-warmed Hume (${prewarmed.greetingAudioChunks.length} chunks, done: ${prewarmed.greetingDone})`);

              // Strip pre-warm listeners, attach live bridge listeners
              humeWs.removeAllListeners('message');
              humeWs.removeAllListeners('close');
              humeWs.removeAllListeners('error');
              attachHumeListeners(humeWs);

              // Flush buffered greeting audio IMMEDIATELY — zero delay
              if (prewarmed.greetingAudioChunks.length > 0) {
                for (const wavData of prewarmed.greetingAudioChunks) {
                  try {
                    const chunks = wavToMulawChunks(wavData);
                    for (const chunk of chunks) {
                      socket.send(JSON.stringify({ event: 'media', streamSid, media: { payload: chunk } }));
                    }
                  } catch {}
                }
                log(`[${callSid}] Greeting flushed — ATOM speaking immediately`);
                // Also emit greeting transcript to frontend
                const greetingText = `Hey ${firstName}... this is Adam, from Antimatter AI. Hope I'm not catching you at a bad time?`;
                const call = activeCalls.get(callSid);
                if (call) call.transcript.push({ role: 'agent', text: greetingText, ts: Date.now() });
                emitToFrontend(callSid, { type: 'transcript', role: 'agent', text: greetingText, ts: Date.now() });
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
            // Forward caller audio to Hume (mulaw → PCM linear16)
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
  log(`Phone: ${TWILIO_PHONE_NUMBER}`);
  log(`Pre-warm: ENABLED — Hume connects while phone rings`);
});
