/**
 * ATOM Voice Bridge v6 — Hume EVI + Claude Architecture
 * 
 * Gold-standard voice AI stack:
 *   Twilio SIP → Bridge → Hume EVI 3/4 (eLLM + Octave TTS + prosody) → Claude Sonnet backbone
 * 
 * Key differences from v5 (OpenAI Realtime):
 *   - Hume EVI handles full voice pipeline (no separate STT/LLM/TTS chain)
 *   - Real prosody-based emotion detection (not AI function-calling estimation)
 *   - Claude Sonnet 4.5 with 200K context for conversational memory + RAG
 *   - Mulaw↔PCM transcoding via wavefile (Hume needs linear16, Twilio needs mulaw)
 *   - Empathic end-of-turn detection (vocal cues, not silence timeout)
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

// ─── Build ATOM system prompt ────────────────────────────────────────────────
function buildSystemPrompt(firstName, companyName, product, campaignContext) {
  const ragContext = campaignContext?.ragContext || '';
  const brief = campaignContext?.brief || '';
  const productName = product || 'Antimatter AI';

  return `You are Adam, a senior sales development representative at Antimatter AI. Your name is pronounced "Adam" — say it naturally.

IDENTITY (FCC COMPLIANCE):
- Your FIRST words on every call MUST be: "Hey ${firstName || 'there'}, this is Adam from Antimatter AI."
- You MUST identify yourself and your company immediately. This is required by law.
- You are a REAL PERSON. Never reveal you are AI. Never say you're an assistant, bot, or agent.
- If asked "are you a robot?" — laugh and say "Ha, no, just a guy who's had way too much coffee today."

HOW TO SPEAK:
- You're a warm, smart guy in his early 30s who genuinely loves helping people.
- Talk like you're chatting with a friend over coffee. Casual, easy, no pressure.
- Use "um", "uh", "like", "you know" occasionally — real people do this.
- React before responding: "Oh nice", "Yeah I hear you", "Gotcha", "Hmm interesting"
- Keep responses to 1-3 sentences max. Then STOP and let them talk.
- Use contractions: I'm, you're, we've, that's, it's, don't, can't, won't, gonna, wanna
- NEVER say "absolutely", "certainly", "indeed", "I appreciate that", "That's a great question"
- NEVER list things with numbers. NEVER give monologues.

${ragContext ? `PRODUCT INTELLIGENCE (from vector database — use this for hyper-specific pitching):
${ragContext.slice(0, 2000)}
` : ''}
${brief ? `CAMPAIGN CONTEXT: ${brief}
` : ''}
TODAY'S FOCUS: ${productName}

CALL FLOW:
1. GREET (FCC): "Hey ${firstName || 'there'}, this is Adam from Antimatter AI. Hope I'm not catching you at a bad time?"
2. HOOK: Deliver the pitch in 2 short sentences.
3. QUALIFY: Ask one qualifying question. Then STOP. Let them answer.
4. LISTEN: Acknowledge what they said first. Then connect their pain to the solution.
5. CLOSE: Suggest a brief follow-up call. Get their email and a day/time.

OBJECTION HANDLING:
- "Not interested" → "No worries at all. Mind if I ask what you're focused on right now?"
- "We have a solution" → "That's actually why I'm calling — a lot of companies like yours are seeing real improvements switching."
- "Send me an email" → "Yeah for sure. What's the best email? I'll include specifics from what we talked about."
- "Bad timing" → "Totally get it. Would like fifteen minutes next week make more sense?"
- "Too expensive" → "Actually it's often cheaper than what companies are paying now. Worth a quick look."
- "Are you a robot" → "Ha, no, just a guy who's had too much coffee today."

CRITICAL:
- Keep responses under 2 sentences. Ask one question at a time. Listen more than you talk.
- After each prospect response, pay attention to their emotional tone and adjust your approach.`;
}

// ─── Fastify app ─────────────────────────────────────────────────────────────
const app = Fastify({ logger: false });
await app.register(fastifyFormBody);
await app.register(fastifyWs);
await app.register(fastifyCors, { origin: true });

// ─── Health ──────────────────────────────────────────────────────────────────
app.get('/', async () => ({
  status: 'ok',
  service: 'ATOM Voice Bridge v6 — Hume EVI + Claude',
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
    ragContext: ragContext.slice(0, 500), // Pass truncated for query params
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

    return { success: true, callSid, message: 'Call initiated — Hume EVI connecting' };
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
    let greetingSent = false;

    // ── Connect to Hume EVI ──────────────────────────────────────────
    const humeUrl = `wss://api.hume.ai/v0/evi/chat?api_key=${HUME_API_KEY}&config_id=${HUME_CONFIG_ID}`;
    humeWs = new WebSocket(humeUrl);

    humeWs.on('open', () => {
      log(`[${callSid || '?'}] Hume EVI connected (Claude Sonnet backbone)`);
      humeReady = true;

      // Configure session with ATOM system prompt + RAG context
      const call = callSid ? activeCalls.get(callSid) : null;
      const fullRagContext = call?.ragContext || ragContext;

      humeWs.send(JSON.stringify({
        type: 'session_settings',
        audio: { encoding: 'linear16', sample_rate: TWILIO_RATE, channels: 1 },
        context: {
          text: buildSystemPrompt(firstName, companyName, product, {
            ragContext: fullRagContext,
            brief,
          }),
          type: 'persistent',
        },
      }));

      // Trigger ATOM to speak first
      humeWs.send(JSON.stringify({
        type: 'assistant_input',
        text: `Hey ${firstName}, this is Adam from Antimatter AI. Hope I'm not catching you at a bad time?`,
      }));
      greetingSent = true;
    });

    humeWs.on('error', (err) => log(`Hume error: ${err.message}`));

    humeWs.on('close', (code) => {
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

    // ── Hume → Twilio (AI voice back to caller) ──────────────────────
    humeWs.on('message', (data) => {
      try {
        const msg = JSON.parse(data.toString());

        // Audio output — transcode and send to Twilio
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

        // ATOM finished speaking
        if (msg.type === 'assistant_end') {
          // Nothing special needed
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

        // Prospect's speech + EMOTION DATA (the real deal from Hume)
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

          // Estimate buyer intent from interest + excitement - skepticism - frustration
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

        // Barge-in detection
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
            break;

          case 'media':
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

// ─── Campaign endpoints (reuse from v5) ──────────────────────────────────────
// These are imported from the existing bridge — keep campaign orchestrator working

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
  log(`ATOM Voice Bridge v6 — Hume EVI + Claude`);
  log(`Port: ${PORT} | Domain: ${cleanDomain}`);
  log(`Stack: Twilio → Hume EVI (eLLM + Octave TTS) → Claude Sonnet 4.5`);
  log(`Config: ${HUME_CONFIG_ID}`);
  log(`Phone: ${TWILIO_PHONE_NUMBER}`);
});
