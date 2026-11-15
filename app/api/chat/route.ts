// POST /api/chat — one turn with the person's own agent. Owner: Role 2.
// Body: { text?, imageBase64?, audioBase64?, coords?, speak?, speechSpeed?, confirm? }
//   → { reply, eventIds, actions, heard?, replyAudio?, pending? }
// confirm: true holds what the agent would file. Nothing reaches the desk yet; `pending` = { id, items }
//   lists it. Then POST { approve: id, send: true|false } files it (or drops it) → { sent, eventIds, actions }.
//   Without confirm the agent files at once, as before (the field phone, the API).
// Who is talking: the login, once SSOJet is in. Until then ?user=<id> (or body.userId),
// accepted only when DEMO_MODE=1 and the request comes from this laptop (isLocalDemo, #11).
// Through a tunnel or the network, people must sign in.
// Voice (ElevenLabs, when ELEVENLABS_API_KEY is set):
//   audioBase64  a voice note (webm/opus from the phone's MediaRecorder, m4a, mp3, wav) is turned
//                into text with Scribe and handled like typed text; `heard` returns what was heard.
//   speak: true  the reply also comes back as speech: `replyAudio` is a data URL (audio/mpeg).
// Without the key, the browser's own speech recognition still works (it sends `text`), and a raw
// audioBase64 gets a polite reply, not an error.

import { after, NextResponse } from "next/server";
import { refreshChatSummary } from "@/agents/personal/memory";
import { runTurn, settlePending, type TurnInput } from "@/agents/personal/turn";
import { desk } from "@/agents/desk";
import { speak, transcribe, voiceEnabled } from "@/agents/shared/voice";
import { sessionUserId } from "@/auth";
import { isLocalDemo } from "@/lib/auth";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_TEXT = 4000;
const MAX_IMAGE_BASE64 = 5 * 1024 * 1024; // the API's 5 MB image limit applies to the base64 data
const MAX_AUDIO_BASE64 = 10 * 1024 * 1024; // about 7 MB of audio: minutes of voice, far more than a report

// Small in-memory rate limit: protects the $20 key from a stuck client or a curious visitor.
const PER_MINUTE = 8;
const PER_HOUR = 80;
const g = globalThis as unknown as { __chatHits?: Map<string, number[]> };
const hits: Map<string, number[]> = (g.__chatHits ??= new Map<string, number[]>());
function limited(key: string): boolean {
  const now = Date.now();
  const xs = (hits.get(key) ?? []).filter((t) => now - t < 3600_000);
  if (xs.filter((t) => now - t < 60_000).length >= PER_MINUTE || xs.length >= PER_HOUR) {
    hits.set(key, xs);
    return true;
  }
  xs.push(now);
  hits.set(key, xs);
  return false;
}

/** Media type from the first bytes, not from what the client claims. */
function sniff(base64: string): NonNullable<TurnInput["image"]>["mediaType"] | undefined {
  if (base64.startsWith("/9j/")) return "image/jpeg";
  if (base64.startsWith("iVBOR")) return "image/png";
  if (base64.startsWith("R0lGOD")) return "image/gif";
  if (base64.startsWith("UklGR")) return "image/webp";
  return undefined;
}

interface Body {
  userId?: unknown;
  text?: unknown;
  imageBase64?: unknown;
  audioBase64?: unknown;
  coords?: unknown;
  speak?: unknown;
  speechSpeed?: unknown;         // 0.7–1.2, the /talk voice-speed slider
  confirm?: unknown;             // true: hold the filing until the person says yes
  approve?: unknown;             // the pending id of an earlier confirm turn
  send?: unknown;                // with approve: true sends it, anything else drops it
}

export async function POST(req: Request) {
  let body: Body;
  try {
    body = (await req.json()) as Body;
  } catch {
    return NextResponse.json({ error: "body must be JSON" }, { status: 400 });
  }

  // The SSOJet login wins. ?user= / body.userId is the demo fallback, only on this laptop.
  let userId = await sessionUserId();
  if (!userId) {
    userId = new URL(req.url).searchParams.get("user") ?? (typeof body.userId === "string" ? body.userId : null);
    if (!userId) return NextResponse.json({ error: "who is talking? (login, or ?user= in demo mode)" }, { status: 401 });
    if (!isLocalDemo(req)) return NextResponse.json({ error: "Sign in. Demo users work only on this laptop." }, { status: 401 });
  }
  if (!(await desk().getUser(userId))) return NextResponse.json({ error: `unknown user ${userId}` }, { status: 404 });

  // The person answered "send this?" for an earlier confirm turn. No model call, so no rate limit.
  if (body.approve !== undefined) {
    if (typeof body.approve !== "string") return NextResponse.json({ error: "approve must be a string" }, { status: 400 });
    const r = await settlePending(userId, body.approve, body.send === true);
    if (!r) return NextResponse.json({ error: "That is no longer waiting. Say it again." }, { status: 410 });
    return NextResponse.json(r);
  }

  const ip = (req.headers.get("x-forwarded-for") ?? "local").split(",")[0].trim();
  if (limited(`${ip}|${userId}`)) {
    return NextResponse.json({ error: "too many messages, wait a minute" }, { status: 429 });
  }

  let text = typeof body.text === "string" ? body.text.slice(0, MAX_TEXT) : undefined;
  let heard: string | undefined;
  if (!text && body.audioBase64 !== undefined) {
    if (typeof body.audioBase64 !== "string") return NextResponse.json({ error: "audioBase64 must be a string" }, { status: 400 });
    if (!voiceEnabled()) {
      if (!body.imageBase64) {
        return NextResponse.json({
          reply: "I can't listen to audio files yet. Hold the mic button so your phone types what you say, or write it.",
          eventIds: [], actions: [],
        });
      }
    } else {
      const m = /^data:([^;,]+)[^,]*;base64,/.exec(body.audioBase64);
      const base64 = body.audioBase64.replace(/^data:[^,]*,/, "").replace(/\s/g, "");
      if (base64.length > MAX_AUDIO_BASE64) return NextResponse.json({ error: "voice note too long" }, { status: 413 });
      try {
        const t = await transcribe(Buffer.from(base64, "base64"), m?.[1] ?? "audio/webm");
        if (!t.text) {
          return NextResponse.json({ reply: "I couldn't hear any words. Hold the button a little longer and speak close to the phone.",
            eventIds: [], actions: [], heard: "" });
        }
        text = t.text.slice(0, MAX_TEXT);
        heard = text;
      } catch (e) {
        console.error("[api/chat] speech-to-text", e);
        return NextResponse.json({ reply: "I couldn't turn that voice note into text. Please try again or type it.",
          eventIds: [], actions: [] });
      }
    }
  }

  let image: TurnInput["image"];
  if (body.imageBase64 !== undefined) {
    if (typeof body.imageBase64 !== "string") return NextResponse.json({ error: "imageBase64 must be a string" }, { status: 400 });
    const base64 = body.imageBase64.replace(/^data:[^;,]*;base64,/, "").replace(/\s/g, "");
    if (base64.length > MAX_IMAGE_BASE64) {
      return NextResponse.json({ error: "photo too large: send at most about 3.5 MB (resize to 1568 px on the long side)" }, { status: 413 });
    }
    const mediaType = sniff(base64);
    if (!mediaType) return NextResponse.json({ error: "photo must be JPEG, PNG, WebP or GIF" }, { status: 415 });
    image = { mediaType, base64 };
  }

  const c = body.coords;
  const coords = Array.isArray(c) && c.length === 2 && c.every((x) => typeof x === "number" && Number.isFinite(x))
    ? (c as [number, number]) : undefined;

  try {
    const result = await runTurn({ userId, text, image, coords, confirm: body.confirm === true });
    // After the reply is sent: fold old chat lines into the summary (MongoDB desk only).
    if (desk().name === "mongodb") after(() => refreshChatSummary(userId!));
    // The reply as speech, when asked for. A voice failure never loses the written reply.
    let replyAudio: string | undefined;
    if (body.speak === true && voiceEnabled() && result.reply) {
      try { replyAudio = `data:audio/mpeg;base64,${(await speak(result.reply, { speed: typeof body.speechSpeed === "number" ? body.speechSpeed : undefined })).toString("base64")}`; }
      catch (e) { console.error("[api/chat] text-to-speech", e); }
    }
    return NextResponse.json({ ...result, ...(heard !== undefined ? { heard } : {}), ...(replyAudio ? { replyAudio } : {}) });
  } catch (e) {
    console.error("[api/chat]", e);
    return NextResponse.json({ error: "the agent failed this turn, try again" }, { status: 502 });
  }
}

// GET /api/chat: whether this server has ElevenLabs voice. The field phone picks its microphone path
// from it (voice notes through ElevenLabs, or the browser's own speech). Only a yes or no leaves.
export async function GET() {
  return NextResponse.json({ voice: voiceEnabled() }, { headers: { "cache-control": "no-store" } });
}
