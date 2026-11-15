"use client";
// /talk: one person and their own agent, voice first, laid out like a phone voice mode.
// A big orb sits at the bottom centre: hold it (or the space bar) and talk, it grows and follows
// your voice; let go and the note goes to your agent. The conversation flows top to bottom above it.
// The agent's answer shows as plain text, whole, and is read out loud at the same time.
// Nothing reaches the desk yet: the agent only prepares it, and once the answer has been read a
// card asks "Send this to the desk?". Send files it, Don't send drops it.
// Pressing while the agent speaks talks over it. Typing still works. Tasks, the person switcher
// and the voice settings are small panels at the top. Owner: Role 2.
import { AnimatePresence, motion } from "motion/react";
import {
  ChevronDown, CircleCheck, CircleX, ClipboardList, HeartPulse, Keyboard, Mic, SendHorizontal, Siren,
  SlidersHorizontal, Truck, Volume2, VolumeX, X,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { cn } from "@/lib/utils";
import type { Tier } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { IntensitySlider } from "@/components/motion/IntensitySlider";
import { PinnedList, type PinnedItem } from "@/components/motion/PinnedList";
import { VoiceOrb } from "./VoiceOrb";
import { useVoice, type LoadedReply, type VoiceState } from "./useVoice";

export interface TalkTask { id: string; title: string; detail: string; tier: Tier; status: string }
export interface TalkQuake { line: string; mag: number; minutesAgo: number }
export interface TalkPerson { id: string; name: string; role: string }

interface Turn {
  id: number;
  from: "person" | "agent";
  text: string;
  voice?: boolean;                       // the person spoke it (ElevenLabs heard it)
  speaking?: boolean;                    // agent: the voice is reading it out right now
  actions?: { tool: string; summary: string; ok: boolean }[];
  // agent: what the desk would get, waiting for the person's yes
  pending?: { id: string; items: string[]; state: "ask" | "sending" | "sent" | "dropped" | "error"; error?: string };
}

const TOOL_WORD: Record<string, string> = {
  answer_task: "Task", file_need: "Need", file_report: "Report", file_closure: "Road",
  file_capacity: "Shelter", report_problem: "Problem", declare_work: "Doing", correct_record: "Fix",
  decide_approval: "Decision", shift_handoff: "Handoff", show_on_map: "Map",
};

const ROLE_WORD: Record<string, string> = {
  field: "Field team", volunteer: "Volunteer", supplier: "Supplier", shelter: "Shelter desk",
  coordinator: "Coordinator", citizen: "Citizen", public: "Citizen",
};
const roleWord = (r: string) => ROLE_WORD[r] ?? r.replaceAll("_", " ").replace(/^\w/, (c) => c.toUpperCase());

const SUGGEST = [
  "Two people are trapped at Örnek Street 12",
  "We need 20 blankets at Shelter B",
  "The road to the hospital is blocked",
];

/** "Problem: road_blocked on task task-5" → "road blocked": no internal ids on screen. */
function chipText(summary: string): string {
  return summary.replace(/^[A-Za-z ]+: /, "").replace(/ on task [\w-]+/g, "").replace(/\b(task|site|need|inc)-\w+/g, "")
    .replace(/_/g, " ").replace(/\s+/g, " ").trim();
}

const initialsOf = (name: string) => name.split(/\s+/).map((w) => w[0]).join("").slice(0, 2).toUpperCase();
/** A small panel that drops down from the top bar. */
function Panel({ open, onClose, title, children, align = "left" }: {
  open: boolean; onClose: () => void; title: string; children: React.ReactNode; align?: "left" | "right";
}) {
  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div className="fixed inset-0 z-30" onClick={onClose} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} />
          <motion.div role="dialog" aria-label={title}
            initial={{ opacity: 0, y: -8, scale: 0.97 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: -8, scale: 0.97 }}
            transition={{ type: "spring", stiffness: 420, damping: 32 }}
            className={cn("absolute top-12 z-40 w-[min(22rem,calc(100vw-2rem))] rounded-2xl border bg-popover p-3 text-popover-foreground shadow-lg",
              align === "left" ? "left-0" : "right-0")}>
            <div className="mb-2 flex items-center justify-between">
              <p className="text-xs font-medium text-muted-foreground">{title}</p>
              <Button variant="ghost" size="icon-sm" onClick={onClose} aria-label="Close"><X /></Button>
            </div>
            {children}
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

export function TalkScreen({ me, people, tasks, quakes, voiceOn }: {
  me: TalkPerson; people: TalkPerson[]; tasks: TalkTask[]; quakes: TalkQuake[]; voiceOn: boolean;
}) {
  const router = useRouter();
  const voice = useVoice();
  const [turns, setTurns] = useState<Turn[]>([]);
  const [state, setStateRaw] = useState<VoiceState>("idle");
  const stateRef = useRef<VoiceState>("idle");
  const setState = useCallback((s: VoiceState) => { stateRef.current = s; setStateRaw(s); }, []);
  const [typing, setTyping] = useState(!voiceOn);
  const [typed, setTyped] = useState("");
  const [speakReplies, setSpeakReplies] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [panel, setPanel] = useState<"people" | "tasks" | "settings" | null>(null);
  const seq = useRef(0);
  const holding = useRef(false);
  const scroller = useRef<HTMLDivElement>(null);

  // Keep the newest line in view while it flows.
  const toBottom = useCallback(() => {
    const el = scroller.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, []);
  const lastTurn = turns.at(-1);
  useEffect(toBottom, [turns.length, state, toBottom, lastTurn?.speaking, lastTurn?.pending?.state]);
  // Aftershocks and the task list come from the server: refresh them every 30 s.
  useEffect(() => { const t = setInterval(() => router.refresh(), 30_000); return () => clearInterval(t); }, [router]);

  // The id is taken here, not inside the updater: React may run an updater twice (Strict Mode).
  const add = (t: Omit<Turn, "id">) => { const id = ++seq.current; setTurns((xs) => [...xs, { ...t, id }]); return id; };
  /** The voice is done (or was talked over): the "send this?" card may show. */
  const stopSpeaking = useCallback((id?: number) =>
    setTurns((xs) => xs.map((t) => (t.speaking && (id === undefined || t.id === id) ? { ...t, speaking: false } : t))), []);
  const patchTurn = (id: number, f: (t: Turn) => Turn) => setTurns((xs) => xs.map((t) => (t.id === id ? f(t) : t)));

  async function send(body: { text?: string; audioBase64?: string }) {
    setState("thinking");
    try {
      const res = await fetch(`/api/chat?user=${encodeURIComponent(me.id)}`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ ...body, speak: speakReplies && voiceOn, speechSpeed: speed, confirm: true }),
      });
      const r = await res.json();
      if (!res.ok) throw new Error(r.error ?? `HTTP ${res.status}`);
      if (body.audioBase64) add({ from: "person", text: r.heard || "(nothing heard)", voice: true });
      let reply: LoadedReply | null = null;
      if (r.replyAudio) reply = await voice.load(r.replyAudio).catch(() => null);
      const id = add({ from: "agent", text: r.reply, actions: r.actions, speaking: Boolean(reply),
        pending: r.pending ? { ...r.pending, state: "ask" } : undefined });
      if (reply) {
        setState("speaking");
        await reply.play().catch(() => {});
        stopSpeaking(id);
      }
    } catch (e) {
      add({ from: "agent", text: `Something went wrong: ${(e as Error).message}. Try again.` });
    } finally {
      // Only settle if nobody started talking over the answer meanwhile.
      if (stateRef.current === "thinking" || stateRef.current === "speaking") setState("idle");
    }
  }

  async function pressStart() {
    const s = stateRef.current;
    if (s === "thinking" || s === "listening" || !voiceOn) return;
    if (s === "speaking") { voice.stopPlayback(); stopSpeaking(); }
    holding.current = true;
    setState("listening");           // the orb grows at once, even while the mic is still opening
    const ok = await voice.start();
    if (!ok) { holding.current = false; setState("idle"); return; }
    if (!holding.current) {          // let go before the mic was ready: nothing to send
      voice.cancel();
      setState("idle");
      voice.setError("Hold the orb while you talk.");
    }
  }
  async function pressEnd() {
    if (!holding.current) return;
    holding.current = false;
    if (stateRef.current !== "listening") return;
    const note = await voice.stop();
    if (!note) {
      // stop() returns null both for a tap and while the mic is still opening (pressStart handles that).
      if (stateRef.current === "listening") { setState("idle"); voice.setError("Hold the orb while you talk."); }
      return;
    }
    await send({ audioBase64: note });
  }
  async function submit(text: string) {
    const t = text.trim();
    if (!t || stateRef.current === "thinking" || stateRef.current === "listening") return;
    if (stateRef.current === "speaking") { voice.stopPlayback(); stopSpeaking(); }
    setTyped("");
    add({ from: "person", text: t });
    await send({ text: t });
  }

  /** The person's answer to "Send this to the desk?" */
  async function decide(turnId: number, pendingId: string, send: boolean) {
    patchTurn(turnId, (t) => (t.pending ? { ...t, pending: { ...t.pending, state: "sending" } } : t));
    try {
      const res = await fetch(`/api/chat?user=${encodeURIComponent(me.id)}`, {
        method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ approve: pendingId, send }),
      });
      const r = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(r.error ?? "Could not reach the desk. Tap Send to try again.");
      patchTurn(turnId, (t) => ({ ...t, actions: r.sent ? r.actions : t.actions,
        pending: t.pending ? { ...t.pending, state: r.sent ? "sent" : "dropped" } : t.pending }));
    } catch (e) {
      patchTurn(turnId, (t) => (t.pending ? { ...t, pending: { ...t.pending, state: "error", error: (e as Error).message } } : t));
    }
  }

  // Space bar: hold to talk, like the orb (not while typing in a field).
  useEffect(() => {
    const inField = (e: KeyboardEvent) => e.target instanceof HTMLElement && e.target.closest("input, textarea, [contenteditable]");
    const down = (e: KeyboardEvent) => {
      if (e.code !== "Space" || e.repeat || inField(e)) return;
      e.preventDefault();
      void pressStart();
    };
    const up = (e: KeyboardEvent) => {
      if (e.code !== "Space" || inField(e)) return;
      e.preventDefault();
      void pressEnd();
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => { window.removeEventListener("keydown", down); window.removeEventListener("keyup", up); };
  });

  const fresh = quakes.filter((q) => q.minutesAgo <= 30);
  const pinned: PinnedItem[] = tasks.map((t, i) => ({
    id: t.id, title: t.title, note: t.detail, pinned: i === 0,
    icon: t.tier === 0 ? HeartPulse : t.tier === 1 ? Truck : ClipboardList,
    tone: t.tier === 0 ? "act" : t.tier === 1 ? "urgent" : "calm",
  }));
  const status = state === "listening" ? "Listening… let go to send"
    : state === "thinking" ? "Filing it on the desk…"
    : state === "speaking" ? "Your agent is answering · hold to talk over it"
    : voiceOn ? "Hold to talk" : "Voice is off here. Type instead.";
  const orbAnalyser = state === "speaking" ? voice.agentAnalyser : state === "listening" ? voice.micAnalyser : null;
  const lastFrom = turns.at(-1)?.from;

  return (
    <main className="relative mx-auto flex h-full w-full max-w-2xl flex-col px-4">
      {/* Top bar: who you are, your tasks, voice settings */}
      <div className="relative z-20 flex items-center gap-2 pt-3">
        <button type="button" onClick={() => setPanel(panel === "people" ? null : "people")} aria-expanded={panel === "people"}
          className="flex min-w-0 items-center gap-2 rounded-full border bg-card py-1 pr-3 pl-1 text-left shadow-xs transition-colors hover:bg-accent">
          <span className="flex size-7 shrink-0 items-center justify-center rounded-full bg-primary text-[10px] font-semibold text-primary-foreground">{initialsOf(me.name)}</span>
          <span className="min-w-0">
            <span className="block truncate text-sm leading-tight font-medium">{me.name}</span>
            <span className="block truncate text-[11px] leading-tight text-muted-foreground">{roleWord(me.role)}</span>
          </span>
          <ChevronDown className="size-3.5 shrink-0 text-muted-foreground" />
        </button>
        <div className="flex-1" />
        <Button variant="outline" size="sm" className="rounded-full" onClick={() => setPanel(panel === "tasks" ? null : "tasks")} aria-expanded={panel === "tasks"}>
          <ClipboardList />Tasks{tasks.length > 0 && <span className="rounded-full bg-primary px-1.5 text-[10px] text-primary-foreground">{tasks.length}</span>}
        </Button>
        <Button variant="ghost" size="icon-sm" aria-pressed={speakReplies} onClick={() => setSpeakReplies((v) => !v)}
          aria-label={speakReplies ? "Stop reading answers out loud" : "Read answers out loud"} disabled={!voiceOn}>
          {speakReplies && voiceOn ? <Volume2 /> : <VolumeX />}
        </Button>
        <Button variant="ghost" size="icon-sm" onClick={() => setPanel(panel === "settings" ? null : "settings")} aria-label="Voice settings" disabled={!voiceOn}>
          <SlidersHorizontal />
        </Button>

        <Panel open={panel === "people"} onClose={() => setPanel(null)} title="Talk as someone else">
          <div className="grid grid-cols-2 gap-2">
            {people.map((p) => (
              <a key={p.id} href={`/talk?user=${p.id}`}
                className={cn("flex items-center gap-2 rounded-lg border p-2 text-sm transition-colors hover:bg-accent", p.id === me.id && "border-primary bg-secondary")}>
                <span className="flex size-7 shrink-0 items-center justify-center rounded-full bg-primary text-[10px] font-semibold text-primary-foreground">{initialsOf(p.name)}</span>
                <span className="min-w-0">
                  <span className="block truncate font-medium">{p.name}</span>
                  <span className="block truncate text-[11px] text-muted-foreground">{roleWord(p.role)}</span>
                </span>
              </a>
            ))}
          </div>
        </Panel>
        <Panel open={panel === "tasks"} onClose={() => setPanel(null)} title="Your tasks" align="right">
          <PinnedList items={pinned} title="All tasks" empty="Nothing assigned to you right now." />
        </Panel>
        <Panel open={panel === "settings"} onClose={() => setPanel(null)} title="Voice" align="right">
          <IntensitySlider label="Voice speed" value={speed} min={0.7} max={1.2} step={0.05}
            unit="×" format={(v) => v.toFixed(2)} onChange={setSpeed} />
        </Panel>
      </div>

      <AnimatePresence>
        {fresh.length > 0 && (
          <motion.section role="alert" initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            className="mt-3 rounded-xl border border-destructive/40 bg-destructive-soft p-3">
            <p className="flex items-center gap-2 text-sm font-semibold text-destructive"><Siren className="size-4" />Aftershock near you</p>
            {fresh.map((q) => <p key={q.line} className="mt-1 text-xs text-foreground">{q.line}</p>)}
            <p className="mt-1 text-xs text-foreground">Leave damaged buildings and stay clear until the shaking stops.</p>
          </motion.section>
        )}
      </AnimatePresence>
      {tasks[0] && turns.length === 0 && (
        <button type="button" onClick={() => setPanel("tasks")}
          className="mt-3 flex items-center gap-2 self-center rounded-full border bg-card px-3 py-1.5 text-xs shadow-xs transition-colors hover:bg-accent">
          {tasks[0].tier === 0 ? <HeartPulse className="size-3.5 text-destructive" /> : <ClipboardList className="size-3.5 text-muted-foreground" />}
          <span className="text-muted-foreground">Top task</span>
          <span className="max-w-[16rem] truncate font-medium">{tasks[0].title}</span>
        </button>
      )}

      {/* The conversation, top to bottom. Older lines fade out under the top edge. */}
      <div ref={scroller} aria-live="polite"
        className="min-h-0 flex-1 overflow-y-auto py-6 [mask-image:linear-gradient(to_bottom,transparent,black_3rem,black_calc(100%-1.5rem),transparent)]">
        {turns.length === 0 && state === "idle" ? (
          <div className="flex h-full flex-col items-center justify-center gap-4 text-center">
            <p className="text-balance text-xl font-medium sm:text-2xl">What do you see, or what do you need?</p>
            <div className="flex flex-wrap justify-center gap-2">
              {SUGGEST.map((s) => (
                <button key={s} type="button" onClick={() => void submit(s)}
                  className="rounded-full border bg-card px-3 py-1.5 text-xs transition-colors hover:bg-accent">
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-6">
            {turns.map((t) => (
              <motion.div key={t.id} initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}
                transition={{ type: "spring", stiffness: 260, damping: 28 }}
                className={cn("flex flex-col gap-2", t.from === "person" && "items-end")}>
                {t.from === "person" ? (
                  <p className="max-w-[85%] rounded-3xl rounded-br-lg bg-secondary px-4 py-2.5 text-base text-secondary-foreground">
                    {t.voice && <Mic className="mr-1.5 mb-0.5 inline size-3.5 opacity-60" />}
                    {t.text}
                  </p>
                ) : (
                  <p className="text-lg leading-relaxed text-pretty sm:text-xl">{t.text}</p>
                )}
                {t.pending && !t.speaking && (
                  <SendCard pending={t.pending} onSend={() => void decide(t.id, t.pending!.id, true)} onDrop={() => void decide(t.id, t.pending!.id, false)} />
                )}
                {t.actions && t.actions.length > 0 && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }} className="flex flex-wrap gap-1.5">
                    {t.actions.map((a, i) => (
                      <span key={i} title={chipText(a.summary)}
                        className={cn("inline-flex max-w-full items-center gap-1 rounded-full border bg-card px-2 py-0.5 text-[11px]",
                          !a.ok && "border-destructive/40 text-destructive")}>
                        {a.ok ? <CircleCheck className="size-3 shrink-0 text-success" /> : <CircleX className="size-3 shrink-0" />}
                        <span className="font-medium">{a.ok ? TOOL_WORD[a.tool] ?? "Filed" : "Not filed"}</span>
                        <span className="truncate text-muted-foreground">{chipText(a.summary)}</span>
                      </span>
                    ))}
                  </motion.div>
                )}
              </motion.div>
            ))}
            {/* What is happening now, as the next line */}
            <AnimatePresence mode="popLayout">
              {state === "thinking" && lastFrom === "person" && (
                <motion.p key="thinking" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="flex items-center gap-1 text-lg text-muted-foreground">
                  <Dots />
                </motion.p>
              )}
              {state === "thinking" && lastFrom !== "person" && (
                <motion.p key="thinking2" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="flex items-center gap-1 self-end text-base text-muted-foreground">
                  Writing down what you said<Dots />
                </motion.p>
              )}
            </AnimatePresence>
          </div>
        )}
      </div>

      {/* The orb, and typing as a fallback */}
      <div className="flex flex-col items-center pb-[max(1rem,env(safe-area-inset-bottom))]">
        <VoiceOrb state={state} analyser={orbAnalyser} disabled={!voiceOn || state === "thinking"}
          onDown={() => void pressStart()} onUp={() => void pressEnd()} />
        <p className="h-5 text-sm text-muted-foreground" aria-live="polite">{status}</p>
        <p className="h-4 text-xs text-destructive">{voice.error ?? ""}</p>
        <div className="mt-1 flex w-full items-center justify-center">
          <AnimatePresence mode="wait" initial={false}>
            {typing ? (
              <motion.form key="type" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 6 }}
                onSubmit={(e) => { e.preventDefault(); void submit(typed); }} className="flex w-full max-w-md items-center gap-2">
                <Input autoFocus value={typed} onChange={(e) => setTyped(e.target.value)} placeholder="Type it…"
                  className="h-11 rounded-full px-4" aria-label="Message" />
                <Button type="submit" size="icon-lg" className="rounded-full" disabled={state === "thinking" || state === "listening" || !typed.trim()} aria-label="Send">
                  <SendHorizontal />
                </Button>
                {voiceOn && (
                  <Button type="button" variant="ghost" size="icon-lg" className="rounded-full" onClick={() => setTyping(false)} aria-label="Back to voice">
                    <Mic />
                  </Button>
                )}
              </motion.form>
            ) : (
              <motion.div key="voice" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <Button variant="ghost" size="sm" className="rounded-full text-muted-foreground" onClick={() => setTyping(true)}>
                  <Keyboard />Type instead
                </Button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </main>
  );
}

/** "Send this to the desk?" Shown once the answer has been read. Nothing is sent before the person taps Send. */
function SendCard({ pending, onSend, onDrop }: { pending: NonNullable<Turn["pending"]>; onSend: () => void; onDrop: () => void }) {
  const { state } = pending;
  if (state === "sent") return <p className="flex items-center gap-1.5 text-sm text-muted-foreground"><CircleCheck className="size-4 text-success" />Sent to the desk.</p>;
  if (state === "dropped") return <p className="flex items-center gap-1.5 text-sm text-muted-foreground"><CircleX className="size-4" />Not sent.</p>;
  const busy = state === "sending";
  return (
    <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} role="group" aria-label="Send this to the desk?"
      className="w-full rounded-2xl border bg-card p-3 shadow-xs">
      <p className="text-sm font-medium">Send this to the desk?</p>
      <ul className="mt-1.5 space-y-0.5 text-sm text-muted-foreground">
        {pending.items.map((it, i) => <li key={i}>{it.replaceAll("_", " ")}</li>)}
      </ul>
      {state === "error" && <p className="mt-2 text-xs text-destructive">{pending.error}</p>}
      <div className="mt-3 flex gap-2">
        <Button className="flex-1 rounded-full" disabled={busy} onClick={onSend}><SendHorizontal />{busy ? "Sending…" : "Send"}</Button>
        <Button variant="outline" className="flex-1 rounded-full" disabled={busy} onClick={onDrop}>Don&apos;t send</Button>
      </div>
    </motion.div>
  );
}

function Dots() {
  return (
    <span className="inline-flex gap-0.5" aria-hidden>
      {[0, 1, 2].map((i) => (
        <motion.span key={i} className="size-1.5 rounded-full bg-current"
          animate={{ opacity: [0.25, 1, 0.25] }} transition={{ duration: 1, repeat: Infinity, delay: i * 0.18 }} />
      ))}
    </span>
  );
}
