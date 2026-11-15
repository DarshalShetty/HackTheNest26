// One turn of a personal agent (Final system design §7):
// load the person's context from the desk → one Claude call with tools → run the tools
// (each posts one event) → loop until Claude answers → save both chat lines → reply.
// The agent keeps nothing between turns. Everything comes from the desk each time.

import type Anthropic from "@anthropic-ai/sdk";
import { desk, type UserContext } from "../desk";
import {
  age, countBy, incidentLine, NEAR_ROAD_METERS, needLine, roadLine, roadsNear, shelterLine, sheltersFor, siteLine, taskLine, type Place,
} from "../desk/view";
import { quakeLine, recentQuakesNear, type NearbyQuake } from "../research/quakes";
import { BudgetError } from "../shared/cost";
import { modelFor, send } from "../shared/llm";
import { maskPhones } from "../shared/privacy";
import { stashPending, takePending, type PendingCall } from "./pending";
import { systemPrompt } from "./prompts";
import { holdWrites, runTool, toolsFor, type ToolOutcome } from "./tools";

export interface TurnInput {
  userId: string;
  text?: string;
  image?: { mediaType: "image/jpeg" | "image/png" | "image/webp" | "image/gif"; base64: string };
  coords?: [number, number];
  /** Hold what the agent would file until the person says yes (the /talk screen). Without it, tools file at once. */
  confirm?: boolean;
}
export interface TurnResult {
  reply: string;
  eventIds: string[];
  // Set when confirm held writes: `id` goes back to POST /api/chat { approve }, `items` is what would be sent.
  pending?: { id: string; items: string[] };
  // For the screen: what the agent filed, and the charts it asked for (type "show_metric", id).
  actions: { tool: string; summary: string; ok: boolean; type?: "show_metric" | "show_map"; id?: string;
    kinds?: string[]; ids?: string[]; text?: string }[];
  model?: string;
}

const MAX_ROUNDS = 5;        // model calls per turn
const MAX_TOOL_CALLS = 8;    // "blankets, a doctor and a report" is 3; leave room, but stop loops

type Msg = Anthropic.Beta.BetaMessageParam;

export async function runTurn(input: TurnInput): Promise<TurnResult> {
  const d = desk();
  const ctx = await d.getUserContext(input.userId);
  if (input.coords) ctx.user = { ...ctx.user, coords: input.coords };
  const said = maskPhones(input.text?.trim() ?? "").text;
  if (!said && !input.image) return { reply: "Say or send something and I will take it from there.", eventIds: [], actions: [] };

  const { model, effort } = modelFor("chat");
  if (input.confirm) holdWrites(ctx);
  const held: (PendingCall & { summary: string })[] = [];
  let tools = toolsFor(ctx);
  const system: Anthropic.Beta.BetaTextBlockParam[] = [
    // Stable per role and shared by everyone in that role (tools + system are the prefix).
    // 1-hour cache: chats pause for more than 5 minutes, and the 5-minute cache then made
    // 16 of 20 measured turns rewrite this ~3.5k-token prefix. A 1-hour write costs 2× once
    // per role per hour; every turn after that reads it at 0.05×.
    { type: "text", text: systemPrompt(ctx.user.role), cache_control: { type: "ephemeral", ttl: "1h" } },
  ];
  // Aftershocks near this person (scenario clock during a replay). Plain code, no model call.
  const quakes = await recentQuakesNear(ctx.user.coords, 60).catch(() => undefined);
  const past = history(ctx);
  // Breakpoint at the end of the history: the next turn starts with the same lines plus two
  // more, so within 5 minutes it reads them from the cache instead of writing them again.
  const last = past[past.length - 1];
  if (last && typeof last.content === "string") {
    last.content = [{ type: "text", text: last.content, cache_control: { type: "ephemeral" } }];
  }
  const messages: Msg[] = [...past, { role: "user", content: userContent(ctx, said, input.image, quakes) }];

  const eventIds: string[] = [];
  const actions: TurnResult["actions"] = [];
  let toolCalls = 0;
  let searches = 0;
  let reply = "";
  let servedBy: string | undefined;

  try {
    for (let round = 0; round < MAX_ROUNDS; round++) {
      const res = await send(`chat:${ctx.user.role}`, {
        model, max_tokens: 8000, system, tools, messages, output_config: { effort },
        // Later rounds of the same turn re-send everything: cache the growing tail too.
        cache_control: { type: "ephemeral" },
      });
      servedBy = res.model;
      searches += res.usage.server_tool_use?.web_search_requests ?? 0;
      const text = res.content.filter((b) => b.type === "text").map((b) => b.text).join("").trim();
      reply = text || reply;   // never let an empty last round wipe an earlier answer

      if (res.stop_reason === "refusal") {
        reply = "I can't help with that one here. If someone is in danger, call 112.";
        break;
      }
      if (res.stop_reason === "pause_turn") {           // web search ran long: resume as is
        messages.push({ role: "assistant", content: res.content });
        continue;
      }
      const calls = res.content.filter((b): b is Anthropic.Beta.BetaToolUseBlock => b.type === "tool_use");
      if (res.stop_reason !== "tool_use" || calls.length === 0) break;

      messages.push({ role: "assistant", content: res.content });
      // Every tool_use needs a tool_result in the same message. Past the cap, answer with an error.
      const results = await Promise.all(calls.map(async (c): Promise<Anthropic.Beta.BetaToolResultBlockParam> => {
        if (toolCalls >= MAX_TOOL_CALLS) {
          actions.push({ tool: c.name, summary: "skipped: too many actions in one message", ok: false });
          return { type: "tool_result", tool_use_id: c.id, is_error: true,
            content: "Skipped: too many actions in one message. Tell the person what was not filed and ask them to repeat it." };
        }
        toolCalls++;
        const out = await runTool(c.name, (c.input ?? {}) as Record<string, unknown>, ctx)
          .catch((e: Error): ToolOutcome => ({ content: `Tool failed: ${e.message}`, isError: true }));
        if (out.eventId) eventIds.push(out.eventId);
        if (out.pending) held.push({ name: c.name, input: (c.input ?? {}) as Record<string, unknown>, summary: out.summary ?? c.name });
        if (c.name !== "ask_desk") actions.push({ tool: c.name, summary: out.summary ?? out.content.slice(0, 120), ok: !out.isError,
          ...("showMetric" in out && out.showMetric ? { type: "show_metric" as const, id: out.showMetric } : {}),
          ...("showMap" in out && out.showMap ? { type: "show_map" as const, ...out.showMap } : {}) });
        return { type: "tool_result", tool_use_id: c.id, content: out.content, is_error: out.isError || undefined };
      }));
      messages.push({ role: "user", content: results });
      // Help-desk web search: at most 2 per turn (max_uses counts per request, not per turn).
      if (searches >= 2) tools = tools.filter((t) => !("type" in t && t.type === "web_search_20260209"));
    }
  } catch (e) {
    if (e instanceof BudgetError) {
      reply = "The agent is paused: the team's AI budget is used up. Your task buttons still work.";
    } else if (eventIds.length === 0) {
      throw e;
    } else {
      // Something was already filed: report it rather than fail, so the person does not repeat it.
      console.error("[agent turn] error after filing", e);
      reply = "";
    }
  }

  if (!reply) {
    const done = actions.filter((a) => a.ok).map((a) => a.summary);
    reply = done.length ? `${held.length ? "Ready to send" : "Filed"}: ${done.join("; ")}.` : "Sorry, I could not finish that. Try again in a moment.";
  }
  const now = new Date();
  await d.saveMessage({ userId: ctx.user._id, from: "user", text: said || "(photo)", ts: now });
  await d.saveMessage({ userId: ctx.user._id, from: "agent", text: reply, ts: new Date(+now + 1), eventIds });
  // Held writes are not "done": they leave the chips and wait for the person's yes.
  const pending = held.length
    ? { id: stashPending(ctx.user._id, held.map(({ name, input }) => ({ name, input }))), items: held.map((h) => h.summary) }
    : undefined;
  return { reply, eventIds, actions: pending ? actions.filter((a) => !held.some((h) => h.summary === a.summary && h.name === a.tool)) : actions,
    ...(pending ? { pending } : {}), model: servedBy };
}

/**
 * The person said yes (send) or no to what a confirm turn held. Sends it through the same tools,
 * so every check runs again on the desk as it is now. Returns null when nothing is waiting any more.
 */
export async function settlePending(userId: string, id: string, send: boolean): Promise<Pick<TurnResult, "eventIds" | "actions"> & { sent: boolean } | null> {
  const calls = takePending(id, userId);
  if (!calls) return null;
  const d = desk();
  const now = new Date();
  if (!send) {
    await d.saveMessage({ userId, from: "user", text: "(did not send it)", ts: now });
    return { sent: false, eventIds: [], actions: [] };
  }
  const ctx = await d.getUserContext(userId);
  const eventIds: string[] = [];
  const actions: TurnResult["actions"] = [];
  for (const c of calls) {
    const out = await runTool(c.name, c.input, ctx)
      .catch((e: Error): ToolOutcome => ({ content: `Tool failed: ${e.message}`, isError: true }));
    if (out.eventId) eventIds.push(out.eventId);
    actions.push({ tool: c.name, summary: out.summary ?? out.content.slice(0, 120), ok: !out.isError });
  }
  const done = actions.filter((a) => a.ok).map((a) => a.summary);
  await d.saveMessage({ userId, from: "user", text: "Yes, send it.", ts: now });
  await d.saveMessage({ userId, from: "agent", ts: new Date(+now + 1), eventIds,
    text: done.length ? `Sent to the desk: ${done.join("; ")}.` : "I could not send that. Please say it again." });
  return { sent: true, eventIds, actions };
}

/** Earlier chat lines as plain alternating turns. Must start with a user turn. */
function history(ctx: UserContext): Msg[] {
  const out: Msg[] = [];
  for (const m of ctx.messages) {
    const role = m.from === "user" ? "user" : "assistant";
    const last = out[out.length - 1];
    if (last && last.role === role) last.content = `${last.content as string}\n${m.text}`;
    else out.push({ role, content: m.text });
  }
  while (out.length && out[0].role !== "user") out.shift();
  // The new user turn comes next, so history must end with the assistant.
  if (out.length && out[out.length - 1].role === "user") out.pop();
  return out;
}

type Quakes = { quakes: NearbyQuake[]; now: Date; replay: boolean } | undefined;

function userContent(ctx: UserContext, said: string, image: TurnInput["image"], quakes: Quakes): Anthropic.Beta.BetaContentBlockParam[] {
  const blocks: Anthropic.Beta.BetaContentBlockParam[] = [{ type: "text", text: contextBlock(ctx, quakes) }];
  if (image) blocks.push({ type: "image", source: { type: "base64", media_type: image.mediaType, data: image.base64 } });
  blocks.push({ type: "text", text: `${ctx.user.name} says: ${said || "(sent a photo, no text)"}` });
  return blocks;
}

const NEAR_KM = NEAR_ROAD_METERS / 1000;

/** Where this person's open tasks take them: the site, shelter or incident a task points to. */
function taskPlaces(ctx: UserContext): Place[] {
  const out: Place[] = [];
  const sites = new Map(ctx.records.sites.map((s) => [s._id, s]));
  const shelters = new Map(ctx.records.shelters.map((s) => [s._id, s]));
  const incidents = new Map(ctx.myIncidents.map((i) => [i._id, i]));
  for (const t of ctx.tasks) {
    if (t.type === "call" || t.type === "notify") continue;   // done by phone: roads do not matter
    const label = `task "${maskPhones(t.title).text}"`;
    const need = t.recordType === "need" ? ctx.records.needs.find((n) => n._id === t.recordId) : undefined;
    const coords =
      t.recordType === "site" ? sites.get(t.recordId)?.coords :
      t.recordType === "shelter" ? shelters.get(t.recordId)?.coords :
      t.recordType === "incident" ? incidents.get(t.recordId)?.coords :
      need ? need.siteIds.map((id) => sites.get(id)?.coords).find(Boolean) : undefined;
    if (coords) out.push({ label, coords });
  }
  return out;
}

/** The volatile part of the prompt: this person's slice of the desk, right now. */
export function contextBlock(ctx: UserContext, quakes?: Quakes): string {
  const { user: u, now } = ctx;
  const p = u.profile ?? {};
  const lines: string[] = [
    `<desk_context generated="${now.toISOString()}">`,
    `Person: ${u.name} (id ${u._id}) · role ${u.role} · trust ${u.trust} · region ${u.region}` +
      (u.coords ? ` · position ${u.coords[0].toFixed(3)},${u.coords[1].toFixed(3)}` : ""),
    `Profile: skills ${p.skills?.join(", ") || "none"} · vehicle ${p.vehicle ?? "unknown"}` +
      (p.stock ? ` · stock ${Object.entries(p.stock).map(([k, v]) => `${v} ${k}`).join(", ")}` : "") +
      (p.notes ? ` · notes: ${p.notes}` : ""),
  ];
  // A crowded person (a coordinator with hundreds of tasks) must not blow up every prompt. Each
  // section shows its top lines in full, then one line that counts the rest by kind, computed
  // here in code (exact and free, no model summary). The agent pages through the rest with ask_desk.
  const section = <T>(title: string, xs: T[], line: (x: T) => string, cap: number,
    rest?: (more: T[]) => string, topic?: string) => {
    lines.push(`${title}:`);
    if (!xs.length) { lines.push("  none"); return; }
    lines.push(...xs.slice(0, cap).map((x) => `  ${line(x)}`));
    if (xs.length > cap) {
      const more = xs.slice(cap);
      lines.push(`  ...and ${more.length} more not shown${rest ? ` (${rest(more)})` : ""}` +
        (topic ? `; ask_desk topic "${topic}" lists them.` : "."));
    }
  };
  const overdue = (ts: { deadline?: Date | string }[]) => ts.filter((t) => t.deadline && +new Date(t.deadline) < +now).length;
  const taskRest = (more: typeof ctx.tasks) => `${countBy(more, (t) => t.type)}; ${countBy(more, (t) => `tier ${t.tier}`)}` +
    (overdue(more) ? `; ${overdue(more)} overdue` : "");

  section("Their open tasks, top first", ctx.tasks, (t) => taskLine(t, now, { showDetail: true }), 8, taskRest, "mine");
  if (u.role === "coordinator") {
    section("Open incidents, most severe first", ctx.openIncidents, (i) => incidentLine(i, now), 10,
      (more) => `${countBy(more, (i) => i.severity)}; ${countBy(more, (i) => i.kind)}`, "incidents");
    section("Waiting for their approval", ctx.approvals, (t) => taskLine(t, now, { showDetail: true }), 8, taskRest, "summary");
    section("Stuck, nobody took them (assign_task gives one to a person)", ctx.stuck, (t) => taskLine(t, now, { showDetail: true }), 8, taskRest, "tasks");
  }
  section("Needs they filed", ctx.myNeeds, (n) => needLine(n, now), 6, (more) => countBy(more, (n) => n.item), "mine");
  section("Their open problems", ctx.myIncidents, (i) => incidentLine(i, now), 6, (more) => countBy(more, (i) => i.status), "mine");
  if (u.role !== "public") {
    const touched: string[] = [
      ...ctx.records.sites.map((s) => siteLine(s, now, u.role)),
      ...ctx.records.needs.map((n) => needLine(n, now)),
    ];
    section("Records their tasks touch", touched, (x) => x, 8);
  }
  section("Roads not open", ctx.records.roads, (r) => roadLine(r, now), 8, (more) => countBy(more, (r) => r.state), "roads");
  const free = (s: (typeof ctx.records.shelters)[number]) => s.bedsTotal - s.bedsUsed;
  const shelters = sheltersFor(ctx.records.shelters, u.coords);
  if (u.role === "shelter" || u.role === "public") {
    section(u.coords ? "Shelters, free beds first, then nearest" : "Shelters, free beds first", shelters,
      (s) => shelterLine(s, now, u.coords), 8,
      (more) => `${more.filter((s) => free(s) > 0).length} with free beds, ${more.reduce((n, s) => n + Math.max(0, free(s)), 0)} free beds in total`,
      "shelters");
  }
  // Places this person is heading to: their task places, and for a citizen the shelters they
  // would be sent to. Closed roads near them, worked out here in code, not by the model.
  const places: Place[] = u.role === "public"
    ? shelters.filter((s) => free(s) > 0 && s.coords).slice(0, 3).map((s) => ({ label: s.name, coords: s.coords! }))
    : taskPlaces(ctx);
  if (places.length) {
    section(`Roads not open within ${NEAR_KM} km of ${u.role === "public" ? "those shelters" : "their task places"}`,
      roadsNear(ctx.records.roads, places, now), (x) => x, 5);
  }
  if (quakes) {
    lines.push(quakes.replay ? `Scenario time (replay): ${quakes.now.toISOString().slice(0, 16)} UTC` : "Earthquake feed: live (AFAD, USGS)");
    section("Earthquakes worth a warning near this person, last 60 min, newest first", quakes.quakes, quakeLine, 5);
  }
  section("Unread inbox", ctx.inbox, (i) => `${i.kind} · ${i.message}${i.taskId ? ` (task ${i.taskId})` : ""} · ${age(i.ts, now)}`, 8,
    (more) => countBy(more, (i) => i.kind));
  section("Already filed this session by this agent", ctx.postedThisSession, (e) =>
    `${e.type} ${e._id} · ${JSON.stringify(e.payload).slice(0, 140)} · ${age(e.ts, now)}`, 6);
  if (ctx.chatSummary) {
    lines.push(`Earlier in this chat, summarized (the lines after it are word for word): ${ctx.chatSummary}`);
  }
  lines.push("</desk_context>");
  return lines.join("\n");
}
