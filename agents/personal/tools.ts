// The personal agent's tools. Every write tool posts exactly one event and nothing else:
// the memory keeper (Role 1) is the only writer of records. ask_desk only reads.
// Schemas are strict: every key required, null for "not said".
// Inputs are checked again here (task ids, roles), because the model's output is untrusted.

import { pilot } from "@/lib/pilot";
import type Anthropic from "@anthropic-ai/sdk";
import type {
  AnswerPayload, ApprovalPayload, CapacityPayload, ClaimPayload, ClosurePayload, CorrectionPayload,
  IncidentKind, IncidentPayload, LatLng, NeedPayload, RecordType, ReportPayload, Role,
} from "@/lib/types";
import { deskNow, observeBoard } from "@/components/clock";
import { METRIC_IDS, metric } from "@/components/metrics/registry";
import type { MetricId } from "@/components/metrics/types";
import { desk, type NewEvent, type UserContext } from "../desk";
import { haversineMeters } from "@/lib/geo";
import { age, askDesk, distance, topicsFor, type Topic } from "../desk/view";
import { maskPhones } from "../shared/privacy";
import { writeHandoff } from "./handoff";

type Tool = Anthropic.Beta.BetaTool;
const str = { type: "string" } as const;
const nstr = { type: ["string", "null"] } as const;
const nint = { type: ["integer", "null"] } as const;

function obj(properties: Record<string, unknown>): Tool["input_schema"] {
  return { type: "object", additionalProperties: false, required: Object.keys(properties), properties } as Tool["input_schema"];
}

const INCIDENT_KINDS: IncidentKind[] = ["vehicle_breakdown", "road_blocked", "cannot_find", "short_supply",
  "unsafe", "injury", "handover", "unreachable", "other"];
const RECORD_TYPES: RecordType[] = ["site", "need", "road", "shelter", "zone", "incident"];

const MAP_KINDS = ["site", "shelter", "road", "need", "incident", "task"] as const;
type MapKind = (typeof MAP_KINDS)[number];

const DEFS: Record<string, Tool> = {
  ask_desk: {
    name: "ask_desk",
    description: "Read the shared desk. Use it for any question about the crisis: sites, needs, shelters with space, closed roads, incidents, tasks, or a summary of what happened and what is stuck. Topic \"mine\" lists this person's own open tasks, needs and problems, for when <desk_context> says more are not shown. Read only: it files nothing. Every line it returns has its source and age; quote them. Answers come 12 lines at a time; ask again with the next page for more.",
    input_schema: obj({
      topic: { type: "string", enum: ["summary", "mine", "sites", "needs", "shelters", "roads", "incidents", "tasks"] },
      region: { ...nstr, description: "District or neighbourhood to filter by, or null for all." },
      // Plain integer, not nullable: the API allows at most 16 nullable parameters across all of a
      // request's strict tools, and the volunteer and field tool sets were at 17 (HTTP 400).
      page: { type: "integer", description: "0 for the first 12 lines, 1 for the next 12, and so on." },
    }),
  },
  answer_task: {
    name: "answer_task",
    description: "Answer one of the person's own tasks: accepted (they will do it), declined (they cannot), or done (finished). Use the task id from the context. For a volunteer's call, put what they heard in note.",
    input_schema: obj({
      task_id: str,
      result: { type: "string", enum: ["accepted", "declined", "done"] },
      note: { ...str, description: "What they said about it, or an empty string. (Not nullable: see page above.)" },
      eta_minutes: { ...nint, description: "Minutes until done, if they said it." },
    }),
  },
  file_need: {
    name: "file_need",
    description: "File one need the person has or reports: supplies, equipment, food, medicine, a doctor, fuel. One call per item. Use the total quantity they need now.",
    input_schema: obj({
      item: { ...str, description: "Short English name, plural if counted: blankets, tent, insulin, doctor, jackhammer, diapers, food for the team." },
      quantity: { type: "integer", description: "How many. 1 if they did not say a number." },
      life_threatening: { type: "boolean" },
      site_id: { ...nstr, description: "Only a site id that appears in the context." },
      region: nstr,
    }),
  },
  file_report: {
    name: "file_report",
    description: "File what the person saw about people: someone trapped, missing or rescued, or news about a collapse site.",
    input_schema: obj({
      kind: { type: "string", enum: ["trapped", "missing", "rescued", "site_update"] },
      text: { ...str, description: "The person's own words, in their language, unchanged." },
      text_en: { ...str, description: "English translation." },
      address: nstr,
      region: nstr,
      people_count: nint,
      site_id: { ...nstr, description: "Only a site id that appears in the context." },
    }),
  },
  file_closure: {
    name: "file_closure",
    description: "File the state of a road: closed, open again, or unconfirmed (someone heard it is blocked).",
    input_schema: obj({
      road: str,
      state: { type: "string", enum: ["closed", "open", "unconfirmed"] },
      reason: { ...str, description: "Why, in English: collapsed building, crash, debris, traffic." },
    }),
  },
  file_capacity: {
    name: "file_capacity",
    description: "File a shelter's beds: used and total. From the person's words or a readable photo of the whiteboard.",
    input_schema: obj({ shelter: str, beds_used: { type: "integer" }, beds_total: { type: "integer" } }),
  },
  report_problem: {
    name: "report_problem",
    description: "The help desk. Use when something goes wrong while the person is on a task (vehicle broke down, road blocked, cannot find the place, short of stock, unsafe, someone hurt, shift ending), or to say an open problem is over (resolved true with its incident_id).",
    input_schema: obj({
      kind: { type: "string", enum: INCIDENT_KINDS },
      text: { ...str, description: "The person's own words, in their language, unchanged." },
      text_en: { ...str, description: "English translation." },
      task_id: { ...nstr, description: "The task in trouble, from the context." },
      can_continue: { type: "boolean", description: "Can they still finish the task themselves?" },
      can_deliver: { ...nint, description: "short_supply only: how many they can still bring." },
      resolved: { type: "boolean", description: "true only to close an open problem." },
      incident_id: { ...nstr, description: "Set when updating or resolving an open problem from the context." },
    }),
  },
  declare_work: {
    name: "declare_work",
    description: "The person says they are already doing something nobody assigned: 'I'm taking water to the school', 'we started digging at Çınar St.'. The desk records it so nobody else does it twice.",
    input_schema: obj({
      what: { type: "string", enum: ["call", "rescue", "deliver", "assist"] },
      text: { ...str, description: "The person's own words, in their language, unchanged." },
      text_en: { ...str, description: "English translation." },
      record_id: { ...nstr, description: "Only an id that appears in the context." },
      eta_minutes: nint,
    }),
  },
  correct_record: {
    name: "correct_record",
    description: "The person says the desk is wrong: a report is false (retract), or one value is wrong (edit one field).",
    input_schema: obj({
      action: { type: "string", enum: ["retract", "edit"] },
      record_type: { type: "string", enum: RECORD_TYPES },
      record_id: str,
      // Nullable enum: the API wants anyOf here, not type ["string","null"] + enum.
      field: { anyOf: [{ type: "string", enum: ["address", "peopleTrapped", "quantity", "bedsUsed", "bedsTotal", "state"] }, { type: "null" }] },
      value: { ...nstr, description: "The new value for field, as text." },
      reason: str,
    }),
  },
  decide_approval: {
    name: "decide_approval",
    description: "Coordinator only. Approve or reject a task that waits for approval. Only when the coordinator clearly said so.",
    input_schema: obj({ task_id: str, approved: { type: "boolean" }, reason: str }),
  },
  assign_task: {
    name: "assign_task",
    description: "Coordinator only. Give a task to a named person: a stuck task nobody took, or a task waiting for approval (this also approves it). Use when the coordinator says who should do it ('give it to Ayşe', 'send the mechanic'). Use the task id from the context and the person's user id. If you do not know who fits, call it with an empty person_id: the result lists who can take it.",
    input_schema: obj({ task_id: str, person_id: str }),
  },
  shift_handoff: {
    name: "shift_handoff",
    description: "Coordinator only. Write the shift handoff note for whoever takes over: what was done, what is open, what is stuck.",
    input_schema: obj({}),
  },
  visualize_metric: {
    name: "visualize_metric",
    description: "Show a chart on the person's screen. Call it whenever they ask something quantitative about the desk: how many, how full, how fast, how fresh, how sure, how far, or a status report. At most 2 per turn. It returns the chart's headline, basis, source and status. Do not repeat those numbers in long form: one sentence is enough, the chart carries the rest.",
    input_schema: obj({
      metric: { type: "string", enum: METRIC_IDS },
      why: { ...str, description: "One short line: what the person asked that this chart answers." },
    }),
  },
  show_on_map: {
    name: "show_on_map",
    description: "Show places on the person's map. Call it when they ask where something is, where something started or hit first, want locations, the nearest thing, or anything on a map. At most 1 per turn. The map flies to the first match and selects it. It returns what matched in that order, with distance when near_me is true and first-report time when order is set. Explain in one or two sentences where the map is pointing and why that place comes first.",
    input_schema: obj({
      kinds: { type: "array", items: { type: "string", enum: MAP_KINDS }, description: "What to show: one or more kinds. A fire, gas leak or falling wall is a site hazard: use site." },
      text: { ...str, description: "Words the name, address, item, region or hazard (fire, gas...) must contain, or an empty string for all. (Not nullable: see ask_desk page.)" },
      status: { ...str, description: "Only records in this state (open, closed, free, assigned...), or an empty string for any." },
      near_me: { type: "boolean", description: "Sort by distance from the person." },
      order: { type: "string", enum: ["any", "earliest", "latest"], description: "Sort by when the desk first heard of each place. earliest for 'where did it start', 'first hit', 'first reported'. latest for 'newest'. any keeps board order, or distance with near_me." },
      why: { ...str, description: "One short line: what the person asked that this map answers." },
    }),
  },
};

/** Who can take each task type. Mirrors the dispatcher's ROLE_FOR (workers/dispatcher.ts). */
const ROLE_FOR_TASK: Record<string, Role[]> = {
  call: ["volunteer"], rescue: ["field"], deliver: ["supplier"],
  assist: ["volunteer", "field"], notify: ["volunteer"], evacuate: ["field"],
};

const ROLE_TOOLS: Record<Role, string[]> = {
  volunteer: ["ask_desk", "answer_task", "file_need", "file_report", "file_closure", "report_problem", "declare_work", "correct_record", "visualize_metric", "show_on_map"],
  field: ["ask_desk", "answer_task", "file_need", "file_report", "file_closure", "report_problem", "declare_work", "correct_record", "visualize_metric", "show_on_map"],
  supplier: ["ask_desk", "answer_task", "file_need", "file_closure", "report_problem", "declare_work", "visualize_metric", "show_on_map"],
  shelter: ["ask_desk", "answer_task", "file_need", "file_report", "file_capacity", "report_problem", "correct_record", "visualize_metric", "show_on_map"],
  coordinator: ["ask_desk", "decide_approval", "assign_task", "shift_handoff", "file_report", "file_closure", "file_need", "correct_record", "visualize_metric", "show_on_map"],
  public: ["ask_desk", "file_need", "file_report", "file_closure"],
};

export function toolsFor(ctx: UserContext): Anthropic.Beta.BetaToolUnion[] {
  const tools: Anthropic.Beta.BetaToolUnion[] = ROLE_TOOLS[ctx.user.role].map((n) => ({ ...DEFS[n], strict: true }));
  // Web search only when a supplier or field team has an open problem: the help-desk lookup.
  // Billed per search, so it is offered only then, and capped at 2 searches per turn.
  if ((ctx.user.role === "supplier" || ctx.user.role === "field") && ctx.myIncidents.length > 0) {
    tools.push({ type: "web_search_20260209", name: "web_search", max_uses: 2,
      user_location: { type: "approximate", city: pilot.name, country: "TR" } });
  }
  return tools;
}

export interface ToolOutcome { content: string; eventId?: string; isError?: boolean; summary?: string; pending?: true; showMetric?: MetricId; showMap?: { kinds: string[]; ids: string[]; text?: string } }

// show_map calls per turn: a turn builds one context, so this caps the map at 1 per turn.
const mapCalls = new WeakMap<UserContext, number>();

class ToolInputError extends Error {}

// Writes held for the person's OK (the /talk screen asks "send this?"). A held tool call runs all its
// checks, then stops short of posting: it returns `pending` and the caller keeps the call to run later.
const held = new WeakSet<UserContext>();
export function holdWrites(ctx: UserContext): void { held.add(ctx); }
const HELD_NOTE = "Held for the person's yes, which the screen asks for after your reply. " +
  "Do not mention confirming, sending, filing or being ready. Reply as if it were on the desk: say what you understood and what they do next, in the present tense.";

/** Runs one tool call for one person. Never throws for bad input: returns an error result instead. */
export async function runTool(name: string, input: Record<string, unknown>, ctx: UserContext): Promise<ToolOutcome> {
  const role = ctx.user.role;
  if (!ROLE_TOOLS[role].includes(name)) return { content: `${name} is not available to the ${role} role.`, isError: true };
  try {
    return await run(name, input, ctx);
  } catch (e) {
    if (e instanceof ToolInputError) return { content: e.message, isError: true };
    throw e;
  }
}

const s = (v: unknown) => (typeof v === "string" && v.trim() ? v.trim() : undefined);
const n = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : undefined);
const masked = (v: unknown) => (s(v) ? maskPhones(s(v)!).text : undefined);

function needId<T>(v: T | undefined, what: string): T {
  if (v === undefined) throw new ToolInputError(`${what} is required.`);
  return v;
}

async function post(ctx: UserContext, type: NewEvent["type"], payload: unknown, summary: string): Promise<ToolOutcome> {
  if (held.has(ctx)) return { content: `Ready: ${summary}. ${HELD_NOTE}`, summary, pending: true };
  const id = await desk().postEvent({ deskId: ctx.user.deskId, type, source: "person", actor: ctx.user._id, payload,
    observedAt: new Date().toISOString() });
  return { content: `Filed as event ${id}. ${summary}. The desk processes it in a few seconds.`, eventId: id, summary };
}

async function run(name: string, i: Record<string, unknown>, ctx: UserContext): Promise<ToolOutcome> {
  const u = ctx.user;
  switch (name) {
    case "ask_desk": {
      const topic = s(i.topic) as Topic;
      if (!topicsFor(u.role).includes(topic)) {
        return { content: `Not available to the ${u.role} role. Allowed: ${topicsFor(u.role).join(", ")}.`, isError: true };
      }
      const d = desk();
      const [board, events] = await Promise.all([d.getBoard(u.deskId), d.recentEvents(u.deskId, 60)]);
      return { content: askDesk({ board, events, role: u.role, topic, region: s(i.region), now: ctx.now, near: u.coords,
        userId: u._id, page: n(i.page) }) };
    }

    case "answer_task": {
      const task = ctx.tasks.find((t) => t._id === s(i.task_id));
      if (!task) throw new ToolInputError(`Task ${String(i.task_id)} is not one of this person's open tasks: ${ctx.tasks.map((t) => t._id).join(", ") || "none"}.`);
      const p: AnswerPayload = { taskId: task._id, result: i.result as AnswerPayload["result"],
        note: masked(i.note), etaMinutes: n(i.eta_minutes) };
      return post(ctx, "answer", p, `Task "${task.title}" ${p.result}`);
    }

    case "file_need": {
      const quantity = Math.max(1, Math.round(needId(n(i.quantity), "quantity")));
      const siteId = s(i.site_id);
      const p: NeedPayload = { item: needId(s(i.item), "item"), quantity, lifeThreatening: i.life_threatening === true,
        siteId: siteId && ctx.records.sites.some((x) => x._id === siteId) ? siteId : undefined,
        region: s(i.region) ?? u.region, coords: u.coords };
      const wait = u.trust === "public" && !p.lifeThreatening
        ? " It stays 'proposed' until three different people ask for it." : "";
      return post(ctx, "need", p, `Need: ${quantity} ${p.item}${p.lifeThreatening ? ", life-threatening" : ""}${wait}`);
    }

    case "file_report": {
      const siteId = s(i.site_id);
      const p: ReportPayload = { kind: i.kind as ReportPayload["kind"], text: needId(masked(i.text), "text"),
        textEn: masked(i.text_en) ?? "", address: s(i.address), region: s(i.region) ?? u.region,
        peopleCount: n(i.people_count),
        siteId: u.role !== "public" && siteId && ctx.records.sites.some((x) => x._id === siteId) ? siteId : undefined };
      return post(ctx, "report", p, `Report: ${p.kind}${p.address ? ` at ${p.address}` : ""}`);
    }

    case "file_closure": {
      const p: ClosurePayload = { road: needId(s(i.road), "road"), state: i.state as ClosurePayload["state"],
        reason: s(i.reason) ?? "" };
      return post(ctx, "closure", p, `Road ${p.road}: ${p.state}`);
    }

    case "file_capacity": {
      const used = needId(n(i.beds_used), "beds_used"), total = needId(n(i.beds_total), "beds_total");
      if (used < 0 || total <= 0) throw new ToolInputError("beds_used must be ≥ 0 and beds_total > 0.");
      const p: CapacityPayload = { shelter: needId(s(i.shelter), "shelter"), bedsUsed: used, bedsTotal: total };
      return post(ctx, "capacity", p, `Shelter ${p.shelter}: ${used}/${total}`);
    }

    case "report_problem": {
      const taskId = s(i.task_id);
      if (taskId && !ctx.tasks.some((t) => t._id === taskId)) {
        throw new ToolInputError(`Task ${taskId} is not one of this person's open tasks.`);
      }
      const incidentId = s(i.incident_id);
      if (i.resolved === true && !incidentId) throw new ToolInputError("resolved needs the incident_id of an open problem.");
      if (incidentId && !ctx.myIncidents.some((x) => x._id === incidentId)) {
        throw new ToolInputError(`Incident ${incidentId} is not an open problem of this person.`);
      }
      // With one open task the problem is about it. With several, the model must say which.
      let forTask = taskId;
      if (!forTask && !incidentId && i.resolved !== true) {
        if (ctx.tasks.length === 1) forTask = ctx.tasks[0]._id;
        else if (ctx.tasks.length > 1 && i.can_continue !== true) {
          throw new ToolInputError(`Which task is in trouble? Ask the person. Open tasks: ${ctx.tasks.map((t) => `${t._id} (${t.title})`).join("; ")}.`);
        }
      }
      const p: IncidentPayload = { kind: i.kind as IncidentKind, text: needId(masked(i.text), "text"),
        textEn: masked(i.text_en) ?? "", taskId: forTask,
        canContinue: i.can_continue === true, coords: u.coords, canDeliver: n(i.can_deliver),
        resolved: i.resolved === true ? true : undefined, incidentId };
      return post(ctx, "incident", p, p.resolved ? `Problem ${incidentId} resolved` : `Problem: ${p.kind}${p.taskId ? ` on task ${p.taskId}` : ""}`);
    }

    case "declare_work": {
      if (u.trust === "public") throw new ToolInputError("Only vetted or official people can declare work.");
      const rid = s(i.record_id);
      const known = rid && ([...ctx.records.sites, ...ctx.records.needs].some((x) => x._id === rid) ||
        ctx.tasks.some((t) => t.recordId === rid));
      const p: ClaimPayload = { what: i.what as ClaimPayload["what"], text: needId(masked(i.text), "text"),
        textEn: masked(i.text_en) ?? "", recordId: known ? rid : undefined, etaMinutes: n(i.eta_minutes) };
      return post(ctx, "claim", p, `Declared work: ${p.what}`);
    }

    case "correct_record": {
      const field = s(i.field);
      if (i.action === "edit" && (!field || s(i.value) === undefined)) throw new ToolInputError("edit needs field and value.");
      const raw = s(i.value);
      const value = raw !== undefined && /^-?\d+$/.test(raw) ? Number(raw) : raw;
      const p: CorrectionPayload = { action: i.action as "retract" | "edit", recordType: i.record_type as RecordType,
        recordId: needId(s(i.record_id), "record_id"), fields: field ? { [field]: value } : undefined,
        reason: needId(s(i.reason), "reason") };
      return post(ctx, "correction", p, `Correction: ${p.action} ${p.recordType} ${p.recordId}`);
    }

    case "decide_approval": {
      if (u.role !== "coordinator") throw new ToolInputError("Only a coordinator can approve.");
      const task = ctx.approvals.find((t) => t._id === s(i.task_id));
      if (!task) throw new ToolInputError(`Task ${String(i.task_id)} is not waiting for approval. Waiting: ${ctx.approvals.map((t) => t._id).join(", ") || "none"}.`);
      const p: ApprovalPayload = { taskId: task._id, approved: i.approved === true };
      return post(ctx, "approval", p, `${p.approved ? "Approved" : "Rejected"}: ${task.title}`);
    }

    case "assign_task": {
      if (u.role !== "coordinator") throw new ToolInputError("Only a coordinator can assign a task.");
      const open = [...ctx.stuck, ...ctx.approvals];
      const task = open.find((t) => t._id === s(i.task_id));
      if (!task) {
        throw new ToolInputError(`Task ${String(i.task_id)} is not stuck or waiting for approval. ` +
          `Stuck: ${ctx.stuck.map((t) => t._id).join(", ") || "none"}. Waiting: ${ctx.approvals.map((t) => t._id).join(", ") || "none"}.`);
      }
      const people = (await desk().listUsers()).filter((p) => p.deskId === u.deskId && p.status !== "offline" && ROLE_FOR_TASK[task.type].includes(p.role));
      const who = people.find((p) => p._id === s(i.person_id));
      if (!who) {
        const list = people.map((p) => `${p._id} (${p.name}, ${p.role}${p.lastSeen && +ctx.now - +new Date(p.lastSeen) <= 3 * 60_000 ? ", on the app now" : ""})`);
        throw new ToolInputError(`${s(i.person_id) ? `${String(i.person_id)} cannot take a ${task.type} task. ` : ""}` +
          `Who can: ${list.join("; ") || `nobody, no ${ROLE_FOR_TASK[task.type].join(" or ")} is available`}. Ask the coordinator which one.`);
      }
      const p: ApprovalPayload = { taskId: task._id, approved: true, assignee: who._id };
      return post(ctx, "approval", p, `Assigned to ${who.name}: ${task.title}`);
    }

    case "shift_handoff": {
      if (u.role !== "coordinator") throw new ToolInputError("Only a coordinator can write the handoff.");
      if (held.has(ctx)) return { content: `Ready: the shift handoff note. ${HELD_NOTE}`, summary: "Handoff note", pending: true };
      const note = await writeHandoff(u.deskId, ctx.now);
      if (!note.ok) return { content: note.text, isError: true, summary: "Handoff failed" };
      return { content: note.text, summary: "Handoff written" };
    }

    case "visualize_metric": {
      const id = s(i.metric) as MetricId;
      if (!METRIC_IDS.includes(id)) throw new ToolInputError(`Unknown metric. Allowed: ${METRIC_IDS.join(", ")}.`);
      // The same engine the chart runs in the browser, on the same board, so the reply matches the chart.
      const d = desk();
      const [board, events, users] = await Promise.all([d.getBoard(u.deskId), d.recentEvents(u.deskId, 60), d.listUsers()]);
      observeBoard(board);
      const r = metric(id, { board, events, users, now: deskNow() });
      const status = r.status === "unknown" ? `unknown: ${r.unknownReason ?? "no data"}` : r.status;
      return { content: `Chart shown: ${r.title}. ${r.headline} Status ${status}. Basis: ${r.basis}. Source: ${r.source.name}.`,
        summary: `Chart: ${r.title}`, showMetric: id };
    }

    case "show_on_map": {
      const kinds = (Array.isArray(i.kinds) ? i.kinds : []).filter((k): k is MapKind => MAP_KINDS.includes(k as MapKind));
      if (kinds.length === 0) throw new ToolInputError(`kinds needs at least one of: ${MAP_KINDS.join(", ")}.`);
      if ((mapCalls.get(ctx) ?? 0) >= 1) throw new ToolInputError("The map was already shown this turn. Answer from what it returned.");
      mapCalls.set(ctx, 1);
      const text = s(i.text)?.toLowerCase(), status = s(i.status)?.toLowerCase();
      const from = i.near_me === true ? u.coords : undefined;
      const d = desk();
      const b = await d.getBoard(u.deskId);
      const siteAt = new Map(b.sites.map((x) => [x._id, x]));
      // first: when the desk first heard of it (the first history entry), for "where did it start".
      type Hit = { id: string; line: string; state: string; coords?: LatLng; first?: Date | string };
      const hits: Hit[] = [];
      for (const x of kinds.includes("site") ? b.sites : []) hits.push({ id: x._id, state: x.status, coords: x.coords,
        first: x.history[0]?.ts ?? x.observedAt,
        line: `${x.address}, ${x.region}: ${x.peopleTrapped} trapped, ${x.status}${x.hazard ? `, hazard: ${x.hazard}` : ""}` });
      for (const x of kinds.includes("shelter") ? b.shelters : []) hits.push({ id: x._id,
        state: x.bedsTotal - x.bedsUsed > 0 ? "free" : "full", coords: x.coords, first: x.history[0]?.ts ?? x.observedAt,
        line: `${x.name}, ${x.region}: ${x.bedsTotal - x.bedsUsed} of ${x.bedsTotal} beds free` });
      for (const x of kinds.includes("road") ? b.roads : []) hits.push({ id: x._id, state: x.state, coords: x.coords,
        first: x.history[0]?.ts ?? x.observedAt,
        line: `${x.road}: ${x.state}${x.reason ? `, ${x.reason}` : ""}` });
      for (const x of kinds.includes("need") ? b.needs : []) hits.push({ id: x._id, state: x.status,
        coords: x.siteIds.map((sid) => siteAt.get(sid)?.coords).find(Boolean), first: x.history[0]?.ts,
        line: `${x.quantity} ${x.item}, ${x.region}: ${x.status}` });
      for (const x of kinds.includes("incident") ? b.incidents : []) hits.push({ id: x._id, state: x.status, coords: x.coords,
        first: x.history[0]?.ts,
        line: `${x.kind}: ${x.severity}, ${x.status}` });
      for (const x of kinds.includes("task") ? b.tasks.filter((t) => t.open) : []) hits.push({ id: x._id, state: x.status,
        coords: (x.recordType === "site" ? siteAt.get(x.recordId)?.coords : undefined),
        line: `${maskPhones(x.title).text}: ${x.status}` });
      // Only things that can be drawn: a record without coords has no pin.
      let found = hits.filter((h) => h.coords && (!text || h.line.toLowerCase().includes(text)) &&
        (!status || h.state.toLowerCase() === status));
      const dist = (h: Hit) => (from && h.coords ? haversineMeters(from, h.coords) : Infinity);
      // Time order wins over distance. A place with no first-report time goes last either way.
      const order = i.order === "earliest" || i.order === "latest" ? i.order : undefined;
      const at = (h: Hit) => (h.first ? +new Date(h.first) : undefined);
      if (order) {
        found = found.sort((a, c) => {
          const x = at(a), y = at(c);
          if (x === undefined || y === undefined) return (x === undefined ? 1 : 0) - (y === undefined ? 1 : 0);
          return order === "earliest" ? x - y : y - x;
        });
      } else if (from) found = found.sort((a, c) => dist(a) - dist(c));
      const now = new Date(deskNow());
      const ids = found.slice(0, 50).map((h) => h.id);
      const shown = found.slice(0, 8).map((h) =>
        `${h.line}${order ? `, first reported ${age(h.first, now)}` : ""}${from ? `, ${distance(dist(h))} away` : ""}`);
      const asked = kinds.join(", ");
      const listed = order === "earliest" ? "earliest 8 listed" : order === "latest" ? "newest 8 listed" : "first 8 listed";
      const content = found.length === 0
        ? `Nothing on the map for ${asked}${text ? ` matching "${text}"` : ""}. Say so, and say the map shows nothing for it.`
        : `Map shown: ${found.length} ${asked} on the map${found.length > 8 ? `, ${listed}` : ""}. The map flies to and selects the first: ${shown[0]}. ${shown.slice(1).join(". ")}${shown.length > 1 ? "." : ""}${i.near_me === true && !from ? " Distance unknown: the person's position is not shared." : ""}`;
      return { content, summary: `Map: ${found.length} ${asked}`, showMap: { kinds, ids, text } };
    }
  }
  return { content: `Unknown tool ${name}.`, isError: true };
}
