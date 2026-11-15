// Owned by Role 1. The sweeper: timers, run from the Brain loop every few seconds.
//   1. Events stuck in "working" (a Brain that crashed mid-event) go back to the queue.
//   2. An assigned task past its deadline: no answer, next person (the dispatcher does it).
//   3. An accepted task past its ETA: one check-in. Still silent after the grace period: an
//      "unreachable" incident event, so the usual incident path protects the mission.
//   4. Every PRIORITY_EVERY_MS: the waiting part of the score grows, so recompute it.
//   5. Every tick: the dispatcher's rules over the open records (tasks for records that were already
//      there), and escalated tasks offered to whoever is free now.
//   6. The rest of the §5 sweeper table (v2.6): an approval nobody answered is sent again and marked
//      late. A person not seen for 3 min loses their unanswered tasks now. An accepted task with no
//      promised time and no word gets a check-in. A shelter number older than 30 min and a road
//      unconfirmed for 20 min get a "please confirm". A proposed need quiet for 2 h is cancelled.
// The sweeper writes no tasks, inbox items or records. It asks the dispatcher and the memory
// keeper (the only writers) or posts an event. Every step is a conditional update or a deduped
// event, so two overlapping ticks, or a restart, handle each task once.
import { bumpDeskVersion, cols, postEvent } from "../lib/db";
import { SEEN_WITHIN_MS } from "../lib/rules/assign";
import type { IncidentPayload } from "../lib/types";
import { approvalLate, askRoadConfirm, askShelterConfirm, checkIn, offlineTimedOut, proposedExpired, reconcile, redelegate, syncPriority, timedOut } from "./dispatcher";
import { expireProposed, refreshPriorities } from "./memory-keeper";

export const STUCK_AFTER_MS = 30_000;
export const MAX_ATTEMPTS = 3;
/** After the check-in, this long without a new answer and the person counts as unreachable. */
export const CHECKIN_GRACE_MINUTES = 2;
/** Priority is recomputed at most this often (the tick itself runs every 5 s). */
export const PRIORITY_EVERY_MS = 30_000;
/** An accepted task with no promised time and no word from its person this long gets a check-in (M in §5). */
export const SILENT_MINUTES = Number(process.env.SILENT_MINUTES) > 0 ? Number(process.env.SILENT_MINUTES) : 10;
/** A shelter number older than this is asked again. */
export const SHELTER_STALE_MINUTES = 30;
/** A road nobody trusted confirmed for this long: the nearest field team is asked. */
export const ROAD_UNCONFIRMED_MINUTES = 20;

export interface TickResult {
  requeued: number; failed: number; timedOut: number; checkins: number; unreachable: number; reprioritized: number;
  created: number; reassigned: number;
  approvalsLate: number; offline: number; sheltersAsked: number; roadsAsked: number; needsExpired: number;
}

let lastPriorityAt: number | null = null;

/**
 * `now` lets a check move time forward. `deskId` limits the tick to one desk (checks use it).
 * `reconcile: false` skips the rules over every open record and the retry of escalated tasks:
 * they are the slow part, and the Brain runs them every RECONCILE_EVERY_MS, not every tick.
 */
export async function tick(
  now: Date = new Date(), opts: { deskId?: string; reconcile?: boolean } = {},
): Promise<TickResult> {
  const { events, tasks, users, shelters, roads } = await cols();
  const desk = opts.deskId ? { deskId: opts.deskId } : {};

  const stuck = { ...desk, status: "working" as const, lockedAt: { $lt: new Date(now.getTime() - STUCK_AFTER_MS) } };
  const failed = await events.updateMany(
    { ...stuck, attempts: { $gte: MAX_ATTEMPTS } },
    { $set: { status: "failed", result: { recordIds: [], reason: `stuck in working ${MAX_ATTEMPTS} times` } }, $unset: { lockedAt: "" } },
  );
  const requeued = await events.updateMany(
    { ...stuck, attempts: { $lt: MAX_ATTEMPTS } },
    { $set: { status: "pending" }, $unset: { lockedAt: "" } },
  );

  // No answer → next person.
  let timed = 0;
  for (const t of await tasks.find({ ...desk, status: "assigned", open: true, deadline: { $lt: now } }).toArray()) {
    await timedOut(t, now);
    timed++;
  }

  const touched = new Set<string>();
  // Not seen for 3 min → offline: their unanswered tasks move on now, not at the deadline.
  let offline = 0;
  const assigned = await tasks.find({ ...desk, status: "assigned", open: true, assignee: { $exists: true } }).toArray();
  if (assigned.length) {
    const seen = new Map((await users.find({ _id: { $in: [...new Set(assigned.map((t) => t.assignee!))] } }, { projection: { lastSeen: 1 } }).toArray())
      .map((u) => [u._id, u.lastSeen]));
    for (const t of assigned) {
      const at = seen.get(t.assignee!);
      if (at && now.getTime() - new Date(at).getTime() <= SEEN_WITHIN_MS) continue;
      if (await offlineTimedOut(t, now)) { offline++; touched.add(t.deskId); }
    }
  }

  // An approval nobody answered → sent again, marked late.
  let approvalsLate = 0;
  for (const t of await tasks.find({ ...desk, status: "waiting_approval", open: true, deadline: { $lt: now } }).toArray()) {
    if (await approvalLate(t, now)) { approvalsLate++; touched.add(t.deskId); }
  }

  // Past the promised time, or no promised time and no word → check in; still silent after the grace → unreachable.
  let checkins = 0;
  for (const t of await tasks.find({ ...desk, status: "accepted", open: true, eta: { $lt: now }, checkinAt: { $exists: false } }).toArray()) {
    if (await checkIn(t, now)) checkins++;
  }
  const quietSince = now.getTime() - SILENT_MINUTES * 60_000;
  for (const t of await tasks.find({ ...desk, status: "accepted", open: true, eta: { $exists: false }, checkinAt: { $exists: false } }).toArray()) {
    // The last word is the newest history line by the person. None at all: the newest line.
    const theirs = t.history.filter((h) => h.by === t.assignee);
    const last = Math.max(...(theirs.length ? theirs : t.history).map((h) => new Date(h.ts).getTime()), 0);
    if (last && last < quietSince && await checkIn(t, now, "silent")) checkins++;
  }
  let unreachable = 0;
  const silentSince = new Date(now.getTime() - CHECKIN_GRACE_MINUTES * 60_000);
  for (const t of await tasks.find({ ...desk, status: "accepted", open: true, checkinAt: { $lte: silentSince } }).toArray()) {
    // A newer ETA than the check-in is a new answer.
    if (!t.assignee || (t.eta && t.checkinAt && t.eta > t.checkinAt)) continue;
    const payload: IncidentPayload = {
      kind: "unreachable", taskId: t._id, canContinue: false,
      text: `${t.assignee} did not answer the check-in for ${CHECKIN_GRACE_MINUTES} min`,
      textEn: `${t.assignee} passed the promised time, was asked "still on your way?" and did not answer for ${CHECKIN_GRACE_MINUTES} min`,
    };
    await postEvent<IncidentPayload>({
      deskId: t.deskId, type: "incident", source: "system", actor: "sweeper",
      dedupeKey: `sweeper:unreachable:${t._id}`, payload,
    });
    unreachable++;
  }

  // Old facts: ask the people who can see them. Each question goes once per reported value.
  let sheltersAsked = 0;
  for (const sh of await shelters.find({ ...desk, observedAt: { $lt: new Date(now.getTime() - SHELTER_STALE_MINUTES * 60_000) } }).toArray()) {
    sheltersAsked += await askShelterConfirm(sh, now);
  }
  let roadsAsked = 0;
  for (const r of await roads.find({
    ...desk, state: { $ne: "open" }, confidence: "unconfirmed", observedAt: { $lt: new Date(now.getTime() - ROAD_UNCONFIRMED_MINUTES * 60_000) },
  }).toArray()) {
    if (await askRoadConfirm(r, now)) roadsAsked++;
  }
  const expired = await expireProposed(now, opts.deskId);
  for (const n of expired) { await proposedExpired(n); touched.add(n.deskId); }

  // Time passes: priorities grow.
  let reprioritized = 0;
  if (lastPriorityAt === null || Math.abs(now.getTime() - lastPriorityAt) >= PRIORITY_EVERY_MS) {
    lastPriorityAt = now.getTime();
    const changed = await refreshPriorities(now, opts.deskId);
    reprioritized = changed.length;
    if (changed.length) await syncPriority(changed);
  }

  // The rules over what is on the desk now, and escalated tasks tried again. Both write through the dispatcher.
  const full = opts.reconcile !== false;
  const rec = full ? await reconcile(opts.deskId) : { created: 0, desks: [] as string[] };
  const re = full ? await redelegate(opts.deskId, now) : { moved: 0, desks: [] as string[] };
  for (const d of new Set([...touched, ...rec.desks, ...re.desks])) await bumpDeskVersion(d);

  return {
    requeued: requeued.modifiedCount, failed: failed.modifiedCount, timedOut: timed, checkins, unreachable, reprioritized,
    created: rec.created, reassigned: re.moved,
    approvalsLate, offline, sheltersAsked, roadsAsked, needsExpired: expired.length,
  };
}
