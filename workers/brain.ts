// Owned by Role 1. The Brain: ONE loop. Take the waiting events (people first, then the oldest),
// let the memory keeper apply each and the dispatcher react, then every few seconds run the
// sweeper (Final system design §10). Events are drained before each sweep, never one per sweep.
//
//   npm run brain
//
// Only one Brain may run against a database: it holds a lease in `meta`. Start a second one
// (a teammate on the same Atlas cluster) and it waits until the first stops or dies.
import { hostname } from "node:os";
import { setTimeout as sleep } from "node:timers/promises";
import { pathToFileURL } from "node:url";
import { bumpDeskVersion, closeDb, cols, markLoop } from "../lib/db";
import type { Event } from "../lib/types";
import * as dispatcher from "./dispatcher";
import { apply, HANDLED } from "./memory-keeper";
import * as sweeper from "./sweeper";

const ME = `${hostname()}:${process.pid}`;
const LEASE_MS = 15_000;
/** Keep handling events for up to this long before the sweeper gets its turn. */
export const DRAIN_MS = 4000;
/** The sweeper runs this long after its last run ENDED, so a slow sweep cannot run back to back. */
export const SWEEP_EVERY_MS = 5000;
/** The rules over every open record (reconcile, escalated tasks again) are the sweep's slow part. */
export const RECONCILE_EVERY_MS = 30_000;
const log = (...m: unknown[]) => console.log(`[brain ${new Date().toISOString().slice(11, 19)}]`, ...m);

/** Hold or renew the lease. False when another live Brain holds it. */
export async function acquireLease(now = new Date()): Promise<boolean> {
  const { meta } = await cols();
  try {
    await meta.updateOne(
      { _id: "brain-lease", $or: [{ holder: ME }, { until: { $lt: now } }] },
      { $set: { holder: ME, until: new Date(now.getTime() + LEASE_MS) } },
      { upsert: true },
    );
    return true;
  } catch (err) {
    // The lease exists, is someone else's and has not run out: the upsert hit the _id.
    if ((err as { code?: number }).code === 11000) return false;
    throw err;
  }
}

export async function releaseLease(): Promise<void> {
  const { meta } = await cols();
  await meta.updateOne({ _id: "brain-lease", holder: ME }, { $set: { until: new Date(0) } });
}

/** Take the next event: lane 0 (people, sweeper) first, then the oldest. Atomic. */
export async function takeNext(filter: { deskId?: string } = {}): Promise<Event | null> {
  const { events } = await cols();
  return events.findOneAndUpdate(
    { status: "pending", type: { $in: HANDLED }, ...filter },
    { $set: { status: "working", lockedAt: new Date() }, $inc: { attempts: 1 } },
    { sort: { lane: 1, ts: 1 }, returnDocument: "after" },
  );
}

/** Apply one event and record how it ended. Never throws: a failing event is retried, then failed. */
export async function processEvent(event: Event): Promise<void> {
  const { events } = await cols();
  try {
    const out = await apply(event);
    await events.updateOne({ _id: event._id }, { $set: { status: out.status, result: out.result }, $unset: { lockedAt: "" } });
    log(`${event.type} ${event._id} (${event.source}) → ${out.status} ${out.result.recordIds.join(", ")}: ${out.result.reason}`);
    await dispatcher.react(out.changes, out.actions);
  } catch (err) {
    const reason = err instanceof Error ? err.message : String(err);
    const giveUp = (event.attempts ?? 1) >= sweeper.MAX_ATTEMPTS;
    await events.updateOne(
      { _id: event._id },
      giveUp
        ? { $set: { status: "failed", result: { recordIds: [], reason } }, $unset: { lockedAt: "" } }
        : { $set: { status: "pending" }, $unset: { lockedAt: "" } },
    );
    log(`${event.type} ${event._id} ${giveUp ? "FAILED" : `error (attempt ${event.attempts}), will retry`}: ${reason}`);
  } finally {
    // Records may have changed (even a failed apply can write part of its work): tell the
    // readers' board copies to refresh. Best effort; their copies also expire after 20 s.
    await bumpDeskVersion(event.deskId).catch((e) => log(`version bump failed: ${e instanceof Error ? e.message : e}`));
  }
}

/**
 * Handle events until the queue is empty or `ms` has passed. One event per sweep made every
 * event wait for a whole sweep (about 9 s on the VM), so a person's tap took that long to show.
 * `beat` runs between events (lease, heartbeat). Returns how many events were handled.
 */
export async function drain(
  filter: { deskId?: string } = {}, ms = DRAIN_MS, beat: () => Promise<boolean> = async () => true,
): Promise<number> {
  const until = Date.now() + ms;
  let handled = 0;
  while (Date.now() < until && (await beat())) {
    const event = await takeNext(filter);
    if (!event) break;
    await processEvent(event);
    handled++;
  }
  return handled;
}

/** One turn without the lease or the timers. Used by npm run keeper:check. */
export async function runOnce(filter: { deskId?: string } = {}): Promise<boolean> {
  const event = await takeNext(filter);
  if (event) await processEvent(event);
  return Boolean(event);
}

async function main() {
  let stopping = false;
  const stop = () => { stopping = true; };
  process.on("SIGINT", stop);
  process.on("SIGTERM", stop);

  log(`starting as ${ME}, handling: ${HANDLED.join(", ")}`);
  let waiting = false;
  let lastSweepEnd = 0;
  let lastReconcile = 0;
  let lastBeat = 0;
  let lastLease = 0;

  // Between events: keep the lease (a long drain or sweep must not let it run out) and the heartbeat.
  const beat = async (): Promise<boolean> => {
    if (stopping) return false;
    const now = Date.now();
    if (now - lastLease >= LEASE_MS / 3) {
      if (!(await acquireLease())) return false;
      lastLease = now;
    }
    if (now - lastBeat >= 1000) { lastBeat = now; await markLoop(); }
    return true;
  };

  while (!stopping) {
    if (!(await acquireLease())) {
      if (!waiting) {
        const { meta } = await cols();
        const lease = await meta.findOne({ _id: "brain-lease" });
        log(`another Brain is running (${lease?.holder}). Waiting for it to stop...`);
        waiting = true;
      }
      await sleep(5000);
      continue;
    }
    lastLease = Date.now();
    if (waiting) { log("lease acquired, running"); waiting = false; }

    // Events first: everything that is waiting, for up to DRAIN_MS.
    const handled = await drain({}, DRAIN_MS, beat);

    if (!stopping && Date.now() - lastSweepEnd >= SWEEP_EVERY_MS && (await acquireLease())) {
      lastLease = Date.now();
      const reconcile = Date.now() - lastReconcile >= RECONCILE_EVERY_MS;
      const s = await sweeper.tick(new Date(), { reconcile });
      lastSweepEnd = Date.now();
      if (reconcile) lastReconcile = lastSweepEnd;
      if (s.requeued || s.failed) log(`sweeper: ${s.requeued} stuck event(s) back to pending, ${s.failed} failed`);
      if (s.timedOut || s.checkins || s.unreachable || s.reprioritized || s.created || s.reassigned) {
        log(`sweeper: ${s.timedOut} no answer, ${s.checkins} check-in(s), ${s.unreachable} unreachable, ${s.reprioritized} priority update(s), ${s.created} task(s) made, ${s.reassigned} task(s) reassigned`);
      }
      if (s.approvalsLate || s.offline || s.sheltersAsked || s.roadsAsked || s.needsExpired) {
        log(`sweeper: ${s.approvalsLate} approval(s) sent again, ${s.offline} task(s) off an offline person, ${s.sheltersAsked} shelter question(s), ${s.roadsAsked} road question(s), ${s.needsExpired} proposed need(s) cancelled`);
      }
    }
    if (Date.now() - lastBeat >= 1000) { lastBeat = Date.now(); await markLoop(); }
    if (!handled) await sleep(300);
  }

  log("stopping");
  await releaseLease();
  await closeDb();
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch(async (err) => {
    console.error("[brain] crashed:", err);
    await closeDb();
    process.exit(1);
  });
}
