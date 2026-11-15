// Check for the Brain's loop order: npm run brain:check
// The old loop handled ONE event, then ran the whole sweeper, so every event waited a sweep.
// The new loop drains the queue first (drain()) and runs the slow rules (reconcile) on a timer.
// This replays both orders on a throwaway desk, "check-brain", with a desk the size of the real
// one (13 open sites with tasks), and compares them. Nothing touches the real desk, and the
// desk is deleted at the end. A live Brain on the same database may take a few of these events
// too: the check waits for the queue to be empty, so that only changes the timings a little.
import assert from "node:assert/strict";
import { drain, DRAIN_MS, runOnce } from "../workers/brain";
import { tick } from "../workers/sweeper";
import { closeDb, cols, postEvent } from "./db";
import type { LatLng, ReportPayload, User } from "./types";

const DESK = "check-brain";
const SITES = 13;
const BATCH = 12;
let passed = 0;
const ok = (msg: string) => { passed++; console.log(`ok  ${msg}`); };
const secs = (ms: number) => `${(ms / 1000).toFixed(1)} s`;
const at = (i: number): LatLng => [38.3 + i * 0.003, 38.3]; // about 330 m apart: separate sites

async function cleanup() {
  const c = await cols();
  const ids = (await c.users.find({ deskId: DESK }, { projection: { _id: 1 } }).toArray()).map((u) => u._id);
  await Promise.all([
    c.inbox.deleteMany({ userId: { $in: ids } }), c.messages.deleteMany({ userId: { $in: ids } }),
    c.events.deleteMany({ deskId: DESK }), c.sites.deleteMany({ deskId: DESK }), c.needs.deleteMany({ deskId: DESK }),
    c.tasks.deleteMany({ deskId: DESK }), c.incidents.deleteMany({ deskId: DESK }), c.users.deleteMany({ deskId: DESK }),
    c.meta.deleteMany({ _id: { $in: [`cases:${DESK}`, `version:${DESK}`] } }),
  ]);
}

const pendingLeft = async () =>
  (await cols()).events.countDocuments({ deskId: DESK, status: { $in: ["pending", "working"] } });

async function waitEmpty() {
  for (let i = 0; i < 300 && (await pendingLeft()); i++) {
    if (!(await runOnce({ deskId: DESK }))) await new Promise((r) => setTimeout(r, 200));
  }
  assert.equal(await pendingLeft(), 0, "queue is empty");
}

/** `n` reports that merge into the existing sites (different words, same places). */
async function postBatch(tag: string, n: number) {
  for (let i = 0; i < n; i++) {
    await postEvent<ReportPayload>({
      deskId: DESK, type: "report", source: "x", actor: "check-brain", dedupeKey: `check-brain:${tag}:${i}`,
      payload: { kind: "trapped", text: `${tag} report ${i}`, textEn: `${tag} report ${i}`, coords: at(i % SITES), peopleCount: 1 },
    });
  }
}

async function main() {
  await cleanup();
  const c = await cols();
  const now = new Date();
  const person = (id: string, role: User["role"], i: number): User => ({
    _id: `${DESK}:${id}`, name: id, role, region: "check", status: "available", deskId: DESK,
    trust: role === "coordinator" ? "official" : "vetted", coords: at(i), lastSeen: now,
  });
  await c.users.insertMany([
    ...[0, 1, 2, 3, 4, 5].map((i) => person(`volunteer-${i}`, "volunteer", i * 2)),
    person("field-0", "field", 3), person("coordinator-0", "coordinator", 0),
  ]);

  // A desk like the real one: 13 open sites, each with its call task.
  for (let i = 0; i < SITES; i++) {
    await postEvent<ReportPayload>({
      deskId: DESK, type: "report", source: "x", actor: "check-brain", dedupeKey: `check-brain:seed:${i}`,
      payload: { kind: "trapped", text: `seed site ${i}`, textEn: `seed site ${i}`, coords: at(i), peopleCount: 1 },
    });
  }
  await waitEmpty();
  await tick(new Date(), { deskId: DESK });
  assert.equal(await c.sites.countDocuments({ deskId: DESK }), SITES);
  ok(`desk ready: ${SITES} open sites, ${await c.tasks.countDocuments({ deskId: DESK, open: true })} open tasks`);

  // How long is a sweep, with and without the slow rules?
  let t = Date.now(); await tick(new Date(), { deskId: DESK }); const fullMs = Date.now() - t;
  t = Date.now(); await tick(new Date(), { deskId: DESK, reconcile: false }); const lightMs = Date.now() - t;
  assert(lightMs < fullMs, `a sweep without reconcile is faster (${lightMs} ms vs ${fullMs} ms)`);
  ok(`one sweep: ${secs(fullMs)} with reconcile, ${secs(lightMs)} without`);

  // OLD order: one event, then a full sweep whenever 5 s passed since the last sweep started.
  await postBatch("old", BATCH);
  let lastSweep = 0; let sweepsOld = 0; const t0 = Date.now();
  while (await runOnce({ deskId: DESK })) {
    if (Date.now() - lastSweep >= 5000) { lastSweep = Date.now(); await tick(new Date(), { deskId: DESK }); sweepsOld++; }
  }
  const oldMs = Date.now() - t0;
  await waitEmpty();

  // NEW order: drain the queue, then the sweep (reconcile only when it is due).
  await postBatch("new", BATCH);
  const t1 = Date.now();
  let handled = 0; let sweepsNew = 0; let drains = 0;
  while (await pendingLeft()) {
    handled += await drain({ deskId: DESK }, DRAIN_MS); drains++;
    await tick(new Date(), { deskId: DESK, reconcile: sweepsNew === 0 }); sweepsNew++;
    if (drains > 20) break;
  }
  const newMs = Date.now() - t1;
  await waitEmpty();
  ok(`${BATCH} waiting events: old order ${secs(oldMs)} (${sweepsOld} sweeps), new order ${secs(newMs)} (${sweepsNew} sweep(s)), ${(oldMs / newMs).toFixed(1)}× faster`);
  assert(handled >= BATCH - 3, `drain handled the batch before sweeping (${handled} of ${BATCH})`);
  assert(newMs < oldMs, "the new order empties the queue sooner");

  // A person's tap while feed events wait: lane 0 goes first, and no sweep stands in its way.
  await postBatch("feed", 6);
  const t2 = Date.now();
  const tap = await postEvent<ReportPayload>({
    deskId: DESK, type: "report", source: "person", actor: `${DESK}:field-0`, dedupeKey: "check-brain:tap",
    payload: { kind: "site_update", text: "team on site", textEn: "team on site", coords: at(3) },
  });
  // One short drain: long enough for the first event or two, far too short for all seven.
  await drain({ deskId: DESK }, 1);
  const tapStatus = (await c.events.findOne({ _id: tap }))?.status;
  const feedWaiting = await c.events.countDocuments({ deskId: DESK, dedupeKey: /^check-brain:feed:/, status: "pending" });
  assert(tapStatus === "done" || tapStatus === "merged", `the person's event was taken first (${tapStatus})`);
  ok(`a person's event is handled first, ${secs(Date.now() - t2)} after the tap, while ${feedWaiting} feed events still wait`);
  await waitEmpty();

  // Same end state as before: nothing lost, nothing doubled.
  const st = await c.events.aggregate<{ _id: string; n: number }>([{ $match: { deskId: DESK } }, { $group: { _id: "$status", n: { $sum: 1 } } }]).toArray();
  assert(st.every((s) => ["done", "merged", "rejected"].includes(s._id)), `every event finished: ${JSON.stringify(st)}`);
  assert.equal(await c.sites.countDocuments({ deskId: DESK }), SITES, "no extra sites");
  const dup = await c.tasks.aggregate([{ $match: { deskId: DESK, open: true } }, { $group: { _id: { r: "$recordId", t: "$type" }, n: { $sum: 1 } } }, { $match: { n: { $gt: 1 } } }]).toArray();
  assert.equal(dup.length, 0, "no duplicate open tasks");
  ok(`end state: ${st.map((s) => `${s.n} ${s._id}`).join(", ")}; ${SITES} sites; no duplicate open tasks`);

  console.log(`\nAll ${passed} checks passed.`);
}

main()
  .catch((err) => { console.error("\nFAILED:", err instanceof Error ? err.message : err); process.exitCode = 1; })
  .finally(async () => { await cleanup(); await closeDb(); });
