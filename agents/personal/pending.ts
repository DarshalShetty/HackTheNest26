// Tool calls that wait for the person's OK ("I'll send this to the desk, right?").
// A turn with confirm=true runs its writes as held (see holdWrites in tools.ts) and keeps them here.
// POST /api/chat { approve: id, send: true|false } then sends them, or drops them.
// In memory, one use, 10 minutes: a restart or a late tap just asks the person to say it again.

export interface PendingCall { name: string; input: Record<string, unknown> }
interface Batch { userId: string; calls: PendingCall[]; expires: number }

const TTL_MS = 10 * 60_000;
const g = globalThis as unknown as { __pendingBatches?: Map<string, Batch> };
const batches: Map<string, Batch> = (g.__pendingBatches ??= new Map<string, Batch>());

export function stashPending(userId: string, calls: PendingCall[]): string {
  const now = Date.now();
  for (const [k, b] of batches) if (b.expires < now) batches.delete(k);
  const id = crypto.randomUUID();
  batches.set(id, { userId, calls, expires: now + TTL_MS });
  return id;
}

/** The held calls of this person, once. Another person's id, an old id or a used id gives null. */
export function takePending(id: string, userId: string): PendingCall[] | null {
  const b = batches.get(id);
  if (!b || b.userId !== userId || b.expires < Date.now()) return null;
  batches.delete(id);
  return b.calls;
}
