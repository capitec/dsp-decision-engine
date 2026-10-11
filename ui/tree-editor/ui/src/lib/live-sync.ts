/**
 * Live sync of one project document over the project's websocket (`/api/v1/projects/{id}/live`).
 *
 * The client keeps `confirmed` (the document at the last server version it has seen) and `pending` (its messages
 * the server has not acknowledged yet); the editor shows `apply(confirmed, pending)`. Local edits are diffed into a
 * message about 150 ms after the last one; every server version arrives in order (my own as the ack), and before one
 * is applied the editor's un-diffed edits are pushed first, so a remote change never overwrites a local one unseen.
 * After a disconnect (a closed socket, or no answer to a ping for ~10 s) the edits made meanwhile, and the messages
 * the catch-up did not show applied, go back as one offline batch the server merges whole or rejects whole. A
 * close for policy (signed out, or access removed) ends it until `resume()`: unsent edits stay in the editor.
 *
 * @example
 * const sync = new LiveSync(liveUrl(projectId), project.project_data, project.version, {
 *   read: () => buildDocument(store.getState()),
 *   show: (doc) => store.getState().applyDocument(doc),
 * })
 * store.subscribe(() => sync.edit())   // after every local edit
 * await sync.settle()                  // Save: push now and wait for the server
 */
import { apply, diff, type Json, type Op } from './live-ops'

/** `ended`: the server closed live editing for good (signed out, or access removed). */
export type LiveStatus = 'connecting' | 'live' | 'reconnecting' | 'ended'
/** `parts`: the earlier messages an offline batch contains, `[id, op count]` in order. */
type Message = { id: string; base: number; ops: Op[]; offline?: boolean; parts?: Array<[string, number]> }
type Socket = Pick<WebSocket, 'send' | 'close' | 'onopen' | 'onmessage' | 'onclose'>

export interface LiveHooks {
  /** The document the editor holds now. */
  read: () => Json
  /** Show `doc` (the server's latest version plus my unacknowledged edits) in the editor. */
  show: (doc: Json) => void
  status?: (status: LiveStatus) => void
  /** The server applied my message without these ops (someone else changed or deleted what they edit). */
  dropped?: (ops: Op[]) => void
  /** The server refused one of my messages whole; its edits are gone from the editor. */
  rejected?: (reason: string) => void
  /** My offline edits were merged. */
  merged?: () => void
  /** My offline edits were refused whole and are gone from the editor; `doc` is my version with them. */
  offlineRejected?: (rejection: { reason: string; conflicts: string[]; doc: Json }) => void
  /** The server's validation errors of the version the editor shows (`version`, my pending edits aside). */
  validation?: (errors: unknown[], version: number) => void
  /** Every edit is on the server (nothing pending). */
  settled?: (version: number) => void
  /** The server ended live editing (`reason`: signed out, or access removed); edits it does not have wait for
   * `resume()`. */
  ended?: (reason: string) => void
}

const POLICY_VIOLATION = 1008
// Not JSON, too big, server error: the server failed on what I sent. Any other drop (a lost connection, a missed
// ping, a restart, a database outage) says nothing about my edits, so it is never counted against them.
const FAILED = [1007, 1009, 1011]
const MAX_FAILURES = 3

export const liveUrl = (projectId: string) =>
  `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/v1/projects/${projectId}/live`

export class LiveSync {
  confirmed: Json
  version: number
  pending: Message[] = []
  view: Json
  readonly clientId = crypto.randomUUID()
  private edited = false
  private ready = false
  private stopped = false
  private retries = 0
  private failures = 0
  private timer?: ReturnType<typeof setTimeout>
  private socket?: Socket
  private waiters: Array<() => void> = []
  private heard = true
  private heartbeat: ReturnType<typeof setInterval>
  /** The version (and document) my edits since the disconnect were made on. */
  private offlineBase?: { version: number; doc: Json }
  private offlineDoc: Json = null
  // Validation runs apart from the fan-out, so it can arrive before the version it is for.
  private validation?: { errors: unknown[]; version: number }

  constructor(
    private url: string,
    doc: Json,
    version: number,
    private hooks: LiveHooks,
    private debounceMs = 150,
    private open: (url: string) => Socket = (u) => new WebSocket(u),
    heartbeatMs = 10000,
  ) {
    this.confirmed = this.view = doc
    this.version = version
    this.connect()
    // A dead TCP connection otherwise goes unnoticed for minutes.
    this.heartbeat = setInterval(() => {
      if (!this.socket) return
      if (!this.heard) return this.lost(this.socket)
      this.heard = false
      if (this.ready) this.send({ type: 'ping' })
    }, heartbeatMs)
  }

  /** The editor changed: push the change once the edits pause. */
  edit() {
    this.edited = true
    clearTimeout(this.timer)
    this.timer = setTimeout(() => this.flush(), this.debounceMs)
  }

  /** Diff the editor against what it showed and queue (and send) the change as one message. */
  flush() {
    clearTimeout(this.timer)
    if (!this.edited) return
    this.edited = false
    // Through JSON, as the server will see it: an undefined field is no field.
    const now = JSON.parse(JSON.stringify(this.hooks.read()))
    const ops = diff(this.view, now)
    if (!ops.length) return
    const message = { id: crypto.randomUUID(), base: this.version, ops }
    this.pending.push(message)
    this.view = now
    if (this.ready) this.send({ type: 'ops', ...message })
  }

  /** True when every edit is on the server. */
  idle() {
    return !this.edited && this.pending.length === 0
  }

  /** Push the edits now and wait until the server has them all; false after `timeoutMs`. */
  settle(timeoutMs = 15000): Promise<boolean> {
    this.flush()
    if (this.idle()) return Promise.resolve(true)
    return new Promise((resolve) => {
      const timeout = setTimeout(() => resolve(false), timeoutMs)
      this.waiters.push(() => {
        clearTimeout(timeout)
        resolve(true)
      })
    })
  }

  /** Connect again after the server ended live editing (the user signed in again); unsent edits go then. */
  resume() {
    if (this.socket || this.stopped) return
    this.hooks.status?.('reconnecting')
    this.connect()
  }

  stop() {
    this.stopped = true
    clearTimeout(this.timer)
    clearInterval(this.heartbeat)
    this.socket?.close()
  }

  private connect() {
    this.ready = false
    this.heard = true
    const socket = (this.socket = this.open(this.url))
    socket.onopen = () => this.hello()
    socket.onmessage = (event) => {
      this.heard = true
      this.receive(JSON.parse(event.data))
    }
    socket.onclose = (event) => this.lost(socket, event)
  }

  private lost(socket: Socket, closed?: { code?: number; reason?: string }) {
    if (this.stopped || socket !== this.socket) return
    this.socket = undefined
    socket.onopen = socket.onmessage = socket.onclose = null
    socket.close()
    const sent = this.ready && this.pending.length > 0
    this.ready = false
    this.offlineBase ??= { version: this.version, doc: this.confirmed }
    if (closed?.code === POLICY_VIOLATION) {
      // No reconnecting until `resume()`: the server would refuse every try.
      this.hooks.status?.('ended')
      return this.hooks.ended?.(closed.reason || 'live editing ended')
    }
    // The server failed right after my edits went out, with none of them answered: one it fails on every time
    // would otherwise be resent on every reconnect for ever.
    if (sent && FAILED.includes(closed?.code ?? 0) && ++this.failures >= MAX_FAILURES) this.handBack()
    this.hooks.status?.('reconnecting')
    setTimeout(() => this.connect(), Math.min(250 * 2 ** this.retries++, 4000))
  }

  /** Give up on the pending edits: the editor shows the server's version and the hook gets mine. */
  private handBack() {
    this.failures = 0
    this.flush()
    const doc = this.view
    this.pending = []
    this.waiters = []  // a Save waiting on them times out unsaved
    this.refresh()
    this.hooks.offlineRejected?.({ reason: `the server failed on them ${MAX_FAILURES} times in a row.`, conflicts: [], doc })
  }

  /** After a reconnect: everything pending goes as one offline batch (resent as is when it already is one). */
  private sendOffline(base: { version: number; doc: Json }) {
    const [first] = this.pending
    // A message still in flight when the socket dropped may be applied after the catch-up; the parts let the
    // server skip it instead of applying it twice.
    const batch = this.pending.length === 1 && first.offline
      ? first
      : {
          id: crypto.randomUUID(), base: base.version, ops: this.pending.flatMap((p) => p.ops), offline: true,
          parts: this.pending.flatMap((p) => p.parts ?? [[p.id, p.ops.length] as [string, number]]),
        }
    this.pending = [batch]
    this.offlineDoc = apply(base.doc, batch.ops)[0]
    this.send({ type: 'ops', ...batch })
  }

  private offlineDone() {
    // A batch settled during a catch-up leaves edits pending that were made on the versions they name, before the
    // ones this catch-up brings: a newer base would hide those versions from the server's conflict check.
    this.offlineBase = this.ready
      ? undefined
      : { version: Math.min(this.version, ...this.pending.map((p) => p.base)), doc: this.confirmed }
  }

  private hello() {
    this.ready = false
    this.send({ type: 'hello', since: this.version, client_id: this.clientId })
  }

  private send(message: object) {
    this.socket?.send(JSON.stringify(message))
  }

  private receive(m: any) {
    if (m.type === 'ready') {
      this.ready = true
      this.retries = 0
      this.hooks.status?.('live')
      // Messages applied before a reconnect came back in the catch-up and are no longer pending.
      if (this.offlineBase && this.pending.length) return this.sendOffline(this.offlineBase)
      this.offlineBase = undefined
      for (const message of this.pending) this.send({ type: 'ops', ...message })
    } else if (m.type === 'snapshot') {
      this.flush()
      this.confirmed = m.doc
      this.version = m.version
      this.refresh()
    } else if (m.type === 'applied') {
      if (m.version <= this.version) return
      if (m.version !== this.version + 1) return this.hello()  // a gap: catch up again
      this.flush()
      this.confirmed = apply(this.confirmed, m.ops)[0]
      this.version = m.version
      const mine = m.client_id === this.clientId ? this.pending.find((p) => p.id === m.msg_id) : undefined
      if (mine) this.failures = 0
      this.pending = this.pending.filter((p) => p !== mine)
      if (mine?.offline) this.offlineDone()
      this.refresh()
      if (mine?.offline) this.hooks.merged?.()
    } else if (m.type === 'rejected') {
      this.flush()
      const mine = this.pending.find((p) => p.id === m.msg_id)
      if (mine) this.failures = 0
      this.pending = this.pending.filter((p) => p !== mine)
      this.refresh()
      if (!mine?.offline) return this.hooks.rejected?.(m.reason)
      this.offlineDone()
      this.hooks.offlineRejected?.({ reason: m.reason, conflicts: m.conflicts ?? [], doc: this.offlineDoc })
    } else if (m.type === 'dropped') {
      this.hooks.dropped?.(m.ops)
    } else if (m.type === 'validation') {
      this.validation = m
      this.report()
    } else if (m.type === 'session') {
      // My edit renewed the session; only the page can update the expiry cookie the session check reads.
      document.cookie = m.cookie
    } else if (m.type === 'error') {
      console.warn('live sync:', m.reason)
    }
  }

  private refresh() {
    this.view = apply(this.confirmed, this.pending.flatMap((p) => p.ops))[0]
    this.hooks.show(this.view)
    this.report()
    if (this.idle()) {
      this.hooks.settled?.(this.version)
      this.waiters.splice(0).forEach((resolve) => resolve())
    }
  }

  private report() {
    if (this.validation && this.validation.version <= this.version) {
      if (this.validation.version === this.version) this.hooks.validation?.(this.validation.errors, this.version)
      this.validation = undefined
    }
  }
}
