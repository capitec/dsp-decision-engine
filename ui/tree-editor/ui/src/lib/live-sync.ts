/**
 * Live sync of one project document over the project's websocket (`/api/v1/projects/{id}/live`).
 *
 * The client keeps `confirmed` (the document at the last server version it has seen) and `pending` (its messages
 * the server has not acknowledged yet); the editor shows `apply(confirmed, pending)`. Local edits are diffed into a
 * message about 150 ms after the last one; every server version arrives in order (my own as the ack), and before one
 * is applied the editor's un-diffed edits are pushed first, so a remote change never overwrites a local one unseen.
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

export type LiveStatus = 'connecting' | 'live' | 'reconnecting'
type Message = { id: string; base: number; ops: Op[] }
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
  /** The server's validation errors of the version the editor shows (`version`, my pending edits aside). */
  validation?: (errors: unknown[], version: number) => void
  /** Every edit is on the server (nothing pending). */
  settled?: (version: number) => void
}

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
  private timer?: ReturnType<typeof setTimeout>
  private socket?: Socket
  private waiters: Array<() => void> = []
  // Validation runs apart from the fan-out, so it can arrive before the version it is for.
  private validation?: { errors: unknown[]; version: number }

  constructor(
    private url: string,
    doc: Json,
    version: number,
    private hooks: LiveHooks,
    private debounceMs = 150,
    private open: (url: string) => Socket = (u) => new WebSocket(u),
  ) {
    this.confirmed = this.view = doc
    this.version = version
    this.connect()
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

  stop() {
    this.stopped = true
    clearTimeout(this.timer)
    this.socket?.close()
  }

  private connect() {
    this.ready = false
    const socket = (this.socket = this.open(this.url))
    socket.onopen = () => this.hello()
    socket.onmessage = (event) => this.receive(JSON.parse(event.data))
    socket.onclose = () => {
      if (this.stopped || socket !== this.socket) return
      this.ready = false
      this.hooks.status?.('reconnecting')
      setTimeout(() => this.connect(), Math.min(250 * 2 ** this.retries++, 4000))
    }
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
      if (m.client_id === this.clientId) this.pending = this.pending.filter((p) => p.id !== m.msg_id)
      this.refresh()
    } else if (m.type === 'rejected') {
      this.flush()
      this.pending = this.pending.filter((p) => p.id !== m.msg_id)
      this.refresh()
      this.hooks.rejected?.(m.reason)
    } else if (m.type === 'dropped') {
      this.hooks.dropped?.(m.ops)
    } else if (m.type === 'validation') {
      this.validation = m
      this.report()
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
