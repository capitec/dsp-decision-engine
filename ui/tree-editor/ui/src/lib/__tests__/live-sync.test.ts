import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { LiveSync } from '../live-sync'
import type { Json } from '../live-ops'

class FakeSocket {
  sent: any[] = []
  onopen: any = null
  onmessage: any = null
  onclose: any = null
  send(text: string) { this.sent.push(JSON.parse(text)) }
  close() { this.onclose?.() }
  receive(message: object) { this.onmessage({ data: JSON.stringify(message) }) }
  ops() { return this.sent.filter((m) => m.type === 'ops') }
}

const doc = (a: string, b = 'B'): Json => ({ rules: [{ rule: { id: 'a' }, meta: { name: a } }, { rule: { id: 'b' }, meta: { name: b } }] })

function setup() {
  const sockets: FakeSocket[] = []
  let editor: Json = doc('A')
  const shown: Json[] = []
  const hooks = {
    read: () => editor, show: vi.fn((d: Json) => { editor = d; shown.push(d) }), rejected: vi.fn(), settled: vi.fn(),
    merged: vi.fn(), offlineRejected: vi.fn(), validation: vi.fn(),
  }
  const sync = new LiveSync('ws://x', doc('A'), 1, hooks, 150, () => { const s = new FakeSocket(); sockets.push(s); return s as any })
  const socket = () => sockets[sockets.length - 1]
  socket().onopen()
  socket().receive({ type: 'ready', version: 1 })
  return { sync, hooks, socket, sockets, shown, type: (d: Json) => { editor = d; sync.edit() } }
}

beforeEach(() => vi.useFakeTimers())
afterEach(() => vi.useRealTimers())

describe('LiveSync', () => {
  it('says hello from its version, pushes an edit after the pause and settles on its own ack', () => {
    const { sync, hooks, socket, type } = setup()
    expect(socket().sent[0]).toMatchObject({ type: 'hello', since: 1 })

    type(doc('Mine'))
    expect(socket().ops()).toEqual([])
    vi.advanceTimersByTime(150)
    const [sent] = socket().ops()
    expect(sent).toMatchObject({ base: 1, ops: [{ op: 'replace', path: '/rules/a/meta/name', value: 'Mine' }] })
    expect(sync.idle()).toBe(false)

    socket().receive({ type: 'applied', version: 2, client_id: sync.clientId, msg_id: sent.id, ops: sent.ops })
    expect(sync.idle()).toBe(true)
    expect(sync.version).toBe(2)
    expect(hooks.settled).toHaveBeenCalledWith(2)
  })

  it('sends a field the editor holds as undefined as no field, as the server will store it', () => {
    const { socket, type } = setup()

    type({ ...(doc('A') as object), extra: undefined } as unknown as Json)
    vi.advanceTimersByTime(150)

    expect(socket().ops()).toEqual([])
  })

  it('pushes an edit not yet sent before it applies a remote version, and shows both', () => {
    const { sync, socket, shown, type } = setup()
    type(doc('Mine'))

    socket().receive({ type: 'applied', version: 2, client_id: 'other', msg_id: 'm', ops: [{ op: 'replace', path: '/rules/b/meta/name', value: 'Theirs' }] })

    expect(socket().ops()).toHaveLength(1)
    expect(shown[shown.length - 1]).toEqual(doc('Mine', 'Theirs'))
    expect(sync.confirmed).toEqual(doc('A', 'Theirs'))
  })

  it('drops a refused message from the editor and says why', () => {
    const { hooks, socket, shown, type } = setup()
    type(doc('Refused'))
    vi.advanceTimersByTime(150)

    socket().receive({ type: 'rejected', msg_id: socket().ops()[0].id, reason: 'read-only access' })

    expect(shown[shown.length - 1]).toEqual(doc('A'))
    expect(hooks.rejected).toHaveBeenCalledWith('read-only access')
  })

  it('reports validation only for the version it shows, also when the validation arrives first', () => {
    const { hooks, socket } = setup()

    socket().receive({ type: 'validation', version: 2, errors: ['ahead'] })
    expect(hooks.validation).not.toHaveBeenCalled()
    socket().receive({ type: 'applied', version: 2, client_id: 'other', msg_id: 'm', ops: [] })
    socket().receive({ type: 'validation', version: 1, errors: ['behind'] })

    expect(hooks.validation.mock.calls).toEqual([[['ahead'], 2]])
  })

  it('catches up again when a version is missing', () => {
    const { sync, socket } = setup()

    socket().receive({ type: 'applied', version: 3, client_id: 'other', msg_id: 'm', ops: [] })

    expect(sync.version).toBe(1)
    expect(socket().sent[socket().sent.length - 1]).toMatchObject({ type: 'hello', since: 1 })
  })

  it('after a reconnect sends the messages the catch-up did not acknowledge as one offline batch, after it is ready', () => {
    const { sync, socket, sockets, type } = setup()
    type(doc('First'))
    vi.advanceTimersByTime(150)
    type(doc('Second'))
    vi.advanceTimersByTime(150)
    const [first, second] = socket().ops()

    socket().close()
    vi.advanceTimersByTime(250)
    expect(sockets).toHaveLength(2)
    socket().onopen()
    socket().receive({ type: 'applied', version: 2, client_id: sync.clientId, msg_id: first.id, ops: first.ops })
    expect(socket().ops()).toEqual([])
    socket().receive({ type: 'ready', version: 2 })

    expect(socket().ops()).toEqual([expect.objectContaining({ base: 1, ops: second.ops, offline: true })])
  })

  function offline(edit: Json) {
    const s = setup()
    s.socket().close()
    s.type(edit)
    vi.advanceTimersByTime(250)
    s.socket().onopen()
    return s
  }

  it('sends the edits made while disconnected as one offline batch based on the version at the drop', () => {
    const { sync, socket, type } = offline(doc('Offline'))
    type(doc('Offline', 'Also offline'))
    vi.advanceTimersByTime(150)
    expect(socket().ops()).toEqual([])

    socket().receive({ type: 'applied', version: 2, client_id: 'other', msg_id: 'm', ops: [{ op: 'add', path: '/mode', value: 'all' }] })
    socket().receive({ type: 'ready', version: 2 })

    const batch = socket().ops()
    expect(batch).toHaveLength(1)
    expect(batch[0]).toMatchObject({ base: 1, offline: true, ops: [
      { op: 'replace', path: '/rules/a/meta/name', value: 'Offline' },
      { op: 'replace', path: '/rules/b/meta/name', value: 'Also offline' },
    ] })
    expect(sync.pending).toHaveLength(1)
  })

  it('says so when the offline batch is merged', () => {
    const { sync, hooks, socket } = offline(doc('Offline'))
    socket().receive({ type: 'ready', version: 1 })
    const [batch] = socket().ops()

    socket().receive({ type: 'applied', version: 2, client_id: sync.clientId, msg_id: batch.id, ops: batch.ops })

    expect(hooks.merged).toHaveBeenCalled()
    expect(sync.idle()).toBe(true)
  })

  it('on a rejected offline batch shows the server document and hands over my version and the conflicts', () => {
    const { hooks, socket, shown } = offline(doc('Offline'))
    const theirs = { type: 'applied', version: 2, client_id: 'other', msg_id: 'm', ops: [{ op: 'replace', path: '/rules/a/meta/name', value: 'Theirs' }] }
    socket().receive(theirs)
    socket().receive({ type: 'ready', version: 2 })
    const [batch] = socket().ops()

    socket().receive({ type: 'rejected', msg_id: batch.id, reason: 'overlap', conflicts: ['/rules/a/meta/name'] })

    expect(shown[shown.length - 1]).toEqual(doc('Theirs'))
    expect(hooks.rejected).not.toHaveBeenCalled()
    expect(hooks.offlineRejected).toHaveBeenCalledWith({ reason: 'overlap', conflicts: ['/rules/a/meta/name'], doc: doc('Offline') })
  })

  it('resends an offline batch whose answer was lost under the same id', () => {
    const { socket } = offline(doc('Offline'))
    socket().receive({ type: 'ready', version: 1 })
    const [batch] = socket().ops()

    socket().close()
    vi.advanceTimersByTime(500)
    socket().onopen()
    socket().receive({ type: 'ready', version: 1 })

    expect(socket().ops()).toEqual([batch])
  })

  it('treats a connection that answers no ping as lost and connects again', () => {
    const { socket, sockets } = setup()
    vi.advanceTimersByTime(10000)
    expect(socket().sent[socket().sent.length - 1]).toEqual({ type: 'ping' })
    socket().receive({ type: 'pong' })
    vi.advanceTimersByTime(10000)
    expect(sockets).toHaveLength(1)

    vi.advanceTimersByTime(10000)
    vi.advanceTimersByTime(250)

    expect(sockets).toHaveLength(2)
  })
})
