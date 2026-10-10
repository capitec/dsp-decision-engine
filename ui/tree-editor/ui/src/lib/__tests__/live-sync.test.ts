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
  const hooks = { read: () => editor, show: vi.fn((d: Json) => { editor = d; shown.push(d) }), rejected: vi.fn(), settled: vi.fn() }
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

  it('catches up again when a version is missing', () => {
    const { sync, socket } = setup()

    socket().receive({ type: 'applied', version: 3, client_id: 'other', msg_id: 'm', ops: [] })

    expect(sync.version).toBe(1)
    expect(socket().sent[socket().sent.length - 1]).toMatchObject({ type: 'hello', since: 1 })
  })

  it('after a reconnect resends only the messages the catch-up did not acknowledge, after it is ready', () => {
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

    expect(socket().ops().map((m) => m.id)).toEqual([second.id])
    expect(socket().ops()[0].base).toBe(1)
  })
})
