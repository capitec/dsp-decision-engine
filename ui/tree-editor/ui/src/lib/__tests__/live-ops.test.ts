import { describe, expect, it } from 'vitest'
// One source of truth: the pytest suite runs the same cases against the Python twin.
import rawCases from '../../../../backend/tests/data/live_ops_cases.json'
import { apply, conflicts, diff, isRelated, staleOps, type Json, type Op } from '../live-ops'

type Case = { name: string; kind: string } & Record<string, any>
const cases = rawCases as Case[]

describe('live ops shared cases', () => {
  it.each(cases.map((c) => [c.name, c] as const))('%s', (_name, c) => {
    if (c.kind === 'diff') {
      const ops = diff(c.old, c.new)
      expect(ops).toStrictEqual(c.ops)
      const [doc, dropped] = apply(c.old, ops)
      expect(doc).toStrictEqual(c.new)
      expect(dropped).toStrictEqual([])
    } else if (c.kind === 'apply') {
      const before = JSON.stringify(c.doc)
      const [doc, dropped] = apply(c.doc, c.ops)
      expect(doc).toStrictEqual(c.expected)
      expect(dropped).toStrictEqual(c.dropped)
      expect(JSON.stringify(c.doc)).toBe(before)
    } else if (c.kind === 'stale') {
      expect(staleOps(c.mine, c.concurrent)).toStrictEqual(c.dropped)
    } else if (c.kind === 'related') {
      expect(isRelated(c.a, c.b)).toBe(c.related)
      expect(isRelated(c.b, c.a)).toBe(c.related)
    } else if (c.kind === 'conflicts') {
      expect(conflicts(c.mine, c.theirs, c.base)).toStrictEqual(c.conflicts)
    } else {
      throw new Error(`unknown case kind ${c.kind}`)
    }
  })
})

describe('live ops apply', () => {
  it('returns a document that shares no values with the ops', () => {
    const ops: Op[] = [{ op: 'add', path: '/a', value: { x: [1] } }]
    const [doc] = apply({}, ops)
    ;((doc as { a: { x: Json[] } }).a.x).push(2)
    expect(ops[0].value).toStrictEqual({ x: [1] })
  })

  it('stores a __proto__ key as data, like the server', () => {
    const [doc, dropped] = apply({}, [{ op: 'add', path: '/__proto__', value: { polluted: true } }])
    expect(dropped).toStrictEqual([])
    expect(Object.keys(doc as object)).toStrictEqual(['__proto__'])
    expect(({} as Record<string, unknown>).polluted).toBeUndefined()
  })
})
