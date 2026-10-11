import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { apply, conflicts, diff, isRelated, rebaseOrders, reconcile, staleOps, touched, type Json, type Op } from '../live-ops'

// One source of truth: the pytest suite runs the same vectors through the Python extension.
const file = resolve(import.meta.dirname, '../../../../merge-core-bench/vectors/op_level.json')
type Case = { name: string; kind: string; semantics: string } & Record<string, any>
const all: Case[] = JSON.parse(readFileSync(file, 'utf8')).cases

// JavaScript itself cannot hold these: numbers past 2**53, and integer-like keys are listed first.
const unrepresentable = (c: Case) => c.semantics === 'large-int' || c.name === 'an added integer-like key goes last'
const cases = all.filter((c) => !unrepresentable(c))

const fns: Record<string, (...args: any[]) => unknown> = {
  is_related: isRelated, stale_ops: staleOps, conflicts, touched, reconcile, rebase_orders: rebaseOrders, diff,
}

describe('live ops shared cases', () => {
  it.each(cases.map((c) => [c.name, c] as const))('%s', (_name, c) => {
    const before = JSON.stringify(c)
    if (c.kind === 'diff') {
      const ops = diff(c.old, c.new)
      expect(ops).toStrictEqual(c.ops)
      const [doc, dropped] = apply(c.old, ops)
      expect(doc).toStrictEqual(c.new)
      expect(dropped).toStrictEqual([])
    } else if (c.kind === 'apply') {
      const [doc, dropped] = apply(c.doc, c.ops)
      expect(doc).toStrictEqual(c.expected)
      expect(dropped).toStrictEqual(c.dropped)
      dropped.forEach((d) => expect(c.ops).toContain(d))
      if (c.keys) expect(Object.keys(doc as object)).toStrictEqual(c.keys)
    } else if (c.kind === 'stale') {
      const stale = staleOps(c.mine, c.concurrent)
      expect(stale).toStrictEqual(c.dropped)
      stale.forEach((d) => expect(c.mine).toContain(d))
    } else if (c.kind === 'related') {
      expect(isRelated(c.a, c.b)).toBe(c.related)
      expect(isRelated(c.b, c.a)).toBe(c.related)
    } else if (c.kind === 'conflicts') {
      expect(conflicts(c.mine, c.theirs, c.base)).toStrictEqual(c.conflicts)
    } else if (c.kind === 'rebase') {
      expect(rebaseOrders(c.mine, c.base, c.current)).toStrictEqual(c.expected)
    } else if (c.kind === 'reconcile') {
      expect(reconcile(c.current, c.given)).toStrictEqual(c.expected)
    } else if (c.kind === 'touched') {
      expect(touched(c.ops)).toStrictEqual(c.expected)
    } else if (c.kind === 'raises') {
      expect(() => fns[c.fn](...c.args)).toThrow()
    } else {
      throw new Error(`unknown case kind ${c.kind}`)
    }
    expect(JSON.stringify(c)).toBe(before)
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
