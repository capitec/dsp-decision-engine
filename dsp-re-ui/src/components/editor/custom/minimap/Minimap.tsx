// File copied from https://github.com/retejs/react-plugin/blob/main/src/presets/minimap/components/Minimap.tsx
import * as React from 'react'
import { useCallback, useRef } from 'react'
import styled from 'styled-components'
import { useResizeObserver } from 'usehooks-ts'
import { Rect, Transform, Translate } from 'rete-react-plugin/_types/presets/minimap/types'
// @ts-ignore
import { px } from 'rete-react-plugin/presets/minimap/utils'
// @ts-ignore
import { MiniNode } from 'rete-react-plugin/presets/minimap/components/MiniNode'
// @ts-ignore
import { MiniViewport } from 'rete-react-plugin/presets/minimap/components/MiniViewport'

const Styles = styled.div<{ size: number }>`
    position: absolute;
    left: 24px;
    bottom: 24px;
    background: rgba(229, 234, 239, 0.65);
    padding: 20px;
    overflow: hidden;
    border: 1px solid #b1b7ff;
    border-radius: 8px;
    box-sizing: border-box;
`

type Props = {
  size: number
  ratio: number
  nodes: Rect[]
  viewport: Rect
  start(): Transform
  translate: Translate
  point(x: number, y: number): void
}

export function Minimap(props: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const { width = 0 } = useResizeObserver({
    ref
  })
  const containerWidth = ref.current?.clientWidth || width
  const scale = useCallback((v: number) => v * containerWidth, [containerWidth])

  return <Styles
    size={props.size}
    style={{
      width: px(props.size * props.ratio),
      height: px(props.size)
    }}
    onPointerDown={e => {
      e.stopPropagation()
      e.preventDefault()
    }}
    onDoubleClick={e => {
      e.stopPropagation()
      e.preventDefault()
      if (!ref.current) return
      const box = ref.current.getBoundingClientRect()
      const x = (e.clientX - box.left) / (props.size * props.ratio)
      const y = (e.clientY - box.top) / (props.size * props.ratio)

      props.point(x, y)
    }}
    ref={ref}
    data-testid="minimap"
  >
    {containerWidth
      ? props.nodes.map((node, i) => <MiniNode
        key={i}
        left={scale(node.left)}
        top={scale(node.top)}
        width={scale(node.width)}
        height={scale(node.height)}
      />)
      : null}
    <MiniViewport
      {...props.viewport}
      start={props.start}
      containerWidth={containerWidth}
      translate={props.translate}
    />
  </Styles>
}