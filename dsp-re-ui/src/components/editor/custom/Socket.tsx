import * as React from 'react'
import { ClassicPreset } from 'rete'
import styled from 'styled-components'

import { $socketmargin, $socketsize } from './vars'

const Styles = styled.div`
    display: inline-block;
    cursor: pointer;
    border: none;
    width: ${$socketsize}px;
    height: ${$socketsize}px;
    vertical-align: middle;
    z-index: 2;
    box-sizing: border-box;
    background: transparent;
`

const Hoverable = styled.div`
    border-radius: ${($socketsize + $socketmargin * 2) / 2.0}px;
    padding: ${$socketmargin}px;
    &:hover ${Styles} {
      border-width: 4px;
    }
`

export function CustomSocket<T extends ClassicPreset.Socket>(props: { data: T }) {
  return (
    <Hoverable>
      <Styles className="background-primary" title={props.data.name} />
    </Hoverable>
  )
}