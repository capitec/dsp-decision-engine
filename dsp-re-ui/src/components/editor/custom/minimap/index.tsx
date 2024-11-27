// Copied from https://github.com/retejs/react-plugin/blob/main/src/presets/minimap/index.tsx
import * as React from 'react'
import { BaseSchemes } from 'rete'

import { RenderPreset } from 'rete-react-plugin/_types/presets/types';
import { MinimapRender } from 'rete-react-plugin/_types/presets/minimap/types'
import { Minimap } from './Minimap'

/**
 * Preset for rendering minimap.
 */
export function setup<Schemes extends BaseSchemes, K extends MinimapRender>(props?: { size?: number }): RenderPreset<Schemes, K> {
  return {
    render(context) {
      if (context.data.type === 'minimap') {
        return <Minimap
          nodes={context.data.nodes}
          size={props?.size || 200}
          ratio={context.data.ratio}
          viewport={context.data.viewport}
          start={context.data.start}
          translate={context.data.translate}
          point={context.data.point}
        />
      }
    }
  }
}