import { useEffect, useMemo, useRef, useState } from 'react'

const COLORS = {
  document: '#63b3ff', party: '#7bd88f', defined_term: '#ffd166', obligation: '#c58cff',
  attribute: '#5fd3c4', event: '#ff9d4d', equation: '#ff5f6b', finding: '#8b97ad',
}

/** Small deterministic force layout — enough structure to read the graph, no dependencies. */
function layout(nodes, edges, width, height, iterations = 220) {
  const index = new Map(nodes.map((n, i) => [n.id, i]))
  const points = nodes.map((n, i) => ({
    x: width / 2 + Math.cos((i / nodes.length) * Math.PI * 2) * (width / 3.2),
    y: height / 2 + Math.sin((i / nodes.length) * Math.PI * 2) * (height / 3.2),
    vx: 0, vy: 0, node: n,
  }))
  const links = edges
    .map((e) => [index.get(e.source), index.get(e.target)])
    .filter(([a, b]) => a !== undefined && b !== undefined)

  for (let step = 0; step < iterations; step += 1) {
    const cooling = 1 - step / iterations
    for (let i = 0; i < points.length; i += 1) {
      for (let j = i + 1; j < points.length; j += 1) {
        const dx = points[j].x - points[i].x
        const dy = points[j].y - points[i].y
        const distanceSq = Math.max(40, dx * dx + dy * dy)
        const force = 2600 / distanceSq
        const nx = (dx / Math.sqrt(distanceSq)) * force
        const ny = (dy / Math.sqrt(distanceSq)) * force
        points[i].vx -= nx; points[i].vy -= ny
        points[j].vx += nx; points[j].vy += ny
      }
    }
    for (const [a, b] of links) {
      const dx = points[b].x - points[a].x
      const dy = points[b].y - points[a].y
      const distance = Math.sqrt(dx * dx + dy * dy) || 1
      const force = (distance - 70) * 0.015
      const nx = (dx / distance) * force
      const ny = (dy / distance) * force
      points[a].vx += nx; points[a].vy += ny
      points[b].vx -= nx; points[b].vy -= ny
    }
    for (const point of points) {
      point.x = Math.min(width - 20, Math.max(20, point.x + point.vx * cooling))
      point.y = Math.min(height - 20, Math.max(20, point.y + point.vy * cooling))
      point.vx *= 0.55; point.vy *= 0.55
    }
  }
  return { points, links }
}

export default function GraphView({ graph }) {
  const wrapper = useRef(null)
  const [width, setWidth] = useState(900)
  const [hover, setHover] = useState(null)
  const height = 520

  useEffect(() => {
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width))
    if (wrapper.current) observer.observe(wrapper.current)
    return () => observer.disconnect()
  }, [])

  const { points, links } = useMemo(
    () => layout(graph.nodes.slice(0, 160), graph.edges, width, height),
    [graph, width])

  return (
    <div ref={wrapper}>
      <svg className="graph-svg" viewBox={`0 0 ${width} ${height}`}>
        {links.map(([a, b], i) => (
          <line key={i} x1={points[a].x} y1={points[a].y} x2={points[b].x} y2={points[b].y}
                stroke="#1e2637" strokeWidth="1" />
        ))}
        {points.map((point, i) => (
          <g key={i} onMouseEnter={() => setHover(point.node)} onMouseLeave={() => setHover(null)}>
            <circle cx={point.x} cy={point.y}
                    r={4 + Math.min(6, (point.node.degree || 1) * 0.7)}
                    fill={COLORS[point.node.kind] || '#8b97ad'}
                    opacity={hover && hover.id !== point.node.id ? 0.35 : 0.95} />
            {(point.node.degree || 0) > 3 && (
              <text x={point.x + 9} y={point.y + 3} fontSize="9" fill="#8b97ad">
                {point.node.label.slice(0, 26)}
              </text>
            )}
          </g>
        ))}
      </svg>
      <div className="legend">
        {Object.entries(COLORS).map(([kind, color]) => (
          <span key={kind} style={{ color }}>{kind.replace('_', ' ')}</span>
        ))}
      </div>
      {hover && (
        <div className="card tight small" style={{ marginTop: '.5rem' }}>
          <Badgeish kind={hover.kind} /> <strong>{hover.label}</strong>
          <div className="muted mono" style={{ marginTop: '.25rem' }}>
            {Object.entries(hover).filter(([k]) => !['id', 'label', 'kind'].includes(k))
              .map(([k, v]) => `${k}=${v}`).join(' · ')}
          </div>
        </div>
      )}
    </div>
  )
}

function Badgeish({ kind }) {
  return <span className="badge" style={{ color: COLORS[kind] }}>{kind.replace('_', ' ')}</span>
}
