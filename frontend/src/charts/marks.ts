/**
 * Marker shapes, drawn to match matplotlib exactly.
 *
 * A segment's identity is its colour AND its shape — on a scatter every pair of colours is on
 * screen at once, and at eight groups the worst pair measures CVD ΔE 3.2, which is two segments a
 * reader with full colour vision cannot tell apart. Shape is what actually separates them.
 *
 * So the interactive chart has to use the SAME shape for a segment as the static one, or the two
 * pictures of the same data disagree about which group is which. matplotlib names its markers with
 * single characters; each is reproduced here as an SVG path on a unit square centred on the origin,
 * scaled by the caller.
 */

/** Marker path for a matplotlib marker code, sized so its AREA is `area` square pixels.
 *
 * Area rather than radius because area is how people read "how many" (Cleveland), and the segment
 * map sizes every dot by the number of people sharing that answer pattern.
 */
export function markerPath(marker: string, area: number): string {
  // Half-extent of a square of this area. Each shape below is then written to cover roughly the
  // same visual weight, matching matplotlib's own marker scaling closely enough that a segment
  // looks like itself in both renderings.
  const r = Math.sqrt(Math.max(area, 1)) / 2
  switch (marker) {
    case 's':
      return `M${-r},${-r}H${r}V${r}H${-r}Z`
    case '^':
      return `M0,${-r * 1.2}L${r * 1.15},${r * 0.9}H${-r * 1.15}Z`
    case 'v':
      return `M0,${r * 1.2}L${r * 1.15},${-r * 0.9}H${-r * 1.15}Z`
    case 'D':
      return `M0,${-r * 1.3}L${r * 1.15},0L0,${r * 1.3}L${-r * 1.15},0Z`
    case 'P': {
      // A filled plus: matplotlib's "P", not the thin line-cross "+".
      const a = r * 0.42
      const b = r * 1.15
      return `M${-a},${-b}H${a}V${-a}H${b}V${a}H${a}V${b}H${-a}V${a}H${-b}V${-a}H${-a}Z`
    }
    case 'X': {
      const a = r * 0.42
      const b = r * 1.1
      const pts: [number, number][] = [
        [-b, -b + a], [-b + a, -b], [0, -a], [b - a, -b], [b, -b + a], [a, 0],
        [b, b - a], [b - a, b], [0, a], [-b + a, b], [-b, b - a], [-a, 0],
      ]
      return `M${pts.map(([x, y]) => `${x},${y}`).join('L')}Z`
    }
    case '*': {
      const outer = r * 1.45
      const inner = outer * 0.42
      const pts: string[] = []
      for (let i = 0; i < 10; i += 1) {
        const radius = i % 2 === 0 ? outer : inner
        // Start at the top point, as matplotlib does, so the star reads the same way up.
        const angle = (Math.PI / 5) * i - Math.PI / 2
        pts.push(`${radius * Math.cos(angle)},${radius * Math.sin(angle)}`)
      }
      return `M${pts.join('L')}Z`
    }
    case 'o':
    default:
      // A circle as two arcs: keeps every mark a <path>, so one code path draws them all.
      return `M${-r},0A${r},${r} 0 1,0 ${r},0A${r},${r} 0 1,0 ${-r},0Z`
  }
}
