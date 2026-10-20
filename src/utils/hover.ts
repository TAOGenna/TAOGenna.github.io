// Scenes that only move while their tile is hovered or focused: `play` on enter, `idle` on leave
// (idle must put the still frame back). Touch devices never hover, so they keep the still.
export function onTileHover(el: Element, play: () => void, idle: () => void) {
	const tile = el.closest('.tile') ?? el
	tile.addEventListener('pointerenter', play)
	tile.addEventListener('pointerleave', idle)
	tile.addEventListener('focusin', play)
	tile.addEventListener('focusout', idle)
}
