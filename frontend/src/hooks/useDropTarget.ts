import { useEffect, useState } from 'react'

/**
 * Drop a survey anywhere on the page, not just on the composer — people aim at the message box
 * but let go early, and a drop that lands nowhere looks like the app ignoring them.
 *
 * Returns whether a drag is currently over the window, which the composer uses to show it will
 * catch the file.
 */
export function useDropTarget(onDrop: (transfer: DataTransfer) => void): boolean {
  const [dragging, setDragging] = useState(false)

  useEffect(() => {
    function over(event: DragEvent) {
      event.preventDefault()
      setDragging(true)
    }
    // relatedTarget is null only when the pointer leaves the window itself, rather than crossing
    // between two elements inside it — otherwise the highlight flickers on every boundary.
    function leave(event: DragEvent) {
      if (event.relatedTarget === null) setDragging(false)
    }
    function drop(event: DragEvent) {
      event.preventDefault()
      setDragging(false)
      if (event.dataTransfer) onDrop(event.dataTransfer)
    }

    document.addEventListener('dragover', over)
    document.addEventListener('dragenter', over)
    document.addEventListener('dragleave', leave)
    document.addEventListener('drop', drop)
    return () => {
      document.removeEventListener('dragover', over)
      document.removeEventListener('dragenter', over)
      document.removeEventListener('dragleave', leave)
      document.removeEventListener('drop', drop)
    }
  }, [onDrop])

  return dragging
}
