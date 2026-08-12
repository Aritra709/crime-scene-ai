import { useEffect, useRef, useState } from "react";
import type { Detection } from "../types";

const CATEGORY_COLORS: Record<string, string> = {
  weapon: "#e03030",
  stain: "#b00020",
  vehicle: "#e07b00",
  person: "#2f80ed",
  container: "#0fa3b1",
  "personal item": "#8e44ad",
  "electronic device": "#27ae60",
  "discarded item": "#6c757d",
  ocr: "#f2c94c",
};

export function categoryColor(category: string): string {
  let colors = Object.values(CATEGORY_COLORS);
  let hash = 0;
  for (const ch of category) hash = (hash * 31 + ch.charCodeAt(0)) | 0;
  return CATEGORY_COLORS[category] ?? colors[Math.abs(hash) % colors.length];
}

interface Props {
  imageUrl: string;
  width: number;
  height: number;
  items: Detection[];
  hiddenIds: Set<string>;
  onToggle: (id: string) => void;
}

export default function ImageAnnotator({ imageUrl, width, height, items, hiddenIds, onToggle }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [img, setImg] = useState<HTMLImageElement | null>(null);

  useEffect(() => {
    const im = new Image();
    im.onload = () => setImg(im);
    im.src = imageUrl;
  }, [imageUrl]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !img) return;
    const iw = width > 0 ? width : img.naturalWidth;
    const ih = height > 0 ? height : img.naturalHeight;
    const cw = canvas.clientWidth || 640;
    const ch = (cw * ih) / iw;
    canvas.width = cw;
    canvas.height = ch;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, cw, ch);
    ctx.drawImage(img, 0, 0, cw, ch);
    const scale = cw / iw;
    for (const item of items) {
      if (hiddenIds.has(item.id)) continue;
      const { x1, y1, x2, y2 } = item.bbox;
      const color = categoryColor(item.category);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x1 * scale, y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale);
      if (x1 < width * scale && y1 > 14) {
        const label = `${item.class} ${Math.round(item.confidence * 100)}%`;
        ctx.font = "12px system-ui, sans-serif";
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = color;
        ctx.fillRect(x1 * scale, y1 * scale - 18, tw + 10, 18);
        ctx.fillStyle = "#fff";
        ctx.fillText(label, x1 * scale + 5, y1 * scale - 4);
      }
    }
  }, [img, items, hiddenIds, width, height]);

  const legend = [...new Set(items.map((i) => i.category))];

  return (
    <div className="annotator">
      <canvas ref={canvasRef} className="annotator-canvas" />
      {legend.length > 0 && (
        <div className="legend">
          {legend.map((cat) => (
            <button
              key={cat}
              type="button"
              className={`legend-chip${hiddenIds.has(`cat:${cat}`) ? " muted" : ""}`}
              onClick={() => onToggle(`cat:${cat}`)}
            >
              <span className="legend-dot" style={{ background: categoryColor(cat) }} />
              {cat}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}