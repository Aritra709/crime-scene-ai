import { useEffect, useRef, useState } from "react";
import type { Detection } from "../types";

interface Props {
  videoUrl: string;
  width: number;
  items: Detection[];
  hiddenIds: Set<string>;
  onToggle: (id: string) => void;
}

export default function VideoAnnotator({ videoUrl, width, items, hiddenIds, onToggle }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const animationRef = useRef<number>();

  // Get unique frame indices from detections
  const frameIndices = [...new Set(items.map((i) => i.frame_idx ?? 0))].sort((a, b) => a - b);
  const frameIndex = frameIndices.findIndex((f) => f >= currentTime) % frameIndices.length;
  const currentFrameIdx = frameIndices[frameIndex] ?? 0;
  const currentFrameDetections = items.filter((i) => (i.frame_idx ?? 0) === currentFrameIdx);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    video.onloadedmetadata = () => setDuration(video.duration);
  }, []);

  const drawFrame = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.paused || video.ended) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const scale = canvas.width / width;

    for (const item of currentFrameDetections) {
      if (hiddenIds.has(item.id)) continue;
      const { x1, y1, x2, y2 } = item.bbox;
      const color = categoryColor(item.category);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x1 * scale, y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale);

      const label = `${item.class} ${Math.round(item.confidence * 100)}%`;
      ctx.font = "12px system-ui, sans-serif";
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = color;
      ctx.fillRect(x1 * scale, y1 * scale - 18, tw + 10, 18);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, x1 * scale + 5, y1 * scale - 4);
    }

    animationRef.current = requestAnimationFrame(drawFrame);
  };

  useEffect(() => {
    if (isPlaying) {
      drawFrame();
    } else if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isPlaying, currentFrameDetections]);

  const togglePlay = () => {
    const video = videoRef.current;
    if (!video) return;
    if (isPlaying) {
      video.pause();
    } else {
      video.play();
    }
    setIsPlaying(!isPlaying);
  };

  const seek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    const video = videoRef.current;
    if (video) {
      video.currentTime = time;
      setCurrentTime(time);
    }
  };

  const formatTime = (t: number) => {
    const m = Math.floor(t / 60);
    const s = Math.floor(t % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  const legend = [...new Set(items.map((i) => i.category))];

  return (
    <div className="annotator">
      <div className="video-annotator">
        <video
          ref={videoRef}
          src={videoUrl}
          className="annotator-video"
          onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
        />
        <canvas ref={canvasRef} className="annotator-canvas" />
      </div>
      <div className="video-controls">
        <button type="button" className="btn" onClick={togglePlay}>
          {isPlaying ? "Pause" : "Play"}
        </button>
        <input
          type="range"
          min="0"
          max={duration || 100}
          step="0.1"
          value={currentTime}
          onChange={seek}
          className="seek-bar"
        />
        <span className="time-display">{formatTime(currentTime)} / {formatTime(duration)}</span>
      </div>
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

function categoryColor(category: string): string {
  let colors = Object.values(CATEGORY_COLORS);
  let hash = 0;
  for (const ch of category) hash = (hash * 31 + ch.charCodeAt(0)) | 0;
  return CATEGORY_COLORS[category] ?? colors[Math.abs(hash) % colors.length];
}