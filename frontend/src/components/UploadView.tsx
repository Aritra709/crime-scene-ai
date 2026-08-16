import { useRef, useState } from "react";
import type { Analysis } from "../types";
import { uploadImage, uploadVideo, type VideoAnalysis } from "../api";

interface Props {
  onAnalyzed: (analysis: Analysis | VideoAnalysis) => void;
}

type MediaType = "image" | "video";

export default function UploadView({ onAnalyzed }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [officerId, setOfficerId] = useState(() => localStorage.getItem("officer_id") ?? "");
  const [gps, setGps] = useState<{ lat: number; lng: number } | null>(null);
  const [gpsNote, setGpsNote] = useState("");
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [mediaType, setMediaType] = useState<MediaType>("image");

  const pickGps = () => {
    if (!navigator.geolocation) {
      setGpsNote("Geolocation not available on this device/browser");
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setGps({ lat: pos.coords.latitude, lng: pos.coords.longitude });
        setGpsNote("GPS captured from device");
      },
      (err) => setGpsNote(`GPS error: ${err.message}`),
      { timeout: 8000 },
    );
  };

  const onFile = (f: File | undefined) => {
    if (!f) return;
    setPreviewUrl(URL.createObjectURL(f));
    setError("");
    setMediaType(f.type.startsWith("video/") ? "video" : "image");
  };

  const submit = async () => {
    const file = fileRef.current?.files?.[0];
    if (!file) {
      setError(mediaType === "video" ? "Choose a video first" : "Choose a photo first");
      return;
    }
    setBusy(true);
    setError("");
    localStorage.setItem("officer_id", officerId);
    try {
      const analysis = mediaType === "video"
        ? await uploadVideo(file, officerId, gps)
        : await uploadImage(file, officerId, gps);
      onAnalyzed(analysis);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const acceptType = mediaType === "video" ? "video/*" : "image/*";
  const dropHint = mediaType === "video" ? "Tap to choose a video (MP4/WebM)" : "Tap to choose a photo (JPEG/PNG)";

  return (
    <div className="card upload-card">
      <h2>New scene {mediaType === "video" ? "video" : "photo"}</h2>
      <p className="hint">
        Upload a scene {mediaType === "video" ? "video" : "photo"}. GPS + timestamp are auto-attached from device / EXIF. The AI
        draft goes to an officer review screen — nothing is logged without your confirmation.
      </p>

      <label className="field">
        <span>Officer ID</span>
        <input
          value={officerId}
          onChange={(e) => setOfficerId(e.target.value)}
          placeholder="e.g. ASI-1024"
        />
      </label>

      <div className="field">
        <span>Capture location</span>
        <button type="button" className="btn" onClick={pickGps}>
          {gps ? `GPS: ${gps.lat.toFixed(5)}, ${gps.lng.toFixed(5)}` : "Attach GPS from device"}
        </button>
        {gpsNote && <em className="hint">{gpsNote}</em>}
      </div>

      <div className="media-type-toggle">
        <label>
          <input
            type="radio"
            name="mediaType"
            value="image"
            checked={mediaType === "image"}
            onChange={() => setMediaType("image")}
          />
          Photo
        </label>
        <label>
          <input
            type="radio"
            name="mediaType"
            value="video"
            checked={mediaType === "video"}
            onChange={() => setMediaType("video")}
          />
          Video
        </label>
      </div>

      <label className="dropzone" onDragOver={(e) => e.preventDefault()}>
        <input
          ref={fileRef}
          type="file"
          accept={acceptType}
          onChange={(e) => onFile(e.target.files?.[0])}
          hidden
        />
        {previewUrl ? (
          mediaType === "video" ? (
            <video src={previewUrl} className="preview" controls />
          ) : (
            <img src={previewUrl} alt="selected" className="preview" />
          )
        ) : (
          <div className="drop-hint">{dropHint}</div>
        )}
      </label>

      {error && <div className="error">{error}</div>}

      <button type="button" className="btn primary" disabled={busy} onClick={submit}>
        {busy ? "Analyzing…" : `Run scene ${mediaType === "video" ? "video" : "photo"} analysis`}
      </button>
      <p className="hint">
        Runs fully offline: object/stain heuristics + offline reasoning draft. Set
        OPENAI_API_KEY on the backend for LLM-quality narratives.
      </p>
    </div>
  );
}