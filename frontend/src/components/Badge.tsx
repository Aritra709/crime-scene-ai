interface Props {
  confidence: number;
  source: string;
}

export default function Badge({ confidence, source }: Props) {
  const tone = confidence >= 0.7 ? "high" : confidence >= 0.45 ? "mid" : "low";
  return (
    <span className={`badge badge-${tone}`} title={`source: ${source}`}>
      {(confidence * 100).toFixed(0)}%
    </span>
  );
}