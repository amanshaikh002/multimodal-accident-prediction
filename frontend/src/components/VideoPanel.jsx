import { PlayCircle } from 'lucide-react';

const API = 'http://localhost:8000';

const VIDEO_URL = {
  ppe:  `${API}/output/ppe_annotated.mp4`,
  pose: `${API}/output/pose_annotated.mp4`,
};

/**
 * VideoPanel — shows the backend-annotated video for the selected mode.
 * The backend already draws bounding boxes + labels on the video frames,
 * so we just play it back here with a status badge overlay.
 */
export default function VideoPanel({ mode, videoUrl, overallSafe, hasResult }) {
  if (!hasResult || !videoUrl) {
    return (
      <div className="vp-empty">
        <PlayCircle size={48} strokeWidth={1.2} color="var(--border-accent)" />
        <p className="vp-empty-text">Annotated output will appear here after analysis</p>
      </div>
    );
  }

  return (
    <div className="vp-wrapper">
      {/* Status badge overlay */}
      <div className={`vp-status-badge ${overallSafe ? 'badge-safe' : 'badge-unsafe'}`}>
        <span className="vp-status-dot" />
        {overallSafe ? 'COMPLIANT' : 'VIOLATIONS DETECTED'}
      </div>

      <video
        key={videoUrl}
        className="vp-video"
        controls
        autoPlay
        muted
        playsInline
        src={videoUrl}
      >
        Your browser does not support HTML5 video.
      </video>

      <div className="vp-footer">
        <span className="vp-mode-tag">
          {mode === 'ppe' ? 'PPE Compliance Detection' : 'Pose Safety Detection'}
        </span>
        <span className="vp-hint">Bounding boxes &amp; labels rendered by AI backend</span>
      </div>
    </div>
  );
}
