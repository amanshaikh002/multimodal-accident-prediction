import { PlayCircle } from 'lucide-react';


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

      {/*
        Both `src` on the <video> element AND a <source> child are specified so
        that older browsers that don't read the element-level `src` still work.
        The explicit type="video/mp4" is mandatory — without it some browsers
        will refuse to probe/play the file at all (blank screen symptom).
      */}
      <video
        key={videoUrl}
        className="vp-video"
        controls
        autoPlay
        muted
        playsInline
      >
        <source src={videoUrl} type="video/mp4" />
        Your browser does not support HTML5 video.
      </video>

      <div className="vp-footer">
        <span className="vp-mode-tag">
          {mode === 'ppe'      ? 'PPE Compliance Detection'  :
           mode === 'pose'     ? 'Pose Safety Detection'     :
           mode === 'fire'     ? 'Fire Hazard Detection'     :
           mode === 'combined' ? 'PPE + Pose Detection'      :
           mode === 'all'      ? 'Full Platform Detection'   :
                                 `${mode.toUpperCase()} Detection`}
        </span>
        <span className="vp-hint">Bounding boxes &amp; labels rendered by AI backend</span>
      </div>
    </div>
  );
}
