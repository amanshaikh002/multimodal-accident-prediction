import { AlertCircle, CheckCircle } from 'lucide-react';

/**
 * ViolationsList — deduplicated, timeline-style list of detected violations.
 * Each entry shows frame number → violation reason with severity colour coding.
 */
export default function ViolationsList({ items }) {
  // Deduplicate by reason text — keep first-seen frame
  const seen = new Map();
  for (const v of items) {
    if (!seen.has(v.reason)) seen.set(v.reason, v);
  }
  const unique = Array.from(seen.values());

  return (
    <div className="side-panel">
      <div className="section-label">
        <AlertCircle size={13} strokeWidth={2} style={{ flexShrink: 0 }} />
        Detected Violations
        {unique.length > 0 && (
          <span className="count-badge count-unsafe">{unique.length}</span>
        )}
      </div>

      {unique.length === 0 ? (
        <div className="empty-state">
          <CheckCircle size={22} color="var(--safe)" strokeWidth={1.5} />
          <span>No violations detected</span>
        </div>
      ) : (
        <ul className="violation-list">
          {unique.map((v, i) => (
            <li key={i} className={`viol-item sev-${v.severity ?? 'high'}`}>
              <span className="viol-frame">F{v.frame}</span>
              <div className="viol-body">
                <span className="viol-reason">{v.reason}</span>
                {v.item && v.item !== v.reason && (
                  <span className="viol-item-tag">{v.item}</span>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
