import { ClipboardCheck, ThumbsUp } from 'lucide-react';

const SEV_ICON = {
  high:   '●',
  medium: '●',
  low:    '●',
};

const SEV_CLASS = {
  high:   'rec-high',
  medium: 'rec-medium',
  low:    'rec-low',
};

/**
 * Recommendations — deduplicated safety actions derived from violations.
 * Each card: trigger label → 1-2 line instruction.
 */
export default function Recommendations({ items }) {
  // Deduplicate by suggestion text
  const seen = new Map();
  for (const v of items) {
    if (v.suggestion && !seen.has(v.suggestion)) seen.set(v.suggestion, v);
  }
  const unique = Array.from(seen.values());

  return (
    <div className="side-panel">
      <div className="section-label">
        <ClipboardCheck size={13} strokeWidth={2} style={{ flexShrink: 0 }} />
        Safety Recommendations
        {unique.length > 0 && (
          <span className="count-badge count-accent">{unique.length}</span>
        )}
      </div>

      {unique.length === 0 ? (
        <div className="empty-state">
          <ThumbsUp size={22} color="var(--safe)" strokeWidth={1.5} />
          <span>All safety checks passed</span>
        </div>
      ) : (
        <ul className="rec-list">
          {unique.map((s, i) => {
            const sev = s.severity ?? 'medium';
            return (
              <li key={i} className={`rec-item ${SEV_CLASS[sev]}`}>
                <span className={`rec-dot sev-dot-${sev}`}>{SEV_ICON[sev]}</span>
                <div className="rec-body">
                  <span className="rec-trigger">{s.reason}</span>
                  <span className="rec-text">{s.suggestion}</span>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
