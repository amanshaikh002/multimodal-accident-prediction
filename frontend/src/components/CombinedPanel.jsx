import { ShieldAlert, ShieldCheck, Activity, HardHat, PersonStanding } from 'lucide-react';

/**
 * CombinedPanel
 * =============
 * Displays the merged PPE + Pose result when mode === 'combined'.
 * Shows a big final-status banner, plus side-by-side scores for each module.
 */

const STATUS_META = {
  'HIGH RISK': { color: 'unsafe', emoji: '🚨', label: 'HIGH RISK' },
  'UNSAFE':    { color: 'unsafe', emoji: '⚠️',  label: 'UNSAFE'    },
  'MODERATE':  { color: 'warn',   emoji: '⚡',  label: 'MODERATE'  },
  'SAFE':      { color: 'safe',   emoji: '✅',  label: 'SAFE'      },
};

function ScoreCard({ icon: Icon, title, score, status, color }) {
  const pct = typeof score === 'number' ? Math.min(Math.max(score, 0), 100) : 0;
  return (
    <div className="metric-card" style={{ flex: 1 }}>
      <div className="metric-card-header">
        <Icon size={14} color="var(--text-muted)" strokeWidth={2} />
        <span className="metric-label">{title}</span>
      </div>
      <div className={`metric-value mc-${color}`}>
        {typeof score === 'number' ? `${score.toFixed(1)}%` : '—'}
      </div>
      <div className="metric-bar-track">
        <div
          className={`metric-bar-fill mc-bar-${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.3rem' }}>
        Status: <strong style={{ color: `var(--${color})` }}>{status}</strong>
      </div>
    </div>
  );
}

export default function CombinedPanel({ data }) {
  if (!data) return null;

  const finalStatus  = data.final_status  ?? 'UNKNOWN';
  const ppeSt        = data.ppe_status    ?? '—';
  const poseSt       = data.pose_status   ?? '—';
  const ppeScore     = data.ppe_score     ?? 0;
  const poseScore    = data.pose_score    ?? 0;
  const message      = data.summary_message ?? '';

  const meta         = STATUS_META[finalStatus] ?? { color: 'neutral', emoji: '❓', label: finalStatus };
  const ppeColor     = ppeSt   === 'SAFE' ? 'safe'   : 'unsafe';
  const poseColor    = poseSt  === 'SAFE' ? 'safe'   : poseSt === 'MODERATE' ? 'warn' : 'unsafe';

  return (
    <div className="analysis-panel">
      <div className="section-label">
        <Activity size={13} strokeWidth={2} />
        Combined Risk Summary
      </div>

      {/* ── Big status banner ── */}
      <div
        className={`combined-status-banner mc-${meta.color}`}
        style={{
          padding:      '1rem 1.25rem',
          borderRadius: '10px',
          border:       `1.5px solid var(--${meta.color})`,
          background:   `color-mix(in srgb, var(--${meta.color}) 12%, transparent)`,
          marginBottom: '1rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <span style={{ fontSize: '1.5rem' }}>{meta.emoji}</span>
          <div>
            <div style={{ fontSize: '1.25rem', fontWeight: 800, color: `var(--${meta.color})` }}>
              {meta.label}
            </div>
            {message && (
              <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '0.2rem', lineHeight: 1.4 }}>
                {message}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── Side-by-side module scores ── */}
      <div style={{ display: 'flex', gap: '0.75rem' }}>
        <ScoreCard
          icon={HardHat}
          title="PPE Compliance"
          score={ppeScore}
          status={ppeSt}
          color={ppeColor}
        />
        <ScoreCard
          icon={PersonStanding}
          title="Pose Safety"
          score={poseScore}
          status={poseSt}
          color={poseColor}
        />
      </div>
    </div>
  );
}
