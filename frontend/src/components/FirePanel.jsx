import { Flame, AlertTriangle, CheckCircle2, Activity } from 'lucide-react';

/**
 * FirePanel
 * =========
 * Displays the fire detection result when mode === 'fire' or inside AllPanel.
 *
 * Expected data shape:
 *   {
 *     module:       "fire",
 *     status:       "UNSAFE" | "SAFE",
 *     fire_ratio:   number,   // 0–1
 *     total_frames: number,
 *     fire_frames:  number,
 *   }
 */

const STATUS_META = {
  UNSAFE: {
    color:  'unsafe',
    emoji:  '🔥',
    label:  'FIRE DETECTED',
    bg:     'rgba(220, 38, 38, 0.12)',
    border: 'var(--unsafe)',
  },
  SAFE: {
    color:  'safe',
    emoji:  '✅',
    label:  'NO FIRE DETECTED',
    bg:     'rgba(16, 185, 129, 0.10)',
    border: 'var(--safe)',
  },
};

function StatCard({ icon: Icon, label, value, color = 'neutral' }) {
  return (
    <div className="metric-card" style={{ flex: 1 }}>
      <div className="metric-card-header">
        <Icon size={14} color="var(--text-muted)" strokeWidth={2} />
        <span className="metric-label">{label}</span>
      </div>
      <div className={`metric-value mc-${color}`}>{value}</div>
    </div>
  );
}

export default function FirePanel({ data }) {
  if (!data) return null;

  const status       = data.status       ?? 'SAFE';
  const fireRatio    = data.fire_ratio   ?? 0;
  const totalF       = data.total_frames ?? 0;
  const fireF        = data.fire_frames  ?? 0;
  const firePct      = (fireRatio * 100).toFixed(1);
  const message      = data.message      ?? null;

  const meta     = STATUS_META[status] ?? STATUS_META.SAFE;
  const isUnsafe = status === 'UNSAFE';

  return (
    <div className="analysis-panel">
      <div className="section-label">
        <Flame size={13} strokeWidth={2} />
        Fire Hazard Analysis
      </div>

      {/* ── Status Banner ── */}
      <div
        style={{
          padding:      '1rem 1.25rem',
          borderRadius: '10px',
          border:       `1.5px solid ${meta.border}`,
          background:   meta.bg,
          marginBottom: '1rem',
          display:      'flex',
          alignItems:   'center',
          gap:          '0.75rem',
        }}
      >
        <span style={{ fontSize: '1.8rem', lineHeight: 1 }}>{meta.emoji}</span>
        <div>
          <div
            style={{
              fontSize:   '1.15rem',
              fontWeight: 800,
              color:      `var(--${meta.color})`,
              letterSpacing: '0.03em',
            }}
          >
            {meta.label}
          </div>
          <div style={{ fontSize: '0.80rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
            {message
              ? message
              : isUnsafe
              ? `Fire detected in ${firePct}% of processed frames — evacuate immediately!`
              : `No fire hazard detected across ${totalF} sampled frames.`}
          </div>
        </div>
      </div>

      {/* ── Alert Badge ── */}
      {isUnsafe && (
        <div
          style={{
            display:      'flex',
            alignItems:   'center',
            gap:          '0.5rem',
            padding:      '0.5rem 0.9rem',
            borderRadius: '8px',
            background:   'rgba(220, 38, 38, 0.15)',
            color:        'var(--unsafe)',
            fontSize:     '0.82rem',
            fontWeight:   600,
            marginBottom: '1rem',
          }}
        >
          <AlertTriangle size={14} strokeWidth={2.5} />
          CRITICAL ALERT — Initiate fire safety protocol immediately
        </div>
      )}

      {/* ── Stat Cards ── */}
      <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
        <StatCard
          icon={Activity}
          label="Fire Coverage"
          value={`${firePct}%`}
          color={isUnsafe ? 'unsafe' : 'safe'}
        />
        <StatCard
          icon={Flame}
          label="Fire Frames"
          value={fireF}
          color={fireF > 0 ? 'unsafe' : 'safe'}
        />
        <StatCard
          icon={CheckCircle2}
          label="Total Frames"
          value={totalF}
          color="neutral"
        />
      </div>

      {/* ── Fire ratio bar ── */}
      <div style={{ marginTop: '1rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Fire Ratio</span>
          <span style={{ fontSize: '0.75rem', color: `var(--${meta.color})`, fontWeight: 700 }}>
            {firePct}%
          </span>
        </div>
        <div className="metric-bar-track">
          <div
            className={`metric-bar-fill mc-bar-${meta.color}`}
            style={{ width: `${Math.min(parseFloat(firePct), 100)}%` }}
          />
        </div>
      </div>
    </div>
  );
}
