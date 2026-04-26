import { ShieldAlert, HardHat, PersonStanding, Flame, Activity } from 'lucide-react';

/**
 * AllPanel
 * ========
 * Displays the unified PPE + Pose + Fire result when mode === 'all'.
 *
 * Expected data shape:
 *   {
 *     mode:         "all",
 *     final_status: "CRITICAL" | "HIGH RISK" | "UNSAFE" | "MODERATE" | "SAFE",
 *     ppe_status:   "SAFE" | "UNSAFE",
 *     pose_status:  "SAFE" | "MODERATE" | "UNSAFE",
 *     fire_status:  "SAFE" | "UNSAFE",
 *     ppe_score:    number,
 *     pose_score:   number,
 *     fire_ratio:   number,
 *     total_frames: number,
 *     fire_frames:  number,
 *   }
 */

const FINAL_META = {
  CRITICAL:   { color: 'unsafe',  emoji: '🚨', label: 'CRITICAL — FIRE HAZARD',  msg: 'Fire detected on site. Evacuate immediately and activate emergency protocol.' },
  'HIGH RISK':{ color: 'unsafe',  emoji: '🚨', label: 'HIGH RISK',               msg: 'Both PPE compliance and posture are unsafe. Immediate intervention required.' },
  UNSAFE:     { color: 'unsafe',  emoji: '⚠️',  label: 'UNSAFE',                 msg: 'A safety violation was detected. Review the annotated video for details.' },
  MODERATE:   { color: 'warn',    emoji: '⚡',  label: 'MODERATE RISK',           msg: 'Minor compliance issue detected. Please review posture or PPE usage.' },
  SAFE:       { color: 'safe',    emoji: '✅',  label: 'ALL SAFE',                msg: 'No PPE, posture, or fire violations detected across all modules.' },
};

function ModuleCard({ icon: Icon, title, score, status, color, extra }) {
  const pct = typeof score === 'number' ? Math.min(Math.max(score, 0), 100) : null;
  return (
    <div className="metric-card" style={{ flex: 1 }}>
      <div className="metric-card-header">
        <Icon size={14} color="var(--text-muted)" strokeWidth={2} />
        <span className="metric-label">{title}</span>
      </div>
      {pct !== null ? (
        <>
          <div className={`metric-value mc-${color}`}>{pct.toFixed(1)}%</div>
          <div className="metric-bar-track">
            <div className={`metric-bar-fill mc-bar-${color}`} style={{ width: `${pct}%` }} />
          </div>
        </>
      ) : (
        <div className={`metric-value mc-${color}`}>{extra ?? '—'}</div>
      )}
      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.3rem' }}>
        Status: <strong style={{ color: `var(--${color})` }}>{status}</strong>
      </div>
    </div>
  );
}

export default function AllPanel({ data }) {
  if (!data) return null;

  const finalStatus = data.final_status ?? 'UNKNOWN';
  const ppeSt       = data.ppe_status   ?? '—';
  const poseSt      = data.pose_status  ?? '—';
  const fireSt      = data.fire_status  ?? '—';
  const ppeScore    = data.ppe_score    ?? 0;
  const poseScore   = data.pose_score   ?? 0;
  const fireRatio   = data.fire_ratio   ?? 0;

  const meta      = FINAL_META[finalStatus] ?? { color: 'neutral', emoji: '❓', label: finalStatus, msg: '' };
  const ppeColor  = ppeSt  === 'SAFE' ? 'safe'  : 'unsafe';
  const poseColor = poseSt === 'SAFE' ? 'safe'  : poseSt === 'MODERATE' ? 'warn' : 'unsafe';
  const fireColor = fireSt === 'SAFE' ? 'safe'  : 'unsafe';

  return (
    <div className="analysis-panel">
      <div className="section-label">
        <Activity size={13} strokeWidth={2} />
        Full Platform Safety Summary
      </div>

      {/* ── Big status banner ── */}
      <div
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
            <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '0.2rem', lineHeight: 1.4 }}>
              {meta.msg}
            </div>
          </div>
        </div>
      </div>

      {/* ── Three module cards ── */}
      <div style={{ display: 'flex', gap: '0.6rem', flexWrap: 'wrap' }}>
        <ModuleCard
          icon={HardHat}
          title="PPE Compliance"
          score={ppeScore}
          status={ppeSt}
          color={ppeColor}
        />
        <ModuleCard
          icon={PersonStanding}
          title="Pose Safety"
          score={poseScore}
          status={poseSt}
          color={poseColor}
        />
        <ModuleCard
          icon={Flame}
          title="Fire Detection"
          score={null}
          extra={`${(fireRatio * 100).toFixed(1)}% fire`}
          status={fireSt}
          color={fireColor}
        />
      </div>
    </div>
  );
}
