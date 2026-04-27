import { HardHat, PersonStanding, Flame, Activity, AlertOctagon } from 'lucide-react';

/**
 * AllPanel
 * ========
 * Displays the unified PPE + Pose + Fire result when mode === 'all'.
 *
 * Expected data shape:
 *   {
 *     mode:             "all",
 *     final_status:     "CRITICAL" | "HIGH RISK" | "UNSAFE" | "MODERATE" | "SAFE",
 *     ppe_status:       "SAFE" | "UNSAFE",
 *     pose_status:      "SAFE" | "MODERATE" | "UNSAFE",
 *     fire_status:      "SAFE" | "UNSAFE",
 *     accident_status:  "SAFE" | "WARN" | "CRITICAL",
 *     accident_events:  [{ frame, type, severity, confidence, reason }, ...],
 *     ppe_score:        number,
 *     pose_score:       number,
 *     fire_ratio:       number,
 *     total_frames:     number,
 *     fire_frames:      number,
 *   }
 */

const NON_CRITICAL_META = {
  'HIGH RISK': { color: 'unsafe', emoji: '🚨', label: 'HIGH RISK',     msg: 'Both PPE compliance and posture are unsafe. Immediate intervention required.' },
  UNSAFE:      { color: 'unsafe', emoji: '⚠️',  label: 'UNSAFE',        msg: 'A safety violation was detected. Review the annotated video for details.' },
  MODERATE:    { color: 'warn',   emoji: '⚡',  label: 'MODERATE RISK', msg: 'Minor compliance issue detected. Please review posture or PPE usage.' },
  SAFE:        { color: 'safe',   emoji: '✅',  label: 'ALL SAFE',      msg: 'No PPE, posture, or fire violations detected across all modules.' },
};

/** Build the banner meta dynamically so CRITICAL's label reflects what actually triggered it. */
function buildFinalMeta(data) {
  const status        = data.final_status    ?? 'UNKNOWN';
  const fireUnsafe    = data.fire_status     === 'UNSAFE';
  const accidentCrit  = data.accident_status === 'CRITICAL';
  const accidentEvts  = Array.isArray(data.accident_events) ? data.accident_events : [];

  if (status === 'CRITICAL') {
    if (fireUnsafe && accidentCrit) {
      return {
        color: 'unsafe', emoji: '🚨',
        label: 'CRITICAL — FIRE & WORKER ACCIDENT',
        msg:   'Fire on site AND a worker accident detected. Evacuate, then dispatch first aid.',
      };
    }
    if (fireUnsafe) {
      return {
        color: 'unsafe', emoji: '🚨',
        label: 'CRITICAL — FIRE HAZARD',
        msg:   'Fire detected on site. Evacuate immediately and activate emergency protocol.',
      };
    }
    if (accidentCrit) {
      // Pick the most-urgent event type for the message.
      const priority = ['MOTIONLESS_DOWN', 'CRUSHED', 'FALL', 'STRUCK', 'STUMBLE'];
      const pretty = {
        FALL:            'Worker fall',
        MOTIONLESS_DOWN: 'Worker is down and not moving',
        CRUSHED:         'Worker may be trapped or covered',
        STRUCK:          'Possible impact event',
        STUMBLE:         'Stumble',
      };
      const seen = new Set(accidentEvts.map(e => e.type));
      const top  = priority.find(p => seen.has(p)) ?? null;
      return {
        color: 'unsafe', emoji: '🚨',
        label: 'CRITICAL — WORKER ACCIDENT',
        msg:   top
          ? `${pretty[top]} detected. Dispatch first aid immediately and check the worker's condition.`
          : 'A worker accident was detected. Investigate the scene immediately.',
      };
    }
    // CRITICAL with no clear cause -- shouldn't really happen, but fail safe.
    return {
      color: 'unsafe', emoji: '🚨',
      label: 'CRITICAL',
      msg:   'A critical safety event was detected. Review the annotated video and act immediately.',
    };
  }

  return NON_CRITICAL_META[status]
    ?? { color: 'neutral', emoji: '❓', label: status, msg: '' };
}

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

  const ppeSt        = data.ppe_status      ?? '—';
  const poseSt       = data.pose_status     ?? '—';
  const fireSt       = data.fire_status     ?? '—';
  const accidentSt   = data.accident_status ?? 'SAFE';
  const accidentEvts = Array.isArray(data.accident_events) ? data.accident_events : [];
  const ppeScore     = data.ppe_score       ?? 0;
  const poseScore    = data.pose_score      ?? 0;
  const fireRatio    = data.fire_ratio      ?? 0;

  const meta         = buildFinalMeta(data);
  const ppeColor     = ppeSt  === 'SAFE' ? 'safe' : 'unsafe';
  const poseColor    = poseSt === 'SAFE' ? 'safe' : poseSt === 'MODERATE' ? 'warn' : 'unsafe';
  const fireColor    = fireSt === 'SAFE' ? 'safe' : 'unsafe';
  const accidentColor =
    accidentSt === 'CRITICAL' ? 'unsafe' :
    accidentSt === 'WARN'     ? 'warn'   : 'safe';

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

      {/* ── Module cards (PPE, Pose, Fire, optional Accident) ── */}
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
        {(accidentEvts.length > 0 || accidentSt !== 'SAFE') && (
          <ModuleCard
            icon={AlertOctagon}
            title="Accident Events"
            score={null}
            extra={`${accidentEvts.length} event${accidentEvts.length === 1 ? '' : 's'}`}
            status={accidentSt}
            color={accidentColor}
          />
        )}
      </div>
    </div>
  );
}
