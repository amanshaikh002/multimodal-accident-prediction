import { Volume2, AlertTriangle, CheckCircle2, Activity, Clock } from 'lucide-react';

/**
 * SoundPanel
 * ==========
 * Displays the anomaly sound detection result when mode === 'sound'.
 *
 * Expected data shape:
 *   {
 *     module:           "sound",
 *     status:           "UNSAFE" | "SAFE",
 *     anomaly_detected: bool,
 *     anomaly_ratio:    number,   // 0–1
 *     total_windows:    number,
 *     anomaly_windows:  number,
 *     duration_sec:     number,
 *     events: [
 *       { start_sec, end_sec, duration_sec, avg_confidence,
 *         max_confidence, max_anomaly_prob }, ...
 *     ],
 *     message:      string,
 *   }
 */

const STATUS_META = {
  UNSAFE: {
    color:  'unsafe',
    emoji:  '🔊',
    label:  'ANOMALY DETECTED',
    bg:     'rgba(220, 38, 38, 0.12)',
    border: 'var(--unsafe)',
  },
  SAFE: {
    color:  'safe',
    emoji:  '✅',
    label:  'AUDIO NORMAL',
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

/* ── Timeline ─────────────────────────────────────────────────────────────
   Horizontal bar showing the full audio duration with anomaly windows
   marked in red. Each band is a contiguous event from the backend.
─────────────────────────────────────────────────────────────────────────── */
function Timeline({ duration, events }) {
  const dur = Math.max(0.1, duration || 0);
  return (
    <div style={{ marginTop: '1rem' }}>
      <div style={{
        display: 'flex', justifyContent: 'space-between',
        marginBottom: '0.3rem',
      }}>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          Audio Timeline ({dur.toFixed(1)}s)
        </span>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          {events.length} event{events.length === 1 ? '' : 's'}
        </span>
      </div>
      <div style={{
        position:     'relative',
        height:       '36px',
        borderRadius: '8px',
        background:   'rgba(255, 255, 255, 0.04)',
        border:       '1px solid rgba(255, 255, 255, 0.08)',
        overflow:     'hidden',
      }}>
        {events.map((ev, i) => {
          const left  = `${Math.max(0,   (ev.start_sec / dur) * 100)}%`;
          const width = `${Math.max(0.6, ((ev.end_sec - ev.start_sec) / dur) * 100)}%`;
          const intensity = Math.min(1, (ev.max_anomaly_prob ?? ev.max_confidence ?? 0.7));
          return (
            <div
              key={i}
              title={`${ev.start_sec.toFixed(1)}s – ${ev.end_sec.toFixed(1)}s • max ${(intensity * 100).toFixed(0)}%`}
              style={{
                position:     'absolute',
                top:          0, bottom: 0,
                left, width,
                background:   `rgba(220, 38, 38, ${0.45 + 0.45 * intensity})`,
                borderLeft:   '1px solid rgba(255, 255, 255, 0.4)',
                borderRight:  '1px solid rgba(255, 255, 255, 0.4)',
              }}
            />
          );
        })}
        {/* second markers every 5s */}
        {Array.from({ length: Math.floor(dur / 5) }, (_, i) => (i + 1) * 5).map((s) => (
          <div key={s} style={{
            position: 'absolute',
            left:     `${(s / dur) * 100}%`,
            top:      '50%',
            bottom:   0,
            width:    '1px',
            background: 'rgba(255, 255, 255, 0.18)',
          }} />
        ))}
      </div>
    </div>
  );
}

function EventList({ events }) {
  if (!events.length) return null;
  return (
    <div style={{ marginTop: '1rem' }}>
      <div className="section-label" style={{ marginBottom: '0.5rem' }}>
        <Clock size={13} strokeWidth={2} />
        Anomalous Audio Events
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
        {events.map((ev, i) => (
          <div
            key={i}
            style={{
              display:       'flex',
              alignItems:    'center',
              justifyContent:'space-between',
              padding:       '0.55rem 0.85rem',
              background:    'rgba(220, 38, 38, 0.08)',
              borderLeft:    '3px solid var(--unsafe)',
              borderRadius:  '6px',
              fontSize:      '0.82rem',
            }}
          >
            <div style={{ color: 'var(--text)' }}>
              <strong style={{ fontFamily: 'ui-monospace, monospace' }}>
                {ev.start_sec.toFixed(1)}s – {ev.end_sec.toFixed(1)}s
              </strong>
              <span style={{ color: 'var(--text-muted)', marginLeft: '0.6rem' }}>
                ({ev.duration_sec?.toFixed(1) ?? (ev.end_sec - ev.start_sec).toFixed(1)}s)
              </span>
            </div>
            <div style={{ color: 'var(--unsafe)', fontWeight: 600 }}>
              max {((ev.max_anomaly_prob ?? ev.max_confidence ?? 0) * 100).toFixed(0)}%
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function SoundPanel({ data }) {
  if (!data) return null;

  const status        = data.status            ?? 'SAFE';
  const ratio         = data.anomaly_ratio     ?? 0;
  const totalW        = data.total_windows     ?? 0;
  const anomW         = data.anomaly_windows   ?? 0;
  const duration      = data.duration_sec      ?? 0;
  const events        = data.events            ?? [];
  const message       = data.message           ?? null;
  const ratioPct      = (ratio * 100).toFixed(1);

  const meta     = STATUS_META[status] ?? STATUS_META.SAFE;
  const isUnsafe = status === 'UNSAFE';

  return (
    <div className="analysis-panel">
      <div className="section-label">
        <Volume2 size={13} strokeWidth={2} />
        Anomaly Sound Analysis
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
              ? `Anomalous sounds in ${ratioPct}% of analysed windows.`
              : `No anomalous sounds detected across ${totalW} windows.`}
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
          Investigate the equipment / scene around the flagged time window(s)
        </div>
      )}

      {/* ── Stat Cards ── */}
      <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
        <StatCard
          icon={Activity}
          label="Anomaly Ratio"
          value={`${ratioPct}%`}
          color={isUnsafe ? 'unsafe' : 'safe'}
        />
        <StatCard
          icon={Volume2}
          label="Anomaly Windows"
          value={anomW}
          color={anomW > 0 ? 'unsafe' : 'safe'}
        />
        <StatCard
          icon={CheckCircle2}
          label="Total Windows"
          value={totalW}
          color="neutral"
        />
      </div>

      {/* ── Anomaly ratio bar ── */}
      <div style={{ marginTop: '1rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Anomaly Ratio</span>
          <span style={{ fontSize: '0.75rem', color: `var(--${meta.color})`, fontWeight: 700 }}>
            {ratioPct}%
          </span>
        </div>
        <div className="metric-bar-track">
          <div
            className={`metric-bar-fill mc-bar-${meta.color}`}
            style={{ width: `${Math.min(parseFloat(ratioPct), 100)}%` }}
          />
        </div>
      </div>

      {/* ── Audio Timeline ── */}
      <Timeline duration={duration} events={events} />

      {/* ── Event List ── */}
      <EventList events={events} />
    </div>
  );
}
