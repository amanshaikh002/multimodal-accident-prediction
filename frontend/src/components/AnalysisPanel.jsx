import { ShieldCheck, AlertTriangle, Activity } from 'lucide-react';
import { riskLevel } from '../suggestionMap';

function MetricCard({ icon: Icon, label, value, color, bar }) {
  return (
    <div className="metric-card">
      <div className="metric-card-header">
        <Icon size={15} color="var(--text-muted)" strokeWidth={2} />
        <span className="metric-label">{label}</span>
      </div>
      <div className={`metric-value mc-${color}`}>{value}</div>
      {bar !== undefined && (
        <div className="metric-bar-track">
          <div
            className={`metric-bar-fill mc-bar-${color}`}
            style={{ width: `${Math.min(Math.max(bar, 0), 100)}%` }}
          />
        </div>
      )}
    </div>
  );
}

export default function AnalysisPanel({ mode, data }) {
  if (!data) return null;

  const score  = mode === 'ppe' ? (data.compliance_score ?? 0) : (data.safety_score ?? 0);
  const unsafe = data.unsafe_frames ?? '—';
  const risk   = riskLevel(score);

  const scoreColor = score >= 80 ? 'safe' : score >= 50 ? 'warn' : 'unsafe';
  const riskColor  = risk === 'LOW' ? 'safe' : risk === 'MEDIUM' ? 'warn' : 'unsafe';

  const RISK_LABELS = { LOW: 'Low Risk', MEDIUM: 'Medium Risk', HIGH: 'High Risk' };

  return (
    <div className="analysis-panel">
      <div className="section-label">Risk Summary</div>
      <div className="metrics-grid">
        <MetricCard
          icon={ShieldCheck}
          label={mode === 'ppe' ? 'Compliance Score' : 'Safety Score'}
          value={`${typeof score === 'number' ? score.toFixed(1) : '—'}%`}
          color={scoreColor}
          bar={score}
        />
        <MetricCard
          icon={AlertTriangle}
          label="Unsafe Events"
          value={unsafe}
          color={unsafe === 0 || unsafe === '—' ? 'neutral' : 'unsafe'}
        />
        <MetricCard
          icon={Activity}
          label="Risk Level"
          value={RISK_LABELS[risk]}
          color={riskColor}
        />
      </div>
    </div>
  );
}
