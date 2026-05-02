import { CheckCircle2, ShieldCheck, Flame, Activity, Volume2, HardHat } from 'lucide-react';

export default function LivePreview() {
  return (
    <section id="preview" className="border-t border-brand-border bg-white">
      <div className="mx-auto max-w-6xl px-6 py-24 sm:py-28">
        {/* Header */}
        <div className="mx-auto max-w-2xl text-center">
          <div className="text-[12px] font-semibold uppercase tracking-[0.18em] text-brand-secondary">
            Dashboard preview
          </div>
          <h2 className="mt-3 text-3xl font-bold tracking-[-0.02em] text-brand-text sm:text-4xl">
            Real-time AI-powered{' '}
            <span className="gradient-text">safety monitoring</span>
          </h2>
          <p className="mx-auto mt-4 max-w-md text-[15.5px] leading-relaxed text-brand-muted">
            Annotated video on the left, multimodal report on the right — from a single upload.
          </p>
        </div>

        {/* Mock dashboard */}
        <div className="mx-auto mt-14 max-w-5xl">
          <div className="overflow-hidden rounded-2xl border border-brand-border bg-white shadow-[0_30px_60px_-25px_rgba(17,24,39,0.18)]">
            <BrowserChrome />
            <DashboardBody />
          </div>
          <p className="mt-5 text-center text-[13px] text-brand-muted">
            Real-time AI-powered safety monitoring dashboard.
          </p>
        </div>
      </div>
    </section>
  );
}

/* ── Browser chrome ────────────────────────────────────────────────── */

function BrowserChrome() {
  return (
    <div className="flex items-center justify-between border-b border-brand-border bg-brand-bg/60 px-4 py-2.5">
      <div className="flex items-center gap-1.5">
        <span className="h-2.5 w-2.5 rounded-full bg-red-400/80" />
        <span className="h-2.5 w-2.5 rounded-full bg-amber-400/80" />
        <span className="h-2.5 w-2.5 rounded-full bg-emerald-400/80" />
      </div>
      <div className="hidden items-center gap-2 rounded-md border border-brand-border bg-white px-2.5 py-1 sm:flex">
        <span className="live-dot" />
        <span className="font-mono text-[10.5px] text-brand-muted">multimodal safety dashboard</span>
      </div>
      <div className="w-12" />
    </div>
  );
}

/* ── Body ──────────────────────────────────────────────────────────── */

function DashboardBody() {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-[1.5fr_1fr]">
      {/* LEFT — video panel */}
      <div className="relative bg-gradient-to-br from-slate-50 to-slate-100 lg:border-r lg:border-brand-border">
        <div className="relative aspect-[16/10] lg:aspect-auto lg:h-full">
          {/* Top status banner */}
          <div className="absolute left-0 right-0 top-0 z-10 flex items-center gap-2 bg-emerald-500 px-4 py-2 text-white">
            <CheckCircle2 size={13} strokeWidth={2.6} />
            <span className="text-[11.5px] font-semibold tracking-wide">
              ALL SAFE — 4 WORKERS COMPLIANT
            </span>
          </div>

          {/* Mock skeletons */}
          <div className="absolute inset-0 flex items-end justify-center pb-12">
            <div className="flex items-end gap-10 sm:gap-14">
              <Skeleton scale={0.95} />
              <Skeleton scale={1.05} />
              <Skeleton scale={1.00} />
              <Skeleton scale={0.90} />
            </div>
          </div>
        </div>
      </div>

      {/* RIGHT — analysis panel */}
      <div className="space-y-4 bg-white p-6">
        <div className="flex items-center gap-3 rounded-xl border border-emerald-200 bg-emerald-50 p-3.5">
          <span className="grid h-9 w-9 flex-shrink-0 place-items-center rounded-full bg-brand-secondary text-white">
            <ShieldCheck size={16} strokeWidth={2.6} />
          </span>
          <div>
            <div className="text-[13px] font-bold tracking-wide text-emerald-700">ALL SAFE</div>
            <div className="text-[11.5px] text-emerald-700/70">No PPE, posture, or fire violations.</div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <Stat icon={HardHat}  label="PPE"   value="95.4%" iconBg="bg-blue-50"    iconFg="text-brand-primary"   />
          <Stat icon={Activity} label="Pose"  value="95.2%" iconBg="bg-emerald-50" iconFg="text-brand-secondary" />
          <Stat icon={Flame}    label="Fire"  value="0.0%"  iconBg="bg-orange-50"  iconFg="text-orange-500"      />
          <Stat icon={Volume2}  label="Sound" value="0"     iconBg="bg-purple-50"  iconFg="text-purple-500"      />
        </div>
      </div>
    </div>
  );
}

function Stat({ icon: Icon, label, value, iconBg, iconFg }) {
  return (
    <div className="rounded-lg border border-brand-border bg-white p-3">
      <div className="mb-1.5 flex items-center justify-between">
        <span className="text-[10.5px] font-semibold uppercase tracking-[0.14em] text-brand-muted">
          {label}
        </span>
        <span className={`grid h-5 w-5 place-items-center rounded ${iconBg} ${iconFg}`}>
          <Icon size={11} strokeWidth={2.4} />
        </span>
      </div>
      <div className="text-[20px] font-bold leading-none text-brand-text">{value}</div>
      <div className="mt-1 text-[10px] font-bold tracking-[0.14em] text-brand-secondary">SAFE</div>
    </div>
  );
}

function Skeleton({ scale = 1 }) {
  const stroke = '#3B82F6';
  const helmet = '#10B981';
  return (
    <svg
      viewBox="0 0 60 100"
      style={{ height: `${9 * scale}rem`, width: `${3.6 * scale}rem` }}
    >
      <rect x="22" y="2"  width="16" height="9" rx="3" fill={helmet} opacity="0.95" />
      <circle cx="30" cy="14" r="6" fill={stroke} opacity="0.9" />
      <line x1="30" y1="22" x2="30" y2="56" stroke={stroke} strokeWidth="3" strokeLinecap="round" />
      <line x1="14" y1="29" x2="46" y2="29" stroke={stroke} strokeWidth="3" strokeLinecap="round" />
      <line x1="14" y1="29" x2="10" y2="51" stroke={stroke} strokeWidth="3" strokeLinecap="round" />
      <line x1="46" y1="29" x2="50" y2="51" stroke={stroke} strokeWidth="3" strokeLinecap="round" />
      <line x1="20" y1="56" x2="40" y2="56" stroke={stroke} strokeWidth="3" strokeLinecap="round" />
      <line x1="20" y1="56" x2="18" y2="92" stroke={stroke} strokeWidth="3" strokeLinecap="round" />
      <line x1="40" y1="56" x2="42" y2="92" stroke={stroke} strokeWidth="3" strokeLinecap="round" />
      {[
        [30, 21], [14, 29], [46, 29], [10, 51], [50, 51],
        [20, 56], [40, 56], [18, 92], [42, 92],
      ].map(([cx, cy], i) => (
        <circle key={i} cx={cx} cy={cy} r="2.4" fill="white" stroke={stroke} strokeWidth="1.6" />
      ))}
    </svg>
  );
}
