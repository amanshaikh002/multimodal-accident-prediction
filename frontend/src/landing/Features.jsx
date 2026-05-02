import { HardHat, Activity, Flame, Volume2 } from 'lucide-react';

const FEATURES = [
  {
    icon: HardHat,
    title: 'PPE Compliance',
    desc:  'Per-worker helmet and vest verification with sticky smoothing for bent or non-frontal poses.',
    iconBg: 'bg-blue-50',
    iconFg: 'text-brand-primary',
  },
  {
    icon: Activity,
    title: 'Pose & Accident',
    desc:  '17-point skeletal tracking. Detects falls, impacts, motionless workers, and ergonomic risk.',
    iconBg: 'bg-emerald-50',
    iconFg: 'text-brand-secondary',
  },
  {
    icon: Flame,
    title: 'Fire Hazard',
    desc:  'Dedicated fire/smoke YOLO model with aspect-preserving inference and temporal persistence.',
    iconBg: 'bg-orange-50',
    iconFg: 'text-orange-500',
  },
  {
    icon: Volume2,
    title: 'Sound Anomaly',
    desc:  'Sliding-window MFCC classifier listens for grinding, hissing, and abnormal machine sounds.',
    iconBg: 'bg-purple-50',
    iconFg: 'text-purple-500',
  },
];

export default function Features() {
  return (
    <section id="features" className="border-t border-brand-border bg-white">
      <div className="mx-auto max-w-6xl px-6 py-24 sm:py-28">
        {/* Header */}
        <div className="mx-auto max-w-2xl text-center">
          <div className="text-[12px] font-semibold uppercase tracking-[0.18em] text-brand-secondary">
            Detection modules
          </div>
          <h2 className="mt-3 text-3xl font-bold tracking-[-0.02em] text-brand-text sm:text-4xl">
            Five safety signals,{' '}
            <span className="gradient-text">one pipeline</span>
          </h2>
          <p className="mx-auto mt-4 max-w-md text-[15.5px] leading-relaxed text-brand-muted">
            Each module runs independently and combines into a single unified verdict.
          </p>
        </div>

        {/* Grid */}
        <div className="mt-16 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {FEATURES.map(({ icon: Icon, title, desc, iconBg, iconFg }) => (
            <article
              key={title}
              className="rounded-xl border border-brand-border bg-white p-6 shadow-[0_1px_2px_rgba(17,24,39,0.04)] transition-all duration-200 hover:-translate-y-1 hover:border-brand-text/15 hover:shadow-[0_18px_36px_-18px_rgba(17,24,39,0.18)]"
            >
              <div className={`mb-5 inline-flex h-10 w-10 items-center justify-center rounded-lg ${iconBg} ${iconFg}`}>
                <Icon size={18} strokeWidth={2.2} />
              </div>
              <h3 className="text-[15.5px] font-semibold tracking-[-0.005em] text-brand-text">
                {title}
              </h3>
              <p className="mt-2 text-[13.5px] leading-relaxed text-brand-muted">
                {desc}
              </p>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}
