import { Upload, Cpu, ShieldAlert, Bell } from 'lucide-react';

const STEPS = [
  { n: '1', icon: Upload,      title: 'Upload Video',  desc: 'Drop in any MP4, MOV, AVI, or MKV.' },
  { n: '2', icon: Cpu,         title: 'Analyze',       desc: 'YOLO + pose tracker + audio classifier in one pass.' },
  { n: '3', icon: ShieldAlert, title: 'Detect Risks',  desc: 'PPE, accidents, fire, and sound surface with timestamps.' },
  { n: '4', icon: Bell,        title: 'Get Alerts',    desc: 'Voice + visual alerts, plus an annotated video and JSON.' },
];

export default function HowItWorks() {
  return (
    <section id="how" className="border-t border-brand-border bg-brand-bg">
      <div className="mx-auto max-w-6xl px-6 py-24 sm:py-28">
        {/* Header */}
        <div className="mx-auto max-w-2xl text-center">
          <div className="text-[12px] font-semibold uppercase tracking-[0.18em] text-brand-primary">
            How it works
          </div>
          <h2 className="mt-3 text-3xl font-bold tracking-[-0.02em] text-brand-text sm:text-4xl">
            From raw video to{' '}
            <span className="gradient-text">actionable alerts</span>
          </h2>
          <p className="mx-auto mt-4 max-w-md text-[15.5px] leading-relaxed text-brand-muted">
            One unified pipeline, four straightforward steps.
          </p>
        </div>

        {/* Steps */}
        <div className="mt-16 grid grid-cols-1 gap-10 sm:grid-cols-2 md:grid-cols-4 md:gap-8">
          {STEPS.map(({ n, icon: Icon, title, desc }) => (
            <div key={n} className="text-center">
              {/* Number + icon */}
              <div className="relative mx-auto flex h-14 w-14 items-center justify-center rounded-2xl border border-brand-border bg-white shadow-sm">
                <Icon size={22} className="text-brand-primary" strokeWidth={2.2} />
                <span className="absolute -right-2 -top-2 grid h-6 w-6 place-items-center rounded-full bg-brand-text text-[11px] font-bold text-white">
                  {n}
                </span>
              </div>

              <h3 className="mt-5 text-[15px] font-semibold text-brand-text">
                {title}
              </h3>
              <p className="mx-auto mt-2 max-w-[14rem] text-[13.5px] leading-relaxed text-brand-muted">
                {desc}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
