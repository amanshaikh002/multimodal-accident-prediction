import { ArrowRight, PlayCircle } from 'lucide-react';

export default function Hero({ onStart, onDemo }) {
  return (
    <section id="top" className="relative">
      <div className="mx-auto max-w-6xl px-6 pt-24 pb-20 text-center sm:pt-32 sm:pb-24">
        {/* Eyebrow badge */}
        <div className="animate-fade-up inline-flex items-center gap-2 rounded-full border border-brand-border bg-white px-3 py-1 shadow-sm">
          <span className="h-1.5 w-1.5 rounded-full bg-brand-secondary" />
          <span className="text-[12px] font-semibold tracking-wide text-brand-muted">
            Vision + Audio + Pose Unified AI
          </span>
        </div>

        {/* Headline */}
        <h1
          className="animate-fade-up mx-auto mt-8 max-w-3xl text-balance text-4xl font-bold leading-[1.08] tracking-[-0.025em] text-brand-text sm:text-5xl md:text-[3.25rem]"
          style={{ animationDelay: '80ms' }}
        >
          AI-Powered Workplace
          <br className="hidden sm:block" />{' '}
          <span className="gradient-text">Safety Monitoring</span>
        </h1>

        {/* Subtitle */}
        <p
          className="animate-fade-up mx-auto mt-6 max-w-xl text-pretty text-[17px] leading-relaxed text-brand-muted"
          style={{ animationDelay: '160ms' }}
        >
          Catch unsafe behaviour, accidents, fires, and equipment anomalies the
          moment they happen — from a single video upload.
        </p>

        {/* CTAs */}
        <div
          className="animate-fade-up mt-10 flex flex-col items-center justify-center gap-3 sm:flex-row"
          style={{ animationDelay: '240ms' }}
        >
          <button
            type="button"
            onClick={onStart}
            className="group inline-flex items-center justify-center gap-2 rounded-lg bg-brand-text px-6 py-3 text-[14.5px] font-semibold text-white shadow-sm transition-all hover:bg-black hover:shadow-md"
          >
            Start Analysis
            <ArrowRight
              size={15}
              strokeWidth={2.6}
              className="transition-transform group-hover:translate-x-0.5"
            />
          </button>

          <button
            type="button"
            onClick={onDemo}
            className="group inline-flex items-center justify-center gap-2 rounded-lg border border-brand-border bg-white px-6 py-3 text-[14.5px] font-semibold text-brand-text shadow-sm transition-all hover:border-brand-text/30 hover:shadow"
          >
            <PlayCircle size={16} strokeWidth={2.2} className="text-brand-muted group-hover:text-brand-text" />
            View Demo
          </button>
        </div>
      </div>
    </section>
  );
}
