import { useState } from 'react';
import { X, Play } from 'lucide-react';
import '../landing.css';

import Navbar      from './Navbar';
import Hero        from './Hero';
import Features    from './Features';
import HowItWorks  from './HowItWorks';
import LivePreview from './LivePreview';
import TechStack   from './TechStack';
import Footer      from './Footer';

/**
 * LandingPage — public entry point that sits in front of the dashboard.
 * onLaunchDashboard flips the parent state to render the existing
 * dashboard — no router needed.
 */
export default function LandingPage({ onLaunchDashboard }) {
  const [demoOpen, setDemoOpen] = useState(false);

  return (
    <div className="landing-root min-h-screen">
      <Navbar onLaunch={onLaunchDashboard} />
      <main>
        <Hero
          onStart={onLaunchDashboard}
          onDemo={() => setDemoOpen(true)}
        />
        <Features />
        <HowItWorks />
        <LivePreview />
        <TechStack />
      </main>
      <Footer />

      {demoOpen && (
        <DemoModal
          onClose={() => setDemoOpen(false)}
          onLaunch={onLaunchDashboard}
        />
      )}
    </div>
  );
}

function DemoModal({ onClose, onLaunch }) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-brand-dark/60 px-4 py-8 backdrop-blur-sm"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
    >
      <div
        className="relative w-full max-w-md overflow-hidden rounded-2xl border border-brand-border bg-white shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          type="button"
          onClick={onClose}
          aria-label="Close demo dialog"
          className="absolute right-3 top-3 rounded-md p-1.5 text-brand-muted hover:bg-brand-bg hover:text-brand-text"
        >
          <X size={18} strokeWidth={2.2} />
        </button>

        <div className="px-8 py-10 text-center">
          <div className="mx-auto mb-5 grid h-14 w-14 place-items-center rounded-full bg-brand-primary/10">
            <Play size={22} className="text-brand-primary" strokeWidth={2.4} />
          </div>
          <h3 className="text-xl font-bold text-brand-text">Try it on your own video</h3>
          <p className="mx-auto mt-3 max-w-md text-[14px] leading-relaxed text-brand-muted">
            Upload any MP4 of a workplace and see PPE, pose, fire, and sound results in seconds.
          </p>

          <div className="mt-7 flex flex-col gap-2.5">
            <button
              type="button"
              onClick={() => { onClose(); onLaunch(); }}
              className="inline-flex items-center justify-center rounded-lg bg-brand-text px-5 py-2.5 text-[14px] font-semibold text-white hover:bg-black"
            >
              Launch the dashboard
            </button>
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg border border-brand-border bg-white px-5 py-2.5 text-[14px] font-semibold text-brand-text hover:border-brand-text/30"
            >
              Maybe later
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
