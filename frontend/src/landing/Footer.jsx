import { ShieldCheck, ExternalLink, Code2 } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="bg-brand-dark text-slate-300">
      <div className="mx-auto max-w-6xl px-6 py-14">
        <div className="grid grid-cols-1 gap-10 md:grid-cols-3">
          {/* Brand */}
          <div>
            <div className="flex items-center gap-2.5">
              <span className="grid h-8 w-8 place-items-center rounded-lg bg-brand-primary">
                <ShieldCheck size={16} className="text-white" strokeWidth={2.4} />
              </span>
              <span className="text-[15px] font-semibold text-white">
                Multimodal Safety AI
              </span>
            </div>
            <p className="mt-4 max-w-sm text-[13.5px] leading-relaxed text-slate-400">
              A unified pipeline for PPE compliance, pose ergonomics, accident
              detection, fire hazards, and acoustic anomalies — built on YOLOv8,
              FastAPI, and React.
            </p>
          </div>

          {/* Sections */}
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
              Sections
            </div>
            <ul className="mt-4 space-y-2 text-[13.5px]">
              <li><a href="#features" className="text-slate-300 hover:text-white">Features</a></li>
              <li><a href="#how"      className="text-slate-300 hover:text-white">How it works</a></li>
              <li><a href="#preview"  className="text-slate-300 hover:text-white">Live preview</a></li>
              <li><a href="#stack"    className="text-slate-300 hover:text-white">Tech stack</a></li>
            </ul>
          </div>

          {/* Resources */}
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
              Resources
            </div>
            <ul className="mt-4 space-y-2 text-[13.5px]">
              <li>
                <a
                  href="http://localhost:8000/docs"
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1.5 text-slate-300 hover:text-white"
                >
                  API documentation
                  <ExternalLink size={12} strokeWidth={2.2} />
                </a>
              </li>
              <li>
                <a
                  href="https://github.com/"
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1.5 text-slate-300 hover:text-white"
                >
                  <Code2 size={13} strokeWidth={2.2} />
                  Source code
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom strip */}
        <div className="mt-12 flex flex-col items-center justify-between gap-3 border-t border-white/10 pt-6 text-[12px] text-slate-500 sm:flex-row">
          <div>© {new Date().getFullYear()} Multimodal Vision Audio Framework. Built for safer workplaces.</div>
          <div className="flex items-center gap-3">
            <span>v3.2.0</span>
            <span className="opacity-50">·</span>
            <span>FastAPI · React · YOLOv8</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
