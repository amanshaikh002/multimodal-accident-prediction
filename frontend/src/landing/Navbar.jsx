import { ShieldCheck } from 'lucide-react';

export default function Navbar({ onLaunch }) {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-brand-border bg-white/80 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-6">
        {/* Brand */}
        <a href="#top" className="flex items-center gap-2.5">
          <span className="grid h-8 w-8 place-items-center rounded-lg bg-brand-dark">
            <ShieldCheck size={16} className="text-white" strokeWidth={2.4} />
          </span>
          <span className="text-[15px] font-semibold text-brand-text">
            Multimodal Safety AI
          </span>
        </a>

        {/* Inline nav */}
        <nav className="hidden items-center gap-8 md:flex">
          <a href="#features" className="text-[14px] font-medium text-brand-muted transition-colors hover:text-brand-text">Features</a>
          <a href="#how"      className="text-[14px] font-medium text-brand-muted transition-colors hover:text-brand-text">How it works</a>
          <a href="#preview"  className="text-[14px] font-medium text-brand-muted transition-colors hover:text-brand-text">Preview</a>
          <a href="#stack"    className="text-[14px] font-medium text-brand-muted transition-colors hover:text-brand-text">Tech</a>
        </nav>

        {/* CTA */}
        <button
          type="button"
          onClick={onLaunch}
          className="inline-flex items-center justify-center rounded-lg bg-brand-text px-4 py-2 text-[13.5px] font-medium text-white shadow-sm transition-all hover:bg-black hover:shadow"
        >
          Launch Dashboard
        </button>
      </div>
    </header>
  );
}
