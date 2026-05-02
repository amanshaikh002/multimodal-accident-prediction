const STACK = [
  { name: 'YOLOv8',          dot: 'bg-blue-500'    },
  { name: 'YOLO-World',      dot: 'bg-blue-500'    },
  { name: 'YOLOv8s-pose',    dot: 'bg-blue-500'    },
  { name: 'OpenCV',          dot: 'bg-blue-500'    },
  { name: 'HuggingFace Hub', dot: 'bg-yellow-500'  },
  { name: 'RandomForest',    dot: 'bg-emerald-500' },
  { name: 'XGBoost',         dot: 'bg-emerald-500' },
  { name: 'librosa',         dot: 'bg-purple-500'  },
  { name: 'imageio-ffmpeg',  dot: 'bg-purple-500'  },
  { name: 'FastAPI',         dot: 'bg-teal-500'    },
  { name: 'Uvicorn',         dot: 'bg-teal-500'    },
  { name: 'React 19',        dot: 'bg-pink-500'    },
  { name: 'Vite',            dot: 'bg-pink-500'    },
  { name: 'Tailwind CSS',    dot: 'bg-pink-500'    },
];

export default function TechStack() {
  return (
    <section id="stack" className="border-t border-brand-border bg-brand-bg">
      <div className="mx-auto max-w-6xl px-6 py-24 sm:py-28">
        <div className="mx-auto max-w-2xl text-center">
          <div className="text-[12px] font-semibold uppercase tracking-[0.18em] text-brand-primary">
            Tech stack
          </div>
          <h2 className="mt-3 text-3xl font-bold tracking-[-0.02em] text-brand-text sm:text-4xl">
            Built on{' '}
            <span className="gradient-text">proven open-source</span>{' '}
            foundations
          </h2>
          <p className="mx-auto mt-4 max-w-md text-[15.5px] leading-relaxed text-brand-muted">
            Modern deep-learning models on a fast Python backend, with a snappy React dashboard.
          </p>
        </div>

        <div className="mx-auto mt-12 flex max-w-3xl flex-wrap items-center justify-center gap-2.5">
          {STACK.map(({ name, dot }) => (
            <span
              key={name}
              className="inline-flex items-center gap-2 rounded-full border border-brand-border bg-white px-3.5 py-1.5 text-[13px] font-medium text-brand-text shadow-sm transition-all hover:-translate-y-0.5 hover:shadow"
            >
              <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
              {name}
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}
