import { useState, useRef, useCallback } from 'react';
import {
  HardHat, Upload, Play, Loader2, Volume2, VolumeX,
  TriangleAlert, CheckCircle2, Flame
} from 'lucide-react';
import './App.css';
import {
  enrichPpeViolations,
  enrichPoseViolations,
  enrichCombinedViolations,
  enrichFireViolations,
  enrichSoundViolations,
} from './suggestionMap';
import VideoPanel       from './components/VideoPanel';
import AnalysisPanel    from './components/AnalysisPanel';
import CombinedPanel    from './components/CombinedPanel';
import FirePanel        from './components/FirePanel';
import SoundPanel       from './components/SoundPanel';
import AllPanel         from './components/AllPanel';
import ViolationsList   from './components/ViolationsList';
import Recommendations  from './components/Recommendations';
import { useAudioAlerts } from './hooks/useAudioAlerts';

const API = 'http://localhost:8000';

/* ══════════════════════════════════════════════════════════════════════════════
   App
══════════════════════════════════════════════════════════════════════════════ */
export default function App() {
  const [mode,       setMode]       = useState('ppe');
  const [file,       setFile]       = useState(null);
  const [loading,    setLoading]    = useState(false);
  const [error,      setError]      = useState(null);
  const [result,     setResult]     = useState(null);
  const [videoUrl,   setVideoUrl]   = useState('');
  const [audioOn,    setAudioOn]    = useState(true);
  const fileRef = useRef();

  const { speak, speakViolations, reset: resetAudio, setEnabled: setAudioEnabled }
    = useAudioAlerts();

  /* ── File pick ─────────────────────────────────────────────────────────── */
  const handleFile = useCallback((e) => {
    const f = e.target.files?.[0];
    if (f) { setFile(f); setError(null); setResult(null); setVideoUrl(''); }
  }, []);

  const openPicker = () => fileRef.current?.click();

  /* ── Audio toggle ──────────────────────────────────────────────────────── */
  const toggleAudio = () => {
    const next = !audioOn;
    setAudioOn(next);
    setAudioEnabled(next);
  };

  /* ── Analyse ───────────────────────────────────────────────────────────── */
  const analyse = async () => {
    if (!file) { setError('Please select a video file first.'); return; }
    setLoading(true);
    setError(null);
    setResult(null);
    setVideoUrl('');
    resetAudio();

    try {
      const fd = new FormData();
      fd.append('video', file);
      const res  = await fetch(`${API}/detect?mode=${mode}`, { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? `Server error ${res.status}`);
      
      setResult({ mode, data });
      // Set the dynamic video URL returned/hosted by backend.
      // Backend services return `output_video`; legacy key `video_output` kept as fallback.
      const generatedFile = data.output_video || data.video_output ||
        (mode === 'ppe'      ? 'ppe_annotated.mp4'      :
         mode === 'combined' ? 'combined_annotated.mp4' :
         mode === 'fire'     ? 'fire_annotated.mp4'     :
         mode === 'sound'    ? 'sound_annotated.mp4'    :
         mode === 'all'      ? 'fire_annotated.mp4'     :
         'pose_annotated.mp4');
      // Cache-bust so the browser always fetches the newly written file
      setVideoUrl(`${API}/output/${generatedFile}?t=${Date.now()}`);

      // Trigger audio alerts for violations found
      const enriched =
        mode === 'ppe'      ? enrichPpeViolations(data.violations ?? []) :
        mode === 'combined' ? enrichCombinedViolations(data.violations ?? []) :
        mode === 'fire'     ? enrichFireViolations(data) :
        mode === 'sound'    ? enrichSoundViolations(data) :
        mode === 'all'      ? enrichCombinedViolations(data.violations ?? []) :
                              enrichPoseViolations(data.violations ?? []);
      speakViolations(enriched);

      // ── Module-specific urgent voice alerts ─────────────────────────────
      // speak() is called directly (bypassing speakViolations) so the alert
      // fires even when the violations list is empty.
      if (mode === 'fire' && data.status === 'UNSAFE') {
        speak('Warning! Fire detected in the workplace. Evacuate immediately and call emergency services.');
      }
      if (mode === 'sound' && data.status === 'UNSAFE') {
        speak('Warning! Anomalous machine sounds detected. Inspect the equipment immediately.');
      }
      if (mode === 'all' && data.fire_detected) {
        speak('Warning! Fire detected in the workplace. Evacuate immediately.');
      }
      if (mode === 'all' && data.sound_status === 'UNSAFE') {
        speak('Warning! Anomalous machine sounds detected.');
      }

    } catch (err) {
      setError(err.message ?? 'Unexpected error. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  /* ── Derived data ──────────────────────────────────────────────────────── */
  let violations  = [];
  let overallSafe = true;

  if (result) {
    const { mode: m, data } = result;
    violations =
      m === 'ppe'      ? enrichPpeViolations(data.violations ?? []) :
      m === 'combined' ? enrichCombinedViolations(data.violations ?? []) :
      m === 'fire'     ? enrichFireViolations(data) :
      m === 'sound'    ? enrichSoundViolations(data) :
      m === 'all'      ? enrichCombinedViolations(data.violations ?? []) :
                         enrichPoseViolations(data.violations ?? []);
    if (m === 'fire' || m === 'sound') {
      overallSafe = data.status === 'SAFE';
    } else if (m === 'all') {
      overallSafe = data.final_status === 'SAFE';
    } else {
      const score =
        m === 'ppe'      ? (data.compliance_score ?? 0) :
        m === 'combined' ? Math.min(data.ppe_score ?? 0, data.pose_score ?? 0) :
                           (data.safety_score ?? 0);
      overallSafe = score >= 70 && (m !== 'combined' || data.final_status === 'SAFE');
    }
  }

  /* ── Render ────────────────────────────────────────────────────────────── */
  return (
    <div className="app">

      {/* ━━━━ HEADER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      <header className="header">
        <div className="header-inner">
          <div className="header-brand">
            <HardHat size={28} strokeWidth={2} color="var(--accent)" className="brand-icon-large" />
            <div className="brand-text">
              <h1 className="brand-title">
                Multimodal Vision Audio Framework
              </h1>
              <div className="brand-sub">
                For Industrial Workplace Accident Prediction System
              </div>
            </div>
          </div>

          <div className="header-right">
            {/* Audio toggle */}
            <button
              id="audio-toggle"
              className={`icon-btn ${audioOn ? 'icon-btn-active' : ''}`}
              onClick={toggleAudio}
              title={audioOn ? 'Disable audio alerts' : 'Enable audio alerts'}
            >
              {audioOn
                ? <Volume2  size={16} strokeWidth={2} />
                : <VolumeX  size={16} strokeWidth={2} />}
              <span>{audioOn ? 'Alerts On' : 'Alerts Off'}</span>
            </button>
          </div>
        </div>
      </header>


      {/* ━━━━ MAIN ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      <main className="main">

        {/* ── Control Bar ─────────────────────────────────────────────────── */}
        <div className="control-bar">
          {/* Module selector */}
          <div className="ctrl-group">
            <label className="ctrl-label" htmlFor="module-select">Module</label>
            <select
              id="module-select"
              className="ctrl-select"
              value={mode}
              onChange={(e) => { setMode(e.target.value); setResult(null); setError(null); }}
            >
              <option value="ppe">🦺 PPE Compliance Detection</option>
              <option value="pose">🧍 Pose Safety Detection</option>
              <option value="fire">🔥 Fire Hazard Detection</option>
              <option value="sound">🔊 Anomaly Sound Detection</option>
              <option value="combined">🔗 PPE + Pose Detection</option>
              <option value="all">🚀 Full Platform (PPE + Pose + Fire + Sound)</option>
            </select>
          </div>

          {/* File picker */}
          <div className="ctrl-group ctrl-file-group">
            <label className="ctrl-label">Video File</label>
            <button
              id="file-picker-btn"
              className={`file-btn ${file ? 'file-btn-loaded' : ''}`}
              onClick={openPicker}
              type="button"
            >
              <Upload size={14} strokeWidth={2} />
              {file ? file.name : 'Select video (.mp4, .avi, .mov)'}
            </button>
            <input
              ref={fileRef}
              type="file"
              accept="video/*"
              className="sr-only"
              onChange={handleFile}
            />
          </div>

          {/* Analyse button */}
          <button
            id="analyse-btn"
            className="analyse-btn"
            onClick={analyse}
            disabled={loading || !file}
          >
            {loading
              ? <><Loader2 size={15} className="spin" /> Analysing…</>
              : <><Play    size={15} strokeWidth={2.5} /> Analyse</>}
          </button>
        </div>

        {/* ── Error Banner ────────────────────────────────────────────────── */}
        {error && (
          <div className="error-banner fade-in">
            <TriangleAlert size={16} strokeWidth={2} />
            {error}
          </div>
        )}

        {/* ── Loading State ────────────────────────────────────────────────── */}
        {loading && (
          <div className="loading-state fade-in">
            <Loader2 size={32} className="spin" color="var(--accent)" strokeWidth={1.5} />
            <p>Running {mode.toUpperCase()} analysis — this may take a moment…</p>
          </div>
        )}

        {/* ── Dashboard (2-column) ─────────────────────────────────────────── */}
        {!loading && (
          result ? (
            <div className="dashboard fade-in">

              {/* LEFT: Video Player */}
              <div className="col-left">
                <div className="section-label">
                  {overallSafe
                    ? <CheckCircle2 size={13} color="var(--safe)" strokeWidth={2} />
                    : <TriangleAlert size={13} color="var(--unsafe)" strokeWidth={2} />}
                  Annotated Output — {result.mode.toUpperCase()} Detection
                </div>
                <VideoPanel
                  mode={result.mode}
                  videoUrl={videoUrl}
                  overallSafe={overallSafe}
                  hasResult={true}
                />
              </div>

              {/* RIGHT: Analysis Panels */}
              <div className="col-right">
                {result.mode === 'combined'
                  ? <CombinedPanel data={result.data} />
                  : result.mode === 'fire'
                  ? <FirePanel data={result.data} />
                  : result.mode === 'sound'
                  ? <SoundPanel data={result.data} />
                  : result.mode === 'all'
                  ? <AllPanel data={result.data} />
                  : <AnalysisPanel mode={result.mode} data={result.data} />}
                <ViolationsList items={violations} />
                <Recommendations items={violations} />
              </div>

            </div>
          ) : (
            /* ── Welcome / Idle ────────────────────────────────────────── */
            <div className="idle-state fade-in">
              <div className="idle-cols">
                {/* Left: description */}
                <div className="idle-left">
                  <div className="idle-icon-wrap">
                    <HardHat size={36} strokeWidth={1.2} color="var(--accent)" />
                  </div>
                  <h2 className="idle-title">Industrial Safety Monitor</h2>
                  <p className="idle-desc">
                    Upload a workplace video and run AI-powered safety analysis.
                    Get violation details, ergonomic insights, and corrective recommendations.
                  </p>
                  <div className="idle-features">
                    {[
                      '🔥 Fire hazard detection (NEW)',
                      'Bounding-box overlays on video',
                      'Real-time violation timeline',
                      'Actionable safety recommendations',
                      'Audio alerts for unsafe events',
                    ].map((f, i) => (
                      <div key={i} className="idle-feature">
                        <CheckCircle2 size={14} color="var(--safe)" strokeWidth={2} />
                        {f}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Right: step guide */}
                <div className="idle-right">
                  <div className="section-label" style={{ marginBottom: '1.25rem' }}>How to use</div>
                  {[
                    ['01', 'Select Module', 'Choose PPE, Pose, Fire, or Full-Platform.'],
                    ['02', 'Upload Video',  'Select a workplace footage file.'],
                    ['03', 'Run Analysis',  'Click Analyse to start AI inference.'],
                    ['04', 'Review Report', 'See annotated video, violations & recommendations.'],
                  ].map(([n, t, d]) => (
                    <div key={n} className="step-card">
                      <span className="step-num">{n}</span>
                      <div>
                        <strong className="step-title">{t}</strong>
                        <span className="step-desc">{d}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Video placeholder */}
              <VideoPanel mode={mode} videoUrl="" overallSafe={true} hasResult={false} />
            </div>
          )
        )}
      </main>
    </div>
  );
}
