import { useRef, useCallback } from 'react';

/**
 * useAudioAlerts — wraps the Web Speech SpeechSynthesis API.
 *
 * Returns { speak, enabled, setEnabled }
 *
 * Rules:
 *  - Only speaks when enabled === true
 *  - Tracks spoken messages to avoid repetition within the same session
 *  - Falls back silently if SpeechSynthesis is unavailable
 */
export function useAudioAlerts() {
  const spokenRef = useRef(new Set());
  const enabledRef = useRef(true);

  const speak = useCallback((message) => {
    if (!enabledRef.current) return;
    if (!window.speechSynthesis) return;
    if (spokenRef.current.has(message)) return;

    spokenRef.current.add(message);

    const utterance = new SpeechSynthesisUtterance(message);
    utterance.rate   = 0.95;
    utterance.pitch  = 1;
    utterance.volume = 1;
    // Prefer a male voice for clarity in industrial environments
    const voices = window.speechSynthesis.getVoices();
    const preferred = voices.find(
      v => v.lang.startsWith('en') && v.name.toLowerCase().includes('male')
    ) ?? voices.find(v => v.lang.startsWith('en')) ?? null;
    if (preferred) utterance.voice = preferred;

    window.speechSynthesis.speak(utterance);
  }, []);

  /**
   * speakViolations — derive and speak alerts from an enriched violation list.
   * @param {Array} violations  — enriched violation objects
   */
  const speakViolations = useCallback((violations) => {
    if (!enabledRef.current || !violations?.length) return;

    // Combined-mode: if BOTH PPE and POSE types are present, fire a combined alert first
    const types = new Set(violations.map(v => (v.type || '').toUpperCase()));
    if (types.has('PPE') && types.has('POSE')) {
      speak('Warning! Worker is unsafe due to missing PPE and unsafe posture. Take immediate corrective action.');
      return; // One combined message is enough to avoid overloading
    }

    // Pose-related alerts are intentionally generic — the model can't tell
    // *what* the worker is doing (lifting, walking, reaching, etc.), only that
    // the body posture is outside the safe range. Avoid activity-specific
    // wording so we don't say "unsafe lifting" on a video where no lifting
    // is happening. We fire one pose alert per analysis at most.
    const POSE_REGEX = /back|posture|lifting|knee|stiff|leg|neck|bend/i;
    let posePlayed = false;

    // Build unique alert messages
    const seen = new Set();
    for (const v of violations) {
      const key = v.reason ?? v.item ?? 'Unknown issue';
      if (seen.has(key)) continue;
      seen.add(key);

      if (/helmet/i.test(key)) {
        speak('Warning! Worker is not wearing a safety helmet.');
      } else if (/vest/i.test(key)) {
        speak('Alert! Worker is missing a high-visibility vest.');
      } else if (/gloves/i.test(key)) {
        speak('Warning! Worker is not wearing protective gloves.');
      } else if (/boots/i.test(key)) {
        speak('Warning! Worker is not wearing safety boots.');
      } else if (POSE_REGEX.test(key)) {
        if (!posePlayed) {
          speak('Warning! Unsafe body posture detected. Review worker form.');
          posePlayed = true;
        }
      } else {
        speak(`Safety violation detected: ${key}`);
      }
    }
  }, [speak]);

  /** Reset spoken history (call before each new analysis) */
  const reset = useCallback(() => {
    spokenRef.current.clear();
  }, []);

  const setEnabled = useCallback((val) => {
    enabledRef.current = val;
  }, []);

  return { speak, speakViolations, reset, setEnabled };
}
