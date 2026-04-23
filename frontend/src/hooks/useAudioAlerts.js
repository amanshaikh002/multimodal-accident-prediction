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
      } else if (/back|posture|lifting/i.test(key)) {
        speak('Alert! Unsafe lifting posture detected.');
      } else if (/knee/i.test(key)) {
        speak('Warning! Unsafe knee bend detected.');
      } else if (/neck/i.test(key)) {
        speak('Warning! Unsafe neck posture detected.');
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
