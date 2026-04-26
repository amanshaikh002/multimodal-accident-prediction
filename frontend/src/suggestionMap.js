/**
 * suggestionMap.js
 * ================
 * Maps backend violation strings → human-readable reasons + actionable suggestions.
 * Works for both PPE (missing items) and Pose (ergonomic issues).
 */

// ── PPE Suggestion Map ────────────────────────────────────────────────────────
const PPE_MAP = {
  helmet: {
    reason:     "Worker is not wearing a safety helmet",
    suggestion: "Ensure the worker wears a certified hard hat at all times on site.",
    severity:   "high",
  },
  vest: {
    reason:     "Worker is not wearing a high-visibility safety vest",
    suggestion: "Wear a high-visibility safety vest to remain visible to machinery operators.",
    severity:   "high",
  },
  gloves: {
    reason:     "Worker is not wearing protective gloves",
    suggestion: "Use appropriate gloves rated for the task to prevent hand injuries.",
    severity:   "medium",
  },
  boots: {
    reason:     "Worker is not wearing safety boots",
    suggestion: "Steel-toe boots must be worn to protect against falling objects.",
    severity:   "medium",
  },
  mask: {
    reason:     "Worker is not wearing a respiratory mask",
    suggestion: "Wear an approved respirator when working in dusty or chemical environments.",
    severity:   "medium",
  },
  goggles: {
    reason:     "Worker is not wearing eye protection",
    suggestion: "Safety goggles must be worn when grinding, cutting, or handling chemicals.",
    severity:   "medium",
  },
};

// ── Pose / Ergonomic Suggestion Map ──────────────────────────────────────────
const POSE_MAP = [
  {
    match:      /excessive back bending/i,
    reason:     "Worker is bending their back excessively while lifting",
    suggestion: "Keep the back straight; bend at the knees and hips instead of the waist.",
    severity:   "high",
  },
  {
    match:      /moderate back lean/i,
    reason:     "Worker shows a moderate forward lean of the back",
    suggestion: "Straighten the torso slightly and engage core muscles to reduce spinal load.",
    severity:   "medium",
  },
  {
    match:      /stiff legs/i,
    reason:     "Worker is lifting with locked or stiff knees",
    suggestion: "Bend at the knees and hips instead of bending the back. Keep legs engaged.",
    severity:   "high",
  },
  {
    match:      /moderate knee stiffness/i,
    reason:     "Moderate knee stiffness detected during the task",
    suggestion: "Bend knees slightly more to distribute weight evenly and protect the lower back.",
    severity:   "medium",
  },
  {
    match:      /significant neck tilt/i,
    reason:     "Worker is significantly tilting their head forward",
    suggestion: "Align head with spine; raise work surface or use a stand to reduce neck strain.",
    severity:   "high",
  },
  {
    match:      /slight neck forward/i,
    reason:     "Slight forward head posture detected",
    suggestion: "Take regular breaks and perform neck stretches every 30 minutes.",
    severity:   "low",
  },
  {
    match:      /bad lifting posture/i,
    reason:     "Improper lifting posture detected",
    suggestion: "Lift with legs, not the back. Keep the load close to the body.",
    severity:   "high",
  },
];

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Map a list of PPE violation objects (from backend) to enriched entries.
 * @param {Array} violations  — e.g. [{frame, missing: ["helmet","vest"]}]
 * @returns {Array}           — [{frame, item, reason, suggestion, severity}]
 */
export function enrichPpeViolations(violations = []) {
  const enriched = [];
  for (const v of violations) {
    const frame = v.frame ?? "?";
    for (const item of v.missing ?? []) {
      const key = item.toLowerCase().trim();
      const info = PPE_MAP[key] ?? {
        reason:     `Missing ${item}`,
        suggestion: `Ensure ${item} is worn as per site safety protocol.`,
        severity:   "medium",
      };
      enriched.push({ frame, item, ...info });
    }
  }
  return enriched;
}

/**
 * Map a list of Pose violation objects (from backend) to enriched entries.
 * @param {Array} violations  — e.g. [{frame, issue: "Excessive back bending"}]
 * @returns {Array}           — [{frame, item, reason, suggestion, severity}]
 */
export function enrichPoseViolations(violations = []) {
  return violations.map((v) => {
    const issue = v.reason || v.issue || "Bad posture"; // Support both v.reason and v.issue
    const frame = v.frame ?? "?";
    const matched = POSE_MAP.find((m) => m.match.test(issue));
    if (matched) {
      const { match: _m, ...info } = matched;
      return { frame, item: issue, ...info };
    }
    return {
      frame,
      item:       issue,
      reason:     issue,
      suggestion: "Consult an ergonomics specialist to review the workstation setup.",
      severity:   "medium",
    };
  });
}

/**
 * Map combined-mode violations (which already carry type + reason) to enriched entries.
 * PPE violations go through PPE_MAP; Pose violations go through POSE_MAP.
 * @param {Array} violations  — [{frame, type, reason, severity}]
 * @returns {Array}           — [{frame, type, reason, suggestion, severity}]
 */
export function enrichCombinedViolations(violations = []) {
  return violations.map((v) => {
    const type   = (v.type   || 'PPE').toUpperCase();
    const reason = v.reason  || '';
    const frame  = v.frame   ?? '?';

    if (type === 'PPE') {
      // Try to match item name inside reason string
      const itemMatch = Object.keys(PPE_MAP).find((k) =>
        reason.toLowerCase().includes(k)
      );
      const info = itemMatch ? PPE_MAP[itemMatch] : {
        reason:     reason || 'Missing PPE item',
        suggestion: 'Ensure all required PPE is worn per site safety protocol.',
        severity:   v.severity ?? 'high',
      };
      return { frame, type, item: reason, ...info, reason: info.reason || reason };
    }

    // POSE
    const matched = POSE_MAP.find((m) => m.match.test(reason));
    if (matched) {
      const { match: _m, ...info } = matched;
      return { frame, type, item: reason, ...info };
    }
    return {
      frame,
      type,
      item:       reason,
      reason:     reason || 'Unsafe posture detected',
      suggestion: 'Consult an ergonomics specialist to review the workstation setup.',
      severity:   v.severity ?? 'high',
    };
  });
}

/**
 * Derive an overall risk level from a safety/compliance score.
 * @param {number} score  0–100
 * @returns {"LOW"|"MEDIUM"|"HIGH"}
 */
export function riskLevel(score) {
  if (score >= 80) return 'LOW';
  if (score >= 50) return 'MEDIUM';
  return 'HIGH';
}
