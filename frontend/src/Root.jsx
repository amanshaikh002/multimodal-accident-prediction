import { useState, useCallback } from 'react';

import App         from './App.jsx';
import LandingPage from './landing/LandingPage.jsx';

/**
 * Root — chooses between the public landing page and the existing
 * analysis dashboard via a single boolean.
 *
 * No router is added: the dashboard is essentially one page already, so a
 * simple state flip is all the navigation we need. The landing page is
 * rendered first; clicking any "Start Analysis" / "Launch Dashboard" CTA
 * flips to the dashboard.
 */
export default function Root() {
  const [showDashboard, setShowDashboard] = useState(false);

  const goToDashboard = useCallback(() => {
    setShowDashboard(true);
    window.scrollTo({ top: 0, behavior: 'instant' });
  }, []);

  if (!showDashboard) {
    return <LandingPage onLaunchDashboard={goToDashboard} />;
  }
  return <App />;
}
