import React, { useEffect, useState } from 'react';
import './SplashScreen.css';

interface SplashScreenProps {
  onFinished?: () => void;
  minDurationMs?: number;
}

export const SplashScreen: React.FC<SplashScreenProps> = ({
  onFinished,
  minDurationMs = 1800,
}) => {
  const [progress, setProgress] = useState<number>(10);
  const [statusText, setStatusText] = useState<string>('Initializing BridgeGuardian AI...');
  const [isFadingOut, setIsFadingOut] = useState<boolean>(false);
  const [isHidden, setIsHidden] = useState<boolean>(false);

  useEffect(() => {
    const t1 = setTimeout(() => {
      setProgress(45);
      setStatusText('Connecting to Structural Health Sensors...');
    }, 400);

    const t2 = setTimeout(() => {
      setProgress(75);
      setStatusText('Loading AI Prediction Models...');
    }, 900);

    const t3 = setTimeout(() => {
      setProgress(100);
      setStatusText('System Ready');
    }, 1400);

    const t4 = setTimeout(() => {
      setIsFadingOut(true);
    }, minDurationMs);

    const t5 = setTimeout(() => {
      setIsHidden(true);
      if (onFinished) {
        onFinished();
      }
    }, minDurationMs + 600);

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
      clearTimeout(t4);
      clearTimeout(t5);
    };
  }, [minDurationMs, onFinished]);

  if (isHidden) return null;

  return (
    <div className={`splash-overlay ${isFadingOut ? 'fade-out' : ''}`}>
      {/* Background Tech Mesh Overlay */}
      <div className="splash-bg-grid" />
      <div className="splash-bg-glow" />

      {/* Main Centered Content */}
      <div className="splash-content">
        <div className="splash-logo-container">
          <img
            src="/logo-full.svg"
            alt="BridgeGuardian AI"
            className="splash-logo-image"
          />
        </div>

        {/* Loading Progress Section */}
        <div className="splash-loader-wrapper">
          <div className="splash-progress-track">
            <div
              className="splash-progress-bar"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="splash-status-text">{statusText}</p>
        </div>
      </div>
    </div>
  );
};

export default SplashScreen;
