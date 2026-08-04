import React from 'react'
import { ShieldCheck } from 'lucide-react'

interface HeroHeaderProps {
  healthScore: number | null
  failureProbability: number | null
  rulDays: number | null
  riskCategory: string | null
  dbConnected?: boolean
  featureCount?: number
}

function formatNumber(val: number | null | undefined, digits = 1) {
  if (val === null || val === undefined || Number.isNaN(val)) return '--'
  return val.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  })
}

export const HeroHeader: React.FC<HeroHeaderProps> = ({
  healthScore,
  failureProbability,
  rulDays,
  riskCategory,
  dbConnected = true,
  featureCount = 42,
}) => {
  return (
    <div className="enterprise-hero">
      <div className="hero-content-wrapper">
        <div className="hero-title-group">
          <h1>
            <ShieldCheck size={28} style={{ color: 'var(--primary)' }} />
            Bridge Structural Health & Inspection AI
          </h1>
          <p className="hero-subtitle">
            <span>● Status: Operational</span>
            <span>|</span>
            <span>Database: {dbConnected ? 'Connected' : 'Offline'}</span>
            <span>|</span>
            <span>Multi-Sensor Telemetry Analysis</span>
            <span>|</span>
            <span>Computer Vision Inspection Service</span>
          </p>
        </div>

        <div className="hero-metrics-pill-bar">
          <div className="hero-stat-block">
            <span className="hero-stat-label">Health Index (SHI)</span>
            <span className="hero-stat-value" style={{ color: 'var(--primary)' }}>
              {formatNumber(healthScore, 1)} / 100
            </span>
          </div>
          <div style={{ width: '1px', height: '32px', background: 'rgba(255, 255, 255, 0.15)' }} />
          <div className="hero-stat-block">
            <span className="hero-stat-label">Failure Prob (PoF)</span>
            <span className="hero-stat-value" style={{ color: 'var(--success)' }}>
              {formatNumber(failureProbability, 2)}%
            </span>
          </div>
          <div style={{ width: '1px', height: '32px', background: 'rgba(255, 255, 255, 0.15)' }} />
          <div className="hero-stat-block">
            <span className="hero-stat-label">Est. RUL</span>
            <span className="hero-stat-value">
              {formatNumber(rulDays, 0)} d
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default HeroHeader
