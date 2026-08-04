import React from 'react'
import type { LucideIcon } from 'lucide-react'

interface MetricCardProps {
  title: string
  value: string | number
  unit?: string
  trendLabel?: string
  trendTone?: 'good' | 'warning' | 'danger' | 'info' | 'neutral'
  icon: LucideIcon
  subtitle?: string
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  unit,
  trendLabel,
  trendTone = 'good',
  icon: Icon,
  subtitle,
}) => {
  const trendClasses: Record<string, string> = {
    good: 'trend-good',
    warning: 'trend-warning',
    danger: 'trend-danger',
    info: 'trend-info',
    neutral: 'trend-neutral',
  }

  return (
    <article className="stat-card">
      <div className="stat-header">
        <span className="stat-title">{title}</span>
        <div className="stat-icon-box">
          <Icon size={18} aria-hidden="true" />
        </div>
      </div>

      <div className="stat-main">
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '4px' }}>
          <span className="stat-value">{value}</span>
          {unit && <span className="stat-unit">{unit}</span>}
        </div>

        {trendLabel && (
          <span className={`stat-trend ${trendClasses[trendTone]}`}>
            {trendLabel}
          </span>
        )}
      </div>

      {subtitle && (
        <span className="stat-subtitle">{subtitle}</span>
      )}
    </article>
  )
}

export default MetricCard
