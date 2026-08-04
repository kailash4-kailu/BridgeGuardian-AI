import React from 'react'
import { CheckCircle2, AlertTriangle, AlertCircle, Info, RefreshCw } from 'lucide-react'

export type StatusTone = 'good' | 'warning' | 'danger' | 'info' | 'neutral'

interface StatusBadgeProps {
  status: string | null | undefined
  tone?: StatusTone
  showIcon?: boolean
}

export function getToneForRisk(risk: string | null | undefined): StatusTone {
  if (!risk) return 'neutral'
  const val = risk.toLowerCase()
  if (val.includes('critical') || val.includes('poor') || val.includes('failed') || val.includes('offline')) {
    return 'danger'
  }
  if (val.includes('fair') || val.includes('medium') || val.includes('warning') || val.includes('degraded') || val.includes('stressed')) {
    return 'warning'
  }
  if (val.includes('good') || val.includes('excellent') || val.includes('healthy') || val.includes('online') || val.includes('completed')) {
    return 'good'
  }
  return 'info'
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  tone,
  showIcon = true,
}) => {
  const label = status || 'Not Available'
  const resolvedTone = tone || getToneForRisk(status)

  const toneClasses: Record<StatusTone, string> = {
    good: 'badge-good',
    warning: 'badge-warning',
    danger: 'badge-danger',
    info: 'badge-info',
    neutral: 'badge-neutral',
  }

  const renderIcon = () => {
    if (!showIcon) return null
    switch (resolvedTone) {
      case 'good':
        return <CheckCircle2 size={13} aria-hidden="true" />
      case 'warning':
        return <AlertTriangle size={13} aria-hidden="true" />
      case 'danger':
        return <AlertCircle size={13} aria-hidden="true" />
      case 'info':
        return <Info size={13} aria-hidden="true" />
      default:
        return null
    }
  }

  return (
    <span className={`badge ${toneClasses[resolvedTone]}`}>
      {renderIcon()}
      <span>{label}</span>
    </span>
  )
}

export default StatusBadge
