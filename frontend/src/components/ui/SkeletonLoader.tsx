import React from 'react'
import { RefreshCw } from 'lucide-react'

interface SkeletonLoaderProps {
  title?: string
  subtitle?: string
  progress?: number
  stage?: string
}

export const SkeletonLoader: React.FC<SkeletonLoaderProps> = ({
  title = 'Processing Request...',
  subtitle = 'Running backend ML models and computer vision pipelines',
  progress,
  stage,
}) => {
  return (
    <div className="surface loader-container" style={{ textAlign: 'center', padding: '48px 24px' }}>
      <RefreshCw size={44} className="spinning" style={{ color: 'var(--primary)', marginBottom: '16px' }} />
      <h3 style={{ fontSize: '1.25rem', fontWeight: 800, margin: '0 0 6px', color: 'var(--ink)' }}>
        {title}
      </h3>
      <p style={{ color: 'var(--muted)', fontSize: '0.9rem', margin: '0 0 20px', maxWidth: '480px', marginLeft: 'auto', marginRight: 'auto' }}>
        {subtitle}
      </p>

      {stage && (
        <span className="badge badge-info" style={{ marginBottom: '16px' }}>
          Current Stage: {stage}
        </span>
      )}

      {typeof progress === 'number' && (
        <div style={{ width: '100%', maxWidth: '460px', margin: '0 auto' }}>
          <div style={{ width: '100%', height: '8px', background: 'var(--surface-alt)', borderRadius: '99px', overflow: 'hidden', border: '1px solid var(--border-line)' }}>
            <div
              style={{
                width: `${Math.min(100, Math.max(0, progress))}%`,
                height: '100%',
                background: 'linear-gradient(90deg, var(--primary) 0%, var(--accent) 100%)',
                transition: 'width 0.3s ease',
              }}
            />
          </div>
          <div style={{ marginTop: '8px', display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--muted)', fontWeight: 600 }}>
            <span>Progress</span>
            <span>{Math.round(progress)}%</span>
          </div>
        </div>
      )}
    </div>
  )
}

export default SkeletonLoader
