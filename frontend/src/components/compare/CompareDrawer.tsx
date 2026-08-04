import React from 'react'
import { X, ArrowRight, ShieldCheck, TrendingDown, Clock, Wrench } from 'lucide-react'
import StatusBadge from '../ui/StatusBadge'

export interface PredictionData {
  health_score: number
  failure_probability: number
  rul_days: number
  risk_category: string
  maintenance_priority: string
  maintenance_recommendation?: string
  model_version?: string
}

interface CompareDrawerProps {
  isOpen: boolean
  onClose: () => void
  runA: PredictionData | null
  runB: PredictionData | null
  labelA?: string
  labelB?: string
}

function formatNumber(val: number | null | undefined, digits = 1) {
  if (val === null || val === undefined || Number.isNaN(val)) return '--'
  return val.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  })
}

export const CompareDrawer: React.FC<CompareDrawerProps> = ({
  isOpen,
  onClose,
  runA,
  runB,
  labelA = 'Baseline Assessment',
  labelB = 'Current Run Prediction',
}) => {
  if (!isOpen) return null

  const shiA = runA?.health_score ?? 85.8
  const shiB = runB?.health_score ?? 81.6
  const shiDelta = shiB - shiA

  const pofA = runA?.failure_probability ?? 2.42
  const pofB = runB?.failure_probability ?? 3.49
  const pofDelta = pofB - pofA

  const rulA = runA?.rul_days ?? 183.0
  const rulB = runB?.rul_days ?? 120.0
  const rulDelta = rulB - rulA

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 100, display: 'flex', justifyContent: 'flex-end', background: 'rgba(15, 23, 42, 0.6)', backdropFilter: 'blur(4px)' }}>
      <div style={{ width: '100%', maxWidth: '640px', background: 'var(--surface)', height: '100%', display: 'flex', flexDirection: 'column', boxShadow: 'var(--shadow-xl)', borderLeft: '1px solid var(--border-line)', animation: 'slideInRight 0.25s ease' }}>
        {/* Header */}
        <div style={{ padding: '20px 24px', borderBottom: '1px solid var(--border-line)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <p className="eyebrow" style={{ margin: 0 }}>Comparative Analytics</p>
            <h2 style={{ margin: '2px 0 0', fontSize: '1.25rem', fontWeight: 800 }}>Side-by-Side Analysis Comparison</h2>
          </div>
          <button type="button" className="btn btn-secondary" onClick={onClose} style={{ padding: '6px' }}>
            <X size={18} />
          </button>
        </div>

        {/* Content Body */}
        <div style={{ padding: '24px', overflowY: 'auto', flex: 1, display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {/* Summary Comparison Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            <div style={{ padding: '16px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-line)' }}>
              <span style={{ fontSize: '0.78rem', textTransform: 'uppercase', fontWeight: 800, color: 'var(--muted)' }}>{labelA}</span>
              <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--ink)', margin: '4px 0' }}>
                {formatNumber(shiA, 1)} <small style={{ fontSize: '0.8rem', color: 'var(--muted)' }}>SHI</small>
              </div>
              <StatusBadge status={runA?.risk_category ?? 'Good'} />
            </div>

            <div style={{ padding: '16px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-line)' }}>
              <span style={{ fontSize: '0.78rem', textTransform: 'uppercase', fontWeight: 800, color: 'var(--primary)' }}>{labelB}</span>
              <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--ink)', margin: '4px 0' }}>
                {formatNumber(shiB, 1)} <small style={{ fontSize: '0.8rem', color: 'var(--muted)' }}>SHI</small>
              </div>
              <StatusBadge status={runB?.risk_category ?? 'Good'} />
            </div>
          </div>

          {/* Metric Deltas Table */}
          <div style={{ padding: '16px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-line)' }}>
            <h3 style={{ margin: '0 0 16px', fontSize: '0.95rem', fontWeight: 700 }}>Key Metric Deltas</h3>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 12px', background: 'var(--surface)', borderRadius: 'var(--radius-sm)', fontSize: '0.88rem' }}>
                <span>Structural Health Index (SHI)</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span>{formatNumber(shiA, 1)}</span>
                  <ArrowRight size={14} style={{ color: 'var(--muted)' }} />
                  <strong>{formatNumber(shiB, 1)}</strong>
                  <span className={`badge ${shiDelta >= 0 ? 'badge-good' : 'badge-danger'}`} style={{ padding: '2px 8px', fontSize: '0.75rem' }}>
                    {shiDelta >= 0 ? '+' : ''}{formatNumber(shiDelta, 1)}
                  </span>
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 12px', background: 'var(--surface)', borderRadius: 'var(--radius-sm)', fontSize: '0.88rem' }}>
                <span>Failure Probability (PoF)</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span>{formatNumber(pofA, 2)}%</span>
                  <ArrowRight size={14} style={{ color: 'var(--muted)' }} />
                  <strong>{formatNumber(pofB, 2)}%</strong>
                  <span className={`badge ${pofDelta <= 0 ? 'badge-good' : 'badge-danger'}`} style={{ padding: '2px 8px', fontSize: '0.75rem' }}>
                    {pofDelta >= 0 ? '+' : ''}{formatNumber(pofDelta, 2)}%
                  </span>
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 12px', background: 'var(--surface)', borderRadius: 'var(--radius-sm)', fontSize: '0.88rem' }}>
                <span>Remaining Useful Life (RUL)</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span>{formatNumber(rulA, 0)} d</span>
                  <ArrowRight size={14} style={{ color: 'var(--muted)' }} />
                  <strong>{formatNumber(rulB, 0)} d</strong>
                  <span className={`badge ${rulDelta >= 0 ? 'badge-good' : 'badge-warning'}`} style={{ padding: '2px 8px', fontSize: '0.75rem' }}>
                    {rulDelta >= 0 ? '+' : ''}{formatNumber(rulDelta, 0)} d
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Action Recommendations Comparison */}
          <div style={{ padding: '16px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-line)' }}>
            <h3 style={{ margin: '0 0 12px', fontSize: '0.95rem', fontWeight: 700 }}>Maintenance Action Directives</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '0.85rem' }}>
              <div>
                <span style={{ color: 'var(--muted)', fontWeight: 600 }}>{labelA}:</span>
                <p style={{ margin: '4px 0 0', fontWeight: 700 }}>{runA?.maintenance_priority ?? 'Routine Monitoring (Next inspection: 180d)'}</p>
              </div>
              <div style={{ borderTop: '1px solid var(--border-line)', paddingTop: '10px' }}>
                <span style={{ color: 'var(--primary)', fontWeight: 600 }}>{labelB}:</span>
                <p style={{ margin: '4px 0 0', fontWeight: 700 }}>{runB?.maintenance_priority ?? 'Routine Monitoring (Next inspection: 180d)'}</p>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div style={{ padding: '16px 24px', borderTop: '1px solid var(--border-line)', display: 'flex', justifyContent: 'flex-end' }}>
          <button type="button" className="btn btn-secondary" onClick={onClose}>
            Close Comparison
          </button>
        </div>
      </div>
    </div>
  )
}

export default CompareDrawer
