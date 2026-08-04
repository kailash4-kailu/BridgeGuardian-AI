import React from 'react'
import { ChevronLeft, ChevronRight, Search } from 'lucide-react'
import StatusBadge from '../ui/StatusBadge'

interface HistoryItem {
  id: number
  created_at: string
  health_score: number | null
  failure_probability: number | null
  rul_days: number | null
  risk_category: string | null
  maintenance_priority: string | null
  maintenance_recommendation?: string | null
  model_version: string | null
  analysis_type?: string | null
  campaign_id?: number | null
  image_count?: number | null
  status?: string | null
  summary_report?: string | null
}

interface AuditTableProps {
  history: HistoryItem[]
  filteredHistory: HistoryItem[]
  paginatedHistory: HistoryItem[]
  historyFilter: string
  onFilterChange: (filter: string) => void
  searchQuery: string
  onSearchChange: (query: string) => void
  currentPage: number
  totalPages: number
  onPageChange: (page: number) => void
}

function formatNumber(val: number | null | undefined, digits = 1) {
  if (val === null || val === undefined || Number.isNaN(val)) return '--'
  return val.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  })
}

function compactDate(value: string) {
  try {
    return new Intl.DateTimeFormat(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }).format(new Date(value))
  } catch {
    return value
  }
}

function formatWorkflowLabel(type: string | null | undefined) {
  if (!type) return 'Structural Health'
  const val = type.toLowerCase()
  if (val.includes('drone') || val.includes('campaign')) return 'Drone Campaign'
  if (val.includes('single') || val.includes('image') || val.includes('vision')) return 'Single Image'
  return 'Structural Health'
}

export const AuditTable: React.FC<AuditTableProps> = ({
  filteredHistory,
  paginatedHistory,
  historyFilter,
  onFilterChange,
  searchQuery,
  onSearchChange,
  currentPage,
  totalPages,
  onPageChange,
}) => {
  return (
    <section className="surface">
      <div className="panel-heading" style={{ flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <p className="eyebrow">Enterprise Audit Trail</p>
          <h2>Prediction & Inspection Log History</h2>
        </div>

        {/* Workflow Filter Chips */}
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          {[
            { id: 'all', label: 'All Workflows' },
            { id: 'drone_campaign', label: 'Drone Campaign' },
            { id: 'single_image', label: 'Single Image' },
            { id: 'structural_health', label: 'Structural Health' },
          ].map((chip) => (
            <button
              key={chip.id}
              type="button"
              className={`btn ${historyFilter === chip.id ? 'btn-primary' : 'btn-secondary'}`}
              style={{ padding: '6px 14px', fontSize: '0.8rem' }}
              onClick={() => onFilterChange(chip.id)}
            >
              {chip.label}
            </button>
          ))}
        </div>
      </div>

      {/* Audit Data Table */}
      <div style={{ overflowX: 'auto' }}>
        <div className="history-table">
          <div className="history-row header">
            <span>Timestamp</span>
            <span>Workflow Type</span>
            <span>Health (SHI)</span>
            <span>Failure Prob</span>
            <span>Status / Priority</span>
          </div>
          {paginatedHistory.length === 0 ? (
            <div style={{ padding: '48px', textAlign: 'center', color: 'var(--muted)' }}>
              No audit log records found matching your filters.
            </div>
          ) : (
            paginatedHistory.map((item) => (
              <div className="history-row" key={item.id}>
                <span>{compactDate(item.created_at)}</span>
                <div>
                  <span className="workflow-tag" title={item.model_version || ''}>
                    {formatWorkflowLabel(item.analysis_type)}
                  </span>
                </div>
                <strong style={{ fontSize: '0.95rem' }}>{formatNumber(item.health_score, 1)}</strong>
                <span style={{ fontSize: '0.9rem', color: 'var(--ink-subtle)' }}>
                  {formatNumber(item.failure_probability, 2)}%
                </span>
                <div>
                  <StatusBadge status={item.risk_category ?? item.maintenance_priority} />
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Pagination Bar */}
      <div style={{ marginTop: '20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
        <span style={{ fontSize: '0.85rem', color: 'var(--muted)' }}>
          Showing {paginatedHistory.length} of {filteredHistory.length} audit records
        </span>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <button
            type="button"
            className="btn btn-secondary"
            disabled={currentPage <= 1}
            onClick={() => onPageChange(Math.max(1, currentPage - 1))}
            style={{ padding: '6px 12px' }}
          >
            <ChevronLeft size={16} /> Prev
          </button>
          <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>
            Page {currentPage} of {totalPages}
          </span>
          <button
            type="button"
            className="btn btn-secondary"
            disabled={currentPage >= totalPages}
            onClick={() => onPageChange(Math.min(totalPages, currentPage + 1))}
            style={{ padding: '6px 12px' }}
          >
            Next <ChevronRight size={16} />
          </button>
        </div>
      </div>
    </section>
  )
}

export default AuditTable
