import React, { useState } from 'react'
import {
  FileText,
  AlertTriangle,
  Layers,
  ArrowRight,
  BarChart3,
  Eye,
  Sparkles,
  HelpCircle,
} from 'lucide-react'
import type {
  TabType,
  PredictionResponse,
  InspectionRecord,
} from '../../types'
import PdfDownloadButton from '../ui/PdfDownloadButton'

interface InspectionIntelligenceProps {
  activeTab: TabType
  droneRecord?: InspectionRecord | null
  telemetryPrediction?: PredictionResponse | null
  visionPrediction?: any | null
  isAnalyzing?: boolean
  onDownloadPdf?: () => void
  onViewPdf?: () => void
  onStartInspection?: () => void
}

export const InspectionIntelligence: React.FC<InspectionIntelligenceProps> = ({
  activeTab,
  droneRecord,
  telemetryPrediction,
  visionPrediction,
  isAnalyzing = false,
  onDownloadPdf,
  onViewPdf,
  onStartInspection,
}) => {
  const [showReportModal, setShowReportModal] = useState(false)

  const isCompleted = droneRecord?.status === 'completed' || Boolean(telemetryPrediction) || Boolean(visionPrediction)
  const isFailed = droneRecord?.status === 'failed' || (droneRecord?.performance_metrics && droneRecord.performance_metrics.accepted_images === 0)
  const hasActiveAnalysis = isCompleted && !isFailed

  const workflowTitle =
    activeTab === 'drone'
      ? 'Drone Flight Campaign'
      : activeTab === 'console'
      ? 'Telemetry Sensor Prediction'
      : activeTab === 'vision'
      ? 'Single Image Defect Vision'
      : 'Structural Inspection'

  const riskCategory = hasActiveAnalysis
    ? (droneRecord?.risk_category || telemetryPrediction?.risk_category || visionPrediction?.predictions?.risk_category || 'Low Risk')
    : (isFailed ? 'Failed' : 'Standby')

  const priorityLevel = hasActiveAnalysis
    ? (droneRecord?.maintenance_priority || telemetryPrediction?.maintenance_priority || 'Routine Monitoring')
    : (isFailed ? 'Inspection Required' : '--')

  const recommendedAction = hasActiveAnalysis
    ? (droneRecord?.maintenance_action || telemetryPrediction?.maintenance_recommendation || 'Continue routine structural monitoring.')
    : (isFailed ? 'Inspection could not be completed because no uploaded images passed validation. Bridge condition remains unknown.' : 'Run inspection to generate repair guidelines.')

  const acceptedCount = droneRecord?.performance_metrics?.accepted_images ?? (hasActiveAnalysis ? 1 : 0)
  const totalCount = (droneRecord?.performance_metrics?.accepted_images ?? 0) + (droneRecord?.performance_metrics?.rejected_images ?? 0)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', marginTop: '24px' }}>
      {/* Panel Heading */}
      <div className="panel-heading" style={{ margin: 0 }}>
        <div>
          <p className="eyebrow">Operational Inspection & Intelligence</p>
          <h2 style={{ fontSize: '1.35rem', fontWeight: 800, margin: 0 }}>
            Bridge Asset Condition & Actionable Intelligence
          </h2>
        </div>
        <span
          className={`badge ${isFailed ? 'badge-critical' : hasActiveAnalysis ? 'badge-good' : 'badge-neutral'}`}
          style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}
        >
          <Sparkles size={14} /> {isFailed ? 'Inspection Failed' : hasActiveAnalysis ? 'Active Analysis Loaded' : 'Awaiting Inspection'}
        </span>
      </div>

      {/* FAILED STATE BANNER */}
      {isFailed && (
        <div
          className="surface"
          style={{
            padding: '24px',
            background: 'rgba(254, 242, 242, 0.8)',
            border: '1px solid var(--danger)',
            borderRadius: 'var(--radius-lg)',
            color: 'var(--danger)'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
            <AlertTriangle size={24} />
            <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 800 }}>Inspection Failed</h3>
          </div>
          <p style={{ margin: 0, fontSize: '0.9rem', lineHeight: 1.5, color: '#991B1B' }}>
            Inspection could not be completed because no uploaded images passed quality validation.
            Bridge condition remains unknown. Predictions intentionally skipped to avoid misleading engineering decisions.
          </p>
        </div>
      )}

      {/* EMPTY STATE BANNER */}
      {!hasActiveAnalysis && !isAnalyzing && !isFailed && (
        <div
          className="surface"
          style={{
            padding: '32px 24px',
            textAlign: 'center',
            background: 'var(--surface-glass)',
            border: '1px dashed var(--border-strong)',
            borderRadius: 'var(--radius-lg)',
          }}
        >
          <HelpCircle size={36} style={{ color: 'var(--primary)', marginBottom: '12px' }} />
          <h3 style={{ margin: '0 0 8px 0', fontSize: '1.2rem', fontWeight: 800 }}>No Analysis Available</h3>
          <p style={{ margin: '0 auto 20px auto', maxWidth: '520px', color: 'var(--muted)', fontSize: '0.9rem', lineHeight: 1.5 }}>
            Upload bridge inspection images or run a structural health prediction to begin AI analysis.
          </p>
          {onStartInspection && (
            <button type="button" className="btn btn-primary" onClick={onStartInspection} style={{ padding: '10px 20px' }}>
              <ArrowRight size={16} /> Start Inspection
            </button>
          )}
        </div>
      )}

      {/* 4 CORE OPERATIONAL CARDS GRID */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
          gap: '20px',
        }}
      >
        {/* CARD 1: Inspection Overview */}
        <div className="surface" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '14px', minHeight: '220px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Layers size={18} style={{ color: 'var(--primary)' }} />
            <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 700 }}>Inspection Overview</h3>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '0.86rem', flex: 1 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-line)', paddingBottom: '8px' }}>
              <span style={{ color: 'var(--muted)' }}>Workflow Type</span>
              <strong style={{ color: 'var(--ink)' }}>{workflowTitle}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-line)', paddingBottom: '8px' }}>
              <span style={{ color: 'var(--muted)' }}>Analysis Status</span>
              <span className={`badge ${isAnalyzing ? 'badge-primary' : isFailed ? 'badge-critical' : hasActiveAnalysis ? 'badge-good' : 'badge-neutral'}`}>
                {isAnalyzing ? 'In Progress' : isFailed ? 'Failed' : hasActiveAnalysis ? 'Completed' : 'Pending'}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-line)', paddingBottom: '8px' }}>
              <span style={{ color: 'var(--muted)' }}>Accepted Images</span>
              <strong>{hasActiveAnalysis ? `${acceptedCount} of ${totalCount}` : isFailed ? '0 Accepted' : '--'}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--muted)' }}>Processing Duration</span>
              <strong>{hasActiveAnalysis && droneRecord?.performance_metrics?.total_processing_time_sec ? `${droneRecord.performance_metrics.total_processing_time_sec.toFixed(1)}s` : '--'}</strong>
            </div>
          </div>
        </div>

        {/* CARD 2: Health & Risk Assessment */}
        <div className="surface" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '14px', minHeight: '220px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <BarChart3 size={18} style={{ color: 'var(--primary)' }} />
            <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 700 }}>Health & Risk Assessment</h3>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '0.86rem', flex: 1 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-line)', paddingBottom: '8px' }}>
              <span style={{ color: 'var(--muted)' }}>Overall Bridge Condition</span>
              <strong style={{ color: isFailed ? 'var(--danger)' : 'var(--ink)' }}>
                {isFailed ? 'Analysis Failed' : hasActiveAnalysis ? riskCategory : 'Awaiting Inspection'}
              </strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-line)', paddingBottom: '8px' }}>
              <span style={{ color: 'var(--muted)' }}>Risk Category</span>
              <span className={`badge ${isFailed ? 'badge-critical' : hasActiveAnalysis ? 'badge-good' : 'badge-neutral'}`}>
                {riskCategory}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-line)', paddingBottom: '8px' }}>
              <span style={{ color: 'var(--muted)' }}>Maintenance Priority</span>
              <strong>{priorityLevel}</strong>
            </div>
          </div>
        </div>

        {/* CARD 3: Inspection Recommendations */}
        <div className="surface" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '14px', minHeight: '220px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <AlertTriangle size={18} style={{ color: 'var(--primary)' }} />
            <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 700 }}>Inspection Recommendations</h3>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '0.85rem', flex: 1 }}>
            <div style={{ padding: '10px 12px', background: isFailed ? 'rgba(254,242,242,0.8)' : 'var(--surface-alt)', borderRadius: 'var(--radius-sm)', borderLeft: `3px solid ${isFailed ? 'var(--danger)' : 'var(--primary)'}`, lineHeight: 1.5 }}>
              <strong style={{ display: 'block', marginBottom: '2px', color: 'var(--ink)' }}>Recommended Action:</strong>
              {recommendedAction}
            </div>
          </div>
        </div>

        {/* CARD 4: Reports */}
        <div className="surface" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '14px', minHeight: '220px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <FileText size={18} style={{ color: 'var(--primary)' }} />
            <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 700 }}>Reports</h3>
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--muted)', margin: 0, lineHeight: 1.5 }}>
            Access compiled PDF inspection documentation and executive summaries for structural compliance.
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginTop: 'auto' }}>
            <button
              type="button"
              className="btn btn-secondary"
              disabled={!hasActiveAnalysis && !isFailed}
              onClick={() => (onViewPdf ? onViewPdf() : setShowReportModal(true))}
              style={{ fontSize: '0.85rem', padding: '10px 14px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
            >
              <Eye size={16} /> View Report
            </button>

            <PdfDownloadButton
              disabled={!hasActiveAnalysis && !isFailed}
              onDownload={async () => {
                if (onDownloadPdf) {
                  await onDownloadPdf()
                }
              }}
            />
          </div>
        </div>
      </div>

      {/* REPORT MODAL IF OPENED */}
      {showReportModal && droneRecord && (
        <div className="modal-overlay" onClick={() => setShowReportModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '680px', padding: '24px' }}>
            <h3>Inspection Assessment Report SUMMARY</h3>
            <p style={{ fontSize: '0.9rem', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
              {droneRecord.summary_report || 'No executive summary available.'}
            </p>
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '20px' }}>
              <button className="btn btn-secondary" onClick={() => setShowReportModal(false)}>Close</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default InspectionIntelligence
