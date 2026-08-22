import React, { useState, useEffect, useRef, useMemo } from 'react'
import {
  Upload,
  Layers,
  AlertTriangle,
  RefreshCw,
  Trash2,
  TrendingDown,
  Clock,
  Wrench,
  ShieldCheck,
  Sparkles,
} from 'lucide-react'

import { API_BASE, getStaticUrl } from '../lib/api'
import { compressImageBatch } from '../lib/imageUtils'
import PdfDownloadButton from './ui/PdfDownloadButton'
import { validateStateConsistency } from '../lib/stateValidator'

type UploadedFile = {
  filename: string
  filepath: string
}

type DefectDetail = {
  defect_id: string
  type: string
  severity: string
  confidence: number
  bbox: [number, number, number, number]
  measurements: {
    width_mm: number
    length_mm: number
    area_pct: number
  }
  images: string[]
  occurrences: number
  component?: string
}

type ImageResult = {
  image_name: string
  is_valid: boolean
  warnings: string[]
  rejection_reason?: string
  metrics: {
    blur_score?: number
    brightness?: number
    width?: number
    height?: number
  }
  features?: any
  visualizations?: {
    original: string
    bboxes: string
    segmentation: string
    heatmap: string
  }
}

type InspectionRecord = {
  id: number
  created_at: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  progress: number
  health_score: number | null
  failure_probability: number | null
  rul_days: number | null
  risk_category: string | null
  maintenance_priority: string | null
  maintenance_action: string | null
  repair_window_days: number | null
  inspection_interval_days: number | null
  summary_report: string | null
  aggregate_results: any | null
  image_results: ImageResult[] | null
  performance_metrics: {
    total_processing_time_sec: number
    images_per_second: number
    accepted_images: number
    rejected_images: number
    avg_image_quality: number
    device: string
    memory_usage_mb: number
  } | null
}

export default function DroneInspection({ onCampaignComplete }: { onCampaignComplete?: (record?: any) => void }) {
  // Upload and queue state
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const [uploadedList, setUploadedList] = useState<UploadedFile[]>([])
  const [inspectionId, setInspectionId] = useState<number | null>(null)
  const [record, setRecord] = useState<InspectionRecord | null>(null)

  // Visualizer overlay controls
  const [selectedImageIndex, setSelectedImageIndex] = useState<number>(0)
  const [activeOverlay, setActiveOverlay] = useState<'original' | 'bboxes' | 'segmentation' | 'heatmap'>('segmentation')

  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const consecutiveErrorsRef = useRef<number>(0)

  const validImageResults = useMemo(() => {
    return record?.image_results?.filter((img) => img.is_valid) || []
  }, [record])

  const rejectedImageResults = useMemo(() => {
    return record?.image_results?.filter((img) => !img.is_valid) || []
  }, [record])

  const activeImage = validImageResults[selectedImageIndex] || validImageResults[0] || null

  // Keyboard navigation for image gallery
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!validImageResults || validImageResults.length <= 1) return
      if (e.key === 'ArrowLeft') {
        setSelectedImageIndex((prev) => (prev > 0 ? prev - 1 : validImageResults.length - 1))
      } else if (e.key === 'ArrowRight') {
        setSelectedImageIndex((prev) => (prev < validImageResults.length - 1 ? prev + 1 : 0))
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [validImageResults])

  // Auto-poll if inspection is running
  useEffect(() => {
    if (inspectionId === null) return
    consecutiveErrorsRef.current = 0

    const interval = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE}/inspection/${inspectionId}`)
        if (!response.ok) {
          consecutiveErrorsRef.current += 1
          if (consecutiveErrorsRef.current >= 4) {
            throw new Error(`Failed to fetch inspection status (HTTP ${response.status})`)
          }
          return
        }

        consecutiveErrorsRef.current = 0
        setErrorMsg(null)

        const data = (await response.json()) as InspectionRecord
        setRecord(data)

        if (data.status === 'completed' || data.status === 'failed') {
          clearInterval(interval)
          if (data.status === 'completed') {
            const acceptedCount = data.performance_metrics?.accepted_images ?? 0
            if (acceptedCount > 0) {
              // Validate state consistency before triggering completion
              validateStateConsistency(
                {
                  healthScore: data.health_score,
                  failureProbability: data.failure_probability,
                  remainingUsefulLife: data.rul_days,
                  maintenancePriority: data.maintenance_priority,
                  acceptedImages: acceptedCount,
                  totalImages: acceptedCount + (data.performance_metrics?.rejected_images ?? 0)
                },
                {
                  healthScore: data.health_score,
                  failureProbability: data.failure_probability,
                  remainingUsefulLife: data.rul_days,
                  maintenancePriority: data.maintenance_priority,
                  acceptedImages: acceptedCount,
                  totalImages: acceptedCount + (data.performance_metrics?.rejected_images ?? 0)
                }
              )
            }
            onCampaignComplete?.(data)
          } else if (data.status === 'failed') {
            setErrorMsg(data.summary_report || 'Campaign inspection failed on backend server.')
          }
        }
      } catch (err: any) {
        setErrorMsg(err.message)
      }
    }, 2000)

    return () => clearInterval(interval)
  }, [inspectionId, onCampaignComplete])

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    if (e.dataTransfer.files) {
      const filesArr = Array.from(e.dataTransfer.files).filter((file) =>
        ['image/jpeg', 'image/png', 'image/webp'].includes(file.type)
      )
      setSelectedFiles((prev) => {
        const combined = [...prev, ...filesArr]
        if (combined.length > 20) {
          setErrorMsg(`Maximum 20 photos allowed per campaign. First 20 photos selected.`)
          return combined.slice(0, 20)
        }
        setErrorMsg(null)
        return combined
      })
    }
  }

  const handleFileSelection = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const filesArr = Array.from(e.target.files)
      setSelectedFiles((prev) => {
        const combined = [...prev, ...filesArr]
        if (combined.length > 20) {
          setErrorMsg(`Maximum 20 photos allowed per campaign. First 20 photos selected.`)
          return combined.slice(0, 20)
        }
        setErrorMsg(null)
        return combined
      })
    }
  }

  const removeSelectedFile = (idx: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== idx))
  }

  const clearSelection = () => {
    // Flush all cached state immediately
    setSelectedFiles([])
    setUploadedList([])
    setInspectionId(null)
    setRecord(null)
    setSelectedImageIndex(0)
    setErrorMsg(null)
    onCampaignComplete?.(null)
  }

  const runInspectionCampaign = async () => {
    if (selectedFiles.length < 5 || selectedFiles.length > 20) {
      setErrorMsg(`Campaign requires between 5 and 20 photos. Currently selected: ${selectedFiles.length}.`)
      return
    }

    // Flush previous inspection cache completely
    setRecord(null)
    setInspectionId(null)
    setSelectedImageIndex(0)
    onCampaignComplete?.(null)

    setIsUploading(true)
    setErrorMsg(null)

    try {
      const optimizedFiles = await compressImageBatch(selectedFiles, 1920, 1920, 0.85)
      const formData = new FormData()
      optimizedFiles.forEach((file) => {
        formData.append('files', file)
      })

      const uploadRes = await fetch(`${API_BASE}/inspection/upload-images`, {
        method: 'POST',
        body: formData,
      })
      if (!uploadRes.ok) {
        let detail = 'Upload failed'
        try {
          const errDetail = await uploadRes.json()
          detail = errDetail.detail || errDetail.message || 'Upload failed'
        } catch {
          detail = `Upload failed with HTTP status ${uploadRes.status}`
        }
        throw new Error(detail)
      }

      const filesData = (await uploadRes.json()) as UploadedFile[]
      setUploadedList(filesData)

      const paths = filesData.map((f) => f.filepath)
      const inspectRes = await fetch(`${API_BASE}/inspection/run-inspection`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_paths: paths, pixel_to_mm: 0.5 }),
      })

      if (!inspectRes.ok) {
        throw new Error('Failed to initiate inspection campaign')
      }

      const inspectData = await inspectRes.json()
      setInspectionId(inspectData.record_id || inspectData.id)
    } catch (err: any) {
      setErrorMsg(err.message)
    } finally {
      setIsUploading(false)
    }
  }

  const downloadReportPdf = async () => {
    if (!inspectionId) return
    window.open(`${API_BASE}/inspection/report/${inspectionId}`, '_blank')
  }

  const isFailedCampaign = record && (record.status === 'failed' || (record.status === 'completed' && validImageResults.length === 0))

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {errorMsg && (
        <div style={{ padding: '14px 18px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid var(--danger)', borderRadius: 'var(--radius-md)', color: 'var(--danger)', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <AlertTriangle size={20} />
          <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>{errorMsg}</span>
        </div>
      )}

      {/* STEP 1: UPLOAD STAGE */}
      {record === null && (
        <div className="surface" style={{ padding: '28px' }}>
          <div className="panel-heading" style={{ marginBottom: '16px' }}>
            <div>
              <p className="eyebrow">Drone Campaign Ingestion</p>
              <h2 style={{ fontSize: '1.25rem', fontWeight: 800, margin: 0 }}>
                Upload Multi-Angle Bridge Inspection Flight Batch
              </h2>
            </div>
            {selectedFiles.length > 0 && (
              <button type="button" className="btn btn-secondary" onClick={clearSelection} style={{ fontSize: '0.8rem' }}>
                Clear Batch
              </button>
            )}
          </div>

          <input
            type="file"
            ref={fileInputRef}
            multiple
            accept="image/jpeg,image/png,image/webp"
            style={{ display: 'none' }}
            onChange={handleFileSelection}
          />

          <div
            className="dropzone-container"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="dropzone-icon-ring">
              <Upload size={32} />
            </div>
            <h3 style={{ margin: '0 0 8px', fontSize: '1.2rem', fontWeight: 800 }}>
              Drag & Drop 5 to 20 Drone Inspection Photos
            </h3>
            <p style={{ margin: '0 0 12px', color: 'var(--muted)', fontSize: '0.9rem' }}>
              Upload multi-angle drone flight photos for AI defect mapping & RUL calculation
            </p>
            <span className="badge badge-minor">Supported Formats: JPEG, PNG, WEBP (Max 20 photos)</span>
          </div>

          {selectedFiles.length > 0 && (
            <div style={{ marginTop: '28px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px', flexWrap: 'wrap', gap: '12px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 700 }}>
                    Queued Images ({selectedFiles.length})
                  </h3>
                  <span className="badge badge-minor">{selectedFiles.length >= 5 ? 'Ready to Run' : 'Need at least 5 photos'}</span>
                </div>

                <button
                  type="button"
                  className="btn btn-primary btn-pulse"
                  style={{ padding: '12px 28px', fontSize: '0.95rem' }}
                  onClick={runInspectionCampaign}
                  disabled={isUploading || selectedFiles.length < 5}
                >
                  {isUploading ? (
                    <>
                      <RefreshCw size={18} className="spinning" />
                      Uploading & Initializing...
                    </>
                  ) : (
                    <>
                      <Sparkles size={18} />
                      Run Campaign Analysis ({selectedFiles.length} Photos)
                    </>
                  )}
                </button>
              </div>

              {/* Thumbnails Grid */}
              <div className="image-preview-grid">
                {selectedFiles.map((file, idx) => (
                  <div key={idx} className="preview-thumbnail">
                    <img src={URL.createObjectURL(file)} alt={`Drone capture ${idx + 1}`} />
                    <button
                      type="button"
                      className="remove-img-btn"
                      onClick={(e) => {
                        e.stopPropagation()
                        removeSelectedFile(idx)
                      }}
                      title="Remove image"
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* STEP 2: PROCESSING STAGE */}
      {inspectionId !== null && record && record.status !== 'completed' && record.status !== 'failed' && (
        <div className="surface" style={{ textAlign: 'center', padding: '50px 24px' }}>
          <RefreshCw size={52} className="spinning" style={{ color: 'var(--primary)', marginBottom: '20px' }} />
          <h2 style={{ fontSize: '1.5rem', fontWeight: 800, margin: '0 0 8px' }}>
            Analyzing Campaign #{inspectionId}
          </h2>
          <p style={{ color: 'var(--muted)', fontSize: '0.95rem', margin: '0 0 24px' }}>
            Computer vision engines running multi-image morphological defect analysis...
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '8px', maxWidth: '640px', margin: '0 auto 20px' }}>
            {[
              { label: 'Queued', step: 0.15 },
              { label: 'Preprocessing', step: 0.35 },
              { label: 'Vision AI', step: 0.65 },
              { label: 'Report Gen', step: 0.90 },
              { label: 'Completed', step: 1.0 },
            ].map((stage) => {
              const isPast = record.progress >= stage.step
              return (
                <div key={stage.label} style={{ textAlign: 'center' }}>
                  <div style={{ height: '6px', borderRadius: '4px', background: isPast ? 'var(--primary)' : 'var(--surface-alt)', transition: 'background 0.3s' }} />
                  <span style={{ fontSize: '0.72rem', fontWeight: 700, color: isPast ? 'var(--primary)' : 'var(--muted)', marginTop: '4px', display: 'block' }}>
                    {stage.label}
                  </span>
                </div>
              )
            })}
          </div>

          <div style={{ width: '100%', maxWidth: '520px', height: '10px', background: 'var(--surface-alt)', borderRadius: '99px', overflow: 'hidden', margin: '0 auto 12px', border: '1px solid var(--border-line)' }}>
            <div
              style={{
                width: `${record.progress * 100}%`,
                height: '100%',
                background: 'linear-gradient(90deg, var(--primary) 0%, var(--accent) 100%)',
                transition: 'width 0.4s ease',
              }}
            />
          </div>
          <span style={{ fontWeight: 800, color: 'var(--primary)', fontSize: '0.9rem' }}>
            {Math.round(record.progress * 100)}% Complete
          </span>
        </div>
      )}

      {/* STEP 3: COMPLETED / FAILED CAMPAIGN DASHBOARD */}
      {record && (record.status === 'completed' || record.status === 'failed') && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>

          {/* New Campaign Action Bar */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <button type="button" className="btn btn-secondary" onClick={clearSelection}>
              ← Start New Campaign Inspection
            </button>
            <span className={`badge ${isFailedCampaign ? 'badge-critical' : 'badge-good'}`}>
              {isFailedCampaign ? 'Inspection Failed' : 'Inspection Completed'}
            </span>
          </div>

          {/* Top KPI Cards Grid */}
          <div className="stats-card-grid">
            <div className="stat-card">
              <div className="stat-card-accent-line" />
              <div className="stat-header">
                <span className="stat-title">Health Score (SHI)</span>
                <div className="stat-icon-box"><ShieldCheck size={18} /></div>
              </div>
              <div className="stat-main">
                <span className="stat-value">{isFailedCampaign ? '—' : `${record.health_score}%`}</span>
                <span className={`stat-trend ${isFailedCampaign ? 'negative' : 'positive'}`}>
                  {isFailedCampaign ? 'Status: Unavailable' : (record.risk_category || 'Good')}
                </span>
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-card-accent-line" style={{ background: 'linear-gradient(90deg, #22C55E 0%, #10B981 100%)' }} />
              <div className="stat-header">
                <span className="stat-title">Failure Probability</span>
                <div className="stat-icon-box" style={{ color: '#22C55E', background: 'rgba(34, 197, 94, 0.12)' }}><TrendingDown size={18} /></div>
              </div>
              <div className="stat-main">
                <span className="stat-value">{isFailedCampaign ? 'N/A' : `${record.failure_probability}%`}</span>
                <span className={`stat-trend ${isFailedCampaign ? 'warning' : 'positive'}`}>
                  {isFailedCampaign ? 'Uncalibrated' : 'Calibrated'}
                </span>
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-card-accent-line" style={{ background: 'linear-gradient(90deg, #F59E0B 0%, #D97706 100%)' }} />
              <div className="stat-header">
                <span className="stat-title">Estimated RUL</span>
                <div className="stat-icon-box" style={{ color: '#F59E0B', background: 'rgba(245, 158, 11, 0.12)' }}><Clock size={18} /></div>
              </div>
              <div className="stat-main">
                <span className="stat-value" style={{ fontSize: typeof record.rul_days === 'number' && record.rul_days >= 3650 ? '1.05rem' : '1.35rem' }}>
                  {isFailedCampaign ? 'Unknown' : (typeof record.rul_days === 'number' && record.rul_days >= 3650 ? 'Baseline (10+ Yrs)' : `${record.rul_days} d`)}
                </span>
                <span className="stat-trend warning">{isFailedCampaign ? 'N/A' : 'Target Life'}</span>
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-card-accent-line" style={{ background: 'linear-gradient(90deg, #EF4444 0%, #DC2626 100%)' }} />
              <div className="stat-header">
                <span className="stat-title">Remediation Action</span>
                <div className="stat-icon-box" style={{ color: '#EF4444', background: 'rgba(239, 68, 68, 0.12)' }}><Wrench size={18} /></div>
              </div>
              <div className="stat-main">
                <span className="stat-value" style={{ fontSize: '1.2rem' }}>
                  {isFailedCampaign ? 'Re-inspection Required' : record.maintenance_action}
                </span>
                <span className="stat-trend negative">
                  {isFailedCampaign ? 'Inspection Required' : record.maintenance_priority}
                </span>
              </div>
            </div>
          </div>

          {/* AI Assessment & PDF Export Action */}
          <div className="content-grid">
            <section className="surface">
              <div className="panel-heading">
                <div>
                  <p className="eyebrow">AI Executive Narrative</p>
                  <h2>Campaign Summary & Action Plan</h2>
                </div>
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                  <PdfDownloadButton
                    onDownload={downloadReportPdf}
                    disabled={!inspectionId || !record}
                    disabledTooltip="Run a drone campaign inspection to generate a PDF report."
                    label={isFailedCampaign ? 'Download Attempt Report PDF' : 'Download PDF Report'}
                  />
                </div>
              </div>

              <p style={{ fontSize: '0.95rem', lineHeight: 1.6, color: 'var(--ink-subtle)', background: 'var(--surface-alt)', padding: '16px 20px', borderRadius: 'var(--radius-md)', margin: '0 0 20px' }}>
                "{record.summary_report}"
              </p>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '16px' }}>
                <div style={{ padding: '12px 16px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-md)' }}>
                  <span style={{ fontSize: '0.78rem', color: 'var(--muted)', textTransform: 'uppercase', fontWeight: 700 }}>Accepted Images</span>
                  <div style={{ fontSize: '1.2rem', fontWeight: 800, marginTop: '4px' }}>
                    {record.performance_metrics?.accepted_images ?? 0} of {(record.performance_metrics?.accepted_images ?? 0) + (record.performance_metrics?.rejected_images ?? 0)}
                  </div>
                </div>

                <div style={{ padding: '12px 16px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-md)' }}>
                  <span style={{ fontSize: '0.78rem', color: 'var(--muted)', textTransform: 'uppercase', fontWeight: 700 }}>Repair Window</span>
                  <div style={{ fontSize: '1.2rem', fontWeight: 800, marginTop: '4px' }}>
                    {isFailedCampaign ? 'N/A' : `${record.repair_window_days} Days`}
                  </div>
                </div>

                <div style={{ padding: '12px 16px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-md)' }}>
                  <span style={{ fontSize: '0.78rem', color: 'var(--muted)', textTransform: 'uppercase', fontWeight: 700 }}>Next Inspection</span>
                  <div style={{ fontSize: '1.2rem', fontWeight: 800, marginTop: '4px' }}>
                    {isFailedCampaign ? 'Immediate' : `${record.inspection_interval_days} Days`}
                  </div>
                </div>
              </div>

              {/* METRIC PROVENANCE BLOCK */}
              {record.aggregate_results?.provenance && (
                <div style={{ marginTop: '20px', padding: '16px', background: 'var(--surface-glass)', border: '1px solid var(--border-line)', borderRadius: 'var(--radius-md)' }}>
                  <h4 style={{ margin: '0 0 10px 0', fontSize: '0.9rem', color: 'var(--primary)', fontWeight: 700 }}>
                    Metric Evidence Traceability & Provenance
                  </h4>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '10px', fontSize: '0.82rem' }}>
                    <div style={{ padding: '8px 12px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-sm)' }}>
                      <strong style={{ color: 'var(--ink)', display: 'block' }}>SHI Lineage:</strong>
                      <span style={{ color: 'var(--muted)' }}>{record.aggregate_results.provenance.shi_provenance?.derivation}</span>
                    </div>
                    <div style={{ padding: '8px 12px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-sm)' }}>
                      <strong style={{ color: 'var(--ink)', display: 'block' }}>Failure Risk Lineage:</strong>
                      <span style={{ color: 'var(--muted)' }}>{record.aggregate_results.provenance.failure_probability_provenance?.derivation}</span>
                    </div>
                    <div style={{ padding: '8px 12px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-sm)' }}>
                      <strong style={{ color: 'var(--ink)', display: 'block' }}>RUL Lineage:</strong>
                      <span style={{ color: 'var(--muted)' }}>{record.aggregate_results.provenance.rul_provenance?.derivation}</span>
                    </div>
                  </div>
                </div>
              )}

              {/* INSPECTION LIMITATIONS UI CARD */}
              {record.aggregate_results?.inspection_limitations && (
                <div style={{ marginTop: '20px', padding: '16px', background: 'rgba(245, 158, 11, 0.08)', border: '1px solid rgba(245, 158, 11, 0.3)', borderRadius: 'var(--radius-md)' }}>
                  <h4 style={{ margin: '0 0 10px 0', fontSize: '0.9rem', color: '#D97706', fontWeight: 700 }}>
                    Inspection Limitations & Unassessed Regions
                  </h4>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '10px', fontSize: '0.82rem' }}>
                    <div>
                      <strong style={{ color: 'var(--ink)' }}>Observed Surface Area:</strong>{' '}
                      <span>{record.aggregate_results.inspection_limitations.estimated_surface_coverage_pct}%</span>
                    </div>
                    <div>
                      <strong style={{ color: 'var(--ink)' }}>Engineering Confidence Index:</strong>{' '}
                      <span>{record.aggregate_results.inspection_limitations.engineering_confidence_pct}%</span>
                    </div>
                    <div>
                      <strong style={{ color: 'var(--ink)' }}>Uninspected Components:</strong>{' '}
                      <span style={{ color: '#D97706', fontWeight: 600 }}>
                        {record.aggregate_results.inspection_limitations.uninspected_components?.join(', ') || 'None'}
                      </span>
                    </div>
                  </div>
                  <p style={{ margin: '10px 0 0 0', fontSize: '0.78rem', color: 'var(--muted)', fontStyle: 'italic' }}>
                    "{record.aggregate_results.inspection_limitations.uncertified_disclaimer}"
                  </p>
                </div>
              )}

              {rejectedImageResults.length > 0 && (
                <div style={{ marginTop: '20px', padding: '16px', background: 'rgba(254,242,242,0.6)', border: '1px solid var(--border-line)', borderRadius: 'var(--radius-md)' }}>
                  <h4 style={{ margin: '0 0 10px 0', fontSize: '0.9rem', color: 'var(--danger)' }}>Rejected Images Quality Summary ({rejectedImageResults.length} files)</h4>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '0.82rem' }}>
                    {rejectedImageResults.map((rej, i) => (
                      <div key={i} style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px dashed #FECACA', paddingBottom: '4px' }}>
                        <span style={{ fontWeight: 600 }}>{rej.image_name}</span>
                        <span style={{ color: 'var(--danger)' }}>{rej.rejection_reason || (rej.warnings?.[0] ?? 'Quality check failed')}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </section>

            {/* Campaign Visualizer Controls */}
            <section className="surface">
              <div className="panel-heading" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
                <div>
                  <p className="eyebrow">Drone Image Gallery Visualizer</p>
                  <h2 style={{ margin: 0 }}>
                    {validImageResults.length > 0 ? `Photo #${selectedImageIndex + 1} of ${validImageResults.length}` : 'No inspection images available'}
                  </h2>
                </div>

                {validImageResults.length > 1 && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <button
                      type="button"
                      className="btn btn-secondary"
                      style={{ padding: '6px 14px', fontSize: '0.85rem' }}
                      onClick={() => setSelectedImageIndex((prev) => (prev > 0 ? prev - 1 : validImageResults.length - 1))}
                    >
                      ← Previous (←)
                    </button>
                    <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--primary)', padding: '0 4px' }}>
                      {selectedImageIndex + 1} / {validImageResults.length}
                    </span>
                    <button
                      type="button"
                      className="btn btn-secondary"
                      style={{ padding: '6px 14px', fontSize: '0.85rem' }}
                      onClick={() => setSelectedImageIndex((prev) => (prev < validImageResults.length - 1 ? prev + 1 : 0))}
                    >
                      Next (→) →
                    </button>
                  </div>
                )}
              </div>

              {validImageResults.length === 0 ? (
                <div style={{ padding: '40px 20px', textAlign: 'center', color: 'var(--muted)', background: 'var(--surface-alt)', borderRadius: 'var(--radius-md)' }}>
                  <AlertTriangle size={36} style={{ color: 'var(--danger)', marginBottom: '12px' }} />
                  <h3 style={{ margin: '0 0 6px 0', fontSize: '1.05rem', fontWeight: 700 }}>No Inspection Images Available</h3>
                  <p style={{ margin: 0, fontSize: '0.85rem' }}>All uploaded images failed quality validation or no valid component photos were found.</p>
                </div>
              ) : (
                activeImage && (
                  <div>
                    <div style={{ position: 'relative', borderRadius: 'var(--radius-md)', overflow: 'hidden', border: '1px solid var(--border-line)', background: '#000', maxHeight: '380px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <img
                        src={getStaticUrl(activeImage.visualizations?.[activeOverlay] || '')}
                        alt={activeImage.image_name}
                        style={{ maxWidth: '100%', maxHeight: '380px', objectFit: 'contain' }}
                      />
                      
                      <div style={{ position: 'absolute', top: '10px', left: '10px', background: 'rgba(0,0,0,0.7)', color: '#fff', padding: '4px 10px', borderRadius: '4px', fontSize: '0.78rem', fontWeight: 600 }}>
                        {activeImage.image_name}
                      </div>
                    </div>

                    {/* Overlay Selector Controls */}
                    <div style={{ marginTop: '16px', display: 'flex', gap: '8px', flexWrap: 'wrap', justifyContent: 'center' }}>
                      {[
                        { key: 'original', label: 'Original' },
                        { key: 'bboxes', label: 'Bounding Boxes' },
                        { key: 'segmentation', label: 'Segmentation Masks' },
                        { key: 'heatmap', label: 'Heatmap' },
                      ].map((ov) => (
                        <button
                          key={ov.key}
                          type="button"
                          className={`btn ${activeOverlay === ov.key ? 'btn-primary' : 'btn-secondary'}`}
                          style={{ padding: '4px 12px', fontSize: '0.78rem' }}
                          onClick={() => setActiveOverlay(ov.key as any)}
                        >
                          {ov.label}
                        </button>
                      ))}
                    </div>

                    {/* Interactive Thumbnail Strip */}
                    <div style={{ display: 'flex', gap: '10px', overflowX: 'auto', padding: '12px 4px 4px', marginTop: '12px', borderTop: '1px border var(--border-line)' }}>
                      {validImageResults.map((img, idx) => (
                        <div
                          key={idx}
                          onClick={() => setSelectedImageIndex(idx)}
                          style={{
                            minWidth: '72px',
                            height: '52px',
                            borderRadius: '6px',
                            overflow: 'hidden',
                            cursor: 'pointer',
                            border: selectedImageIndex === idx ? '2px solid var(--primary)' : '1px solid var(--border-line)',
                            opacity: selectedImageIndex === idx ? 1 : 0.55,
                            boxShadow: selectedImageIndex === idx ? '0 0 8px rgba(37,99,235,0.4)' : 'none',
                            transition: 'all 0.2s ease'
                          }}
                          title={`Click to view Photo #${idx + 1}: ${img.image_name}`}
                        >
                          <img
                            src={getStaticUrl(img.visualizations?.bboxes || img.visualizations?.original || '')}
                            alt={img.image_name}
                            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                )
              )}
            </section>
          </div>
        </div>
      )}
    </div>
  )
}
