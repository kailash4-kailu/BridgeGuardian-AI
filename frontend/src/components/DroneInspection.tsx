import React, { useState, useEffect, useRef } from 'react'
import {
  Upload,
  Layers,
  AlertTriangle,
  FileDown,
  Sparkles,
  RefreshCw,
  Search,
  CheckCircle,
  Eye,
  Trash2,
  Cpu,
  TrendingDown,
  Clock,
  Wrench,
  Activity,
  Layers3,
  X,
} from 'lucide-react'

import { API_BASE } from '../lib/api'

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
  metrics: {
    blur_score?: number
    brightness?: number
    width?: number
    height?: number
  }
  features?: {
    crack_count: number
    crack_density: number
    avg_crack_length: number
    max_crack_length: number
    avg_crack_width: number
    max_crack_width: number
    corrosion_percent: number
    spalling_percent: number
    leakage_percent: number
    missing_bolts: number
    missing_nuts: number
    loose_connections: number
    vegetation_percent: number
    surface_damage_percent: number
    bridge_tilt: number
    defects: any[]
  }
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
  explainability: {
    summary_report: string
    vision_explanation: string
    feature_explanation: string[]
    ml_contributions: string[]
  } | null
  aggregate_results: {
    largest_crack_width: number
    largest_crack_length: number
    total_crack_area_percent: number
    rust_coverage_percent: number
    corrosion_coverage_percent: number
    critical_defect_count: number
    critical_defect_locations: any[]
    most_damaged_structural_component: string
    affected_structural_components: string[]
    damage_diversity_index: number
    images_containing_damage_percent: number
    maximum_severity: string
    critical_zones: any[]
    hierarchy: Record<string, DefectDetail[]>
    defects?: DefectDetail[]
    prediction_confidence?: number
    health_baseline_score?: number
    point_deductions?: any[]
    component_findings?: any
  } | null
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
  model_metadata: {
    model_name: string
    version: string
    device: string
    threshold: number
  } | null
}

export default function DroneInspection() {
  // Upload and queue state
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const [uploadedList, setUploadedList] = useState<UploadedFile[]>([])
  const [inspectionId, setInspectionId] = useState<number | null>(null)
  const [record, setRecord] = useState<InspectionRecord | null>(null)
  
  // UI filter and navigation state
  const [selectedComponent, setSelectedComponent] = useState<string | null>(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedDefectCategory, setSelectedDefectCategory] = useState<string>('All')
  
  // Visualizer overlay controls
  const [selectedImageIndex, setSelectedImageIndex] = useState<number>(0)
  const [activeOverlay, setActiveOverlay] = useState<'original' | 'bboxes' | 'segmentation' | 'heatmap'>('segmentation')
  
  // Modal for defect details
  const [selectedDefectModal, setSelectedDefectModal] = useState<DefectDetail | null>(null)
  
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  
  const fileInputRef = useRef<HTMLInputElement>(null)
  const folderInputRef = useRef<HTMLInputElement>(null)

  // Auto-poll if inspection is running
  useEffect(() => {
    if (inspectionId === null) return
    
    const interval = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE}/inspection/${inspectionId}`)
        if (!response.ok) throw new Error('Failed to fetch inspection status')
        const data = (await response.json()) as InspectionRecord
        setRecord(data)
        
        if (data.status === 'completed' || data.status === 'failed') {
          clearInterval(interval)
        }
      } catch (err: any) {
        setErrorMsg(err.message)
      }
    }, 2000)
    
    return () => clearInterval(interval)
  }, [inspectionId])

  // Drag and Drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    if (e.dataTransfer.files) {
      const filesArr = Array.from(e.dataTransfer.files).filter((file) =>
        ['image/jpeg', 'image/png'].includes(file.type)
      )
      setSelectedFiles((prev) => [...prev, ...filesArr])
    }
  }

  const handleFileSelection = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const filesArr = Array.from(e.target.files)
      setSelectedFiles((prev) => [...prev, ...filesArr])
    }
  }

  const removeSelectedFile = (idx: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== idx))
  }

  const clearSelection = () => {
    setSelectedFiles([])
    setUploadedList([])
    setInspectionId(null)
    setRecord(null)
    setErrorMsg(null)
  }

  // Upload trigger
  const runInspectionCampaign = async () => {
    if (selectedFiles.length < 5 || selectedFiles.length > 100) {
      setErrorMsg(`Campaign requires between 5 and 100 photos. Currently selected: ${selectedFiles.length}.`)
      return
    }
    setIsUploading(true)
    setErrorMsg(null)
    
    const formData = new FormData()
    selectedFiles.forEach((file) => {
      formData.append('files', file)
    })
    
    try {
      // 1. Upload files
      const uploadRes = await fetch(`${API_BASE}/inspection/upload-images`, {
        method: 'POST',
        body: formData,
      })
      if (!uploadRes.ok) {
        const errDetail = await uploadRes.json()
        throw new Error(errDetail.detail || 'Upload failed')
      }
      
      const filesData = (await uploadRes.json()) as UploadedFile[]
      setUploadedList(filesData)
      
      // 2. Trigger inspection task
      const paths = filesData.map((f) => f.filepath)
      const inspectRes = await fetch(`${API_BASE}/inspection/run-inspection`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_paths: paths, pixel_to_mm: 0.5 }),
      })
      
      if (!inspectRes.ok) {
        throw new Error('Failed to start inspection task')
      }
      
      const inspectData = await inspectRes.json()
      setInspectionId(inspectData.inspection_id)
    } catch (err: any) {
      setErrorMsg(err.message || 'Inspection failed')
    } finally {
      setIsUploading(false)
    }
  }

  // Download PDF Report
  const downloadReportPdf = async () => {
    if (!inspectionId) return
    try {
      const response = await fetch(`${API_BASE}/inspection/report/${inspectionId}`)
      if (!response.ok) throw new Error('Report is not ready or not found')
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `Inspection_Report_CAMP_${inspectionId}.pdf`
      document.body.appendChild(a)
      a.click()
      a.remove()
      window.URL.revokeObjectURL(url)
    } catch (err: any) {
      setErrorMsg(err.message || 'Download failed')
    }
  }

  // Defect categories derived from defects in structural findings
  const defectCategories = ['All', 'Crack', 'Rust', 'Corrosion', 'Spalling', 'Water Leakage', 'Missing Bolt']

  // Filter defects lists based on UI controls
  const filteredUniqueDefects = React.useMemo(() => {
    if (!record || !record.aggregate_results) return []
    
    // Flat list of unique defects
    let list = record.aggregate_results.defects || []
    
    if (selectedComponent) {
      list = list.filter((d: DefectDetail) => d.component === selectedComponent)
    }
    
    if (selectedDefectCategory !== 'All') {
      list = list.filter((d: DefectDetail) => d.type === selectedDefectCategory)
    }
    
    if (searchTerm) {
      list = list.filter((d: DefectDetail) =>
        d.defect_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
        d.type.toLowerCase().includes(searchTerm.toLowerCase())
      )
    }
    
    return list
  }, [record, selectedComponent, selectedDefectCategory, searchTerm])

  const riskCategoryClass = (cat: string | null) => {
    if (!cat) return ''
    if (['Critical', 'Poor'].includes(cat)) return 'risk-red'
    if (cat === 'Fair') return 'risk-yellow'
    return 'risk-green'
  }

  // Map coordinate structure of standard bridge
  const bridgeComponentsLayout = [
    { name: 'Guard Rail', style: { top: '5%', left: '10%', width: '80%', height: '10%' } },
    { name: 'Deck', style: { top: '15%', left: '5%', width: '90%', height: '15%' } },
    { name: 'Expansion Joint', style: { top: '15%', left: '80%', width: '5%', height: '15%' } },
    { name: 'Girder', style: { top: '30%', left: '10%', width: '80%', height: '15%' } },
    { name: 'Connection Plate', style: { top: '30%', left: '25%', width: '12%', height: '15%' } },
    { name: 'Bearing', style: { top: '45%', left: '20%', width: '10%', height: '10%' } },
    { name: 'Pier', style: { top: '55%', left: '15%', width: '20%', height: '40%' } },
    { name: 'Pier', style: { top: '55%', left: '65%', width: '20%', height: '40%' } },
  ]

  // Render variables
  const validImageResults = record?.image_results?.filter((img) => img.is_valid) || []
  const rejectedImageResults = record?.image_results?.filter((img) => !img.is_valid) || []
  const activeImage = validImageResults[selectedImageIndex]

  return (
    <div className="drone-inspection-container">
      {errorMsg && (
        <div className="banner error-banner" role="alert">
          <AlertTriangle size={18} />
          <span>{errorMsg}</span>
          <button className="clear-btn" onClick={() => setErrorMsg(null)}><X size={14} /></button>
        </div>
      )}

      {/* STEP 1: UPLOADER REGION */}
      {inspectionId === null && (
        <div className="surface upload-workspace">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">BridgeGuardian AI Campaigns</p>
              <h2>New Drone Inspection Campaign</h2>
            </div>
            {selectedFiles.length > 0 && (
              <button className="secondary-button" onClick={clearSelection}>Reset</button>
            )}
          </div>

          <input
            type="file"
            ref={fileInputRef}
            multiple
            accept="image/jpeg,image/png"
            style={{ display: 'none' }}
            onChange={handleFileSelection}
          />
          <div
            className="upload-dropzone multi-upload"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="dropzone-inner">
              <Upload size={44} className="upload-icon" />
              <h3>Select 5 - 100 Drone Inspection Photos</h3>
              <p>Drag and drop image files here, or click to upload</p>
              <small>Supported formats: JPG, JPEG, PNG</small>
            </div>
          </div>

          {selectedFiles.length > 0 && (
            <div className="selected-files-preview">
              <div className="preview-header">
                <h3>Images queued ({selectedFiles.length})</h3>
                <button
                  className="primary-button trigger-inspection-btn"
                  onClick={runInspectionCampaign}
                  disabled={isUploading}
                >
                  {isUploading ? (
                    <>
                      <RefreshCw size={16} className="spinner-icon spinning" />
                      Uploading...
                    </>
                  ) : (
                    <>
                      <Sparkles size={16} />
                      Run Campaign Analysis
                    </>
                  )}
                </button>
              </div>

              <div className="thumbnails-grid">
                {selectedFiles.map((file, idx) => (
                  <div key={idx} className="thumb-item">
                    <img src={URL.createObjectURL(file)} alt="thumbnail" />
                    <span className="file-name">{file.name}</span>
                    <button
                      className="remove-thumb-btn"
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation()
                        removeSelectedFile(idx)
                      }}
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

      {/* STEP 2: PROCESSING RUNS */}
      {inspectionId !== null && record && record.status !== 'completed' && record.status !== 'failed' && (
        <div className="surface progress-workspace text-center">
          <RefreshCw size={44} className="spinner-icon spinning text-primary mb-3" />
          <h2>Analyzing Inspection Campaign #{inspectionId}</h2>
          <p>Vision engines running object detection and structural mappings...</p>
          
          <div className="progress-bar-container">
            <div
              className="progress-bar-fill"
              style={{ width: `${record.progress * 100}%` }}
            ></div>
          </div>
          <span className="progress-label">{Math.round(record.progress * 100)}% Complete</span>
        </div>
      )}

      {/* STEP 2b: FAILED STATE */}
      {record && record.status === 'failed' && (
        <div className="surface progress-workspace text-center">
          <AlertTriangle size={44} className="text-red mb-3" />
          <h2>Campaign Analysis Failed</h2>
          <p className="text-muted">The background campaign analysis task encountered an error.</p>
          <button className="primary-button mt-3" onClick={clearSelection} style={{ margin: '0 auto' }}>
            Back to Upload
          </button>
        </div>
      )}

      {/* STEP 3: FINAL DASHBOARD RENDERING */}
      {record && record.status === 'completed' && record.aggregate_results && (
        <div className="campaign-dashboard">
          {/* BANNER DEMO INDICATOR */}
          {import.meta.env.DEV && (
            <div className="banner demo-banner" style={{ background: '#FFF5F5', border: '1px solid #C53030', color: '#C53030', marginBottom: '20px' }}>
              <Sparkles size={18} />
              <span><b>Demo Mode Enabled:</b> Visual outputs and model predictions are simulated. Physical calculations are calibrated for demonstration.</span>
            </div>
          )}

          {/* KPI CARDS GRID */}
          <div className="kpi-cards-grid">
            <div className="kpi-card">
              <span className="kpi-label">Health Score (SHI)</span>
              <strong className="kpi-value">{record.health_score}%</strong>
              <span className={`kpi-badge risk-${record.risk_category?.toLowerCase() || ''}`}>{record.risk_category || 'Unknown'} Condition</span>
            </div>
            <div className="kpi-card">
              <span className="kpi-label">Failure Probability</span>
              <strong className="kpi-value text-red">{record.failure_probability}%</strong>
              <span className="kpi-subtext">Calibrated Overlays</span>
            </div>
            <div className="kpi-card">
              <span className="kpi-label">Remaining Useful Life</span>
              <strong className="kpi-value text-info" style={{ color: '#2B6CB0' }}>{record.rul_days} Days</strong>
              <span className="kpi-subtext">Estimated End-of-Life</span>
            </div>
            <div className="kpi-card">
              <span className="kpi-label">Risk Category</span>
              <strong className="kpi-value text-warning" style={{ color: '#DD6B20' }}>{record.risk_category}</strong>
              <span className="kpi-subtext">Structural Alert Level</span>
            </div>
            <div className="kpi-card">
              <span className="kpi-label">Maintenance Priority</span>
              <strong className="kpi-value text-danger" style={{ color: '#E53E3E' }}>{record.maintenance_priority}</strong>
              <span className="kpi-subtext">Remediation Urgency</span>
            </div>
            <div className="kpi-card">
              <span className="kpi-label">Inspection Confidence</span>
              <strong className="kpi-value text-success" style={{ color: '#48BB78' }}>
                {Math.round((record.aggregate_results.prediction_confidence ?? 0.95) * 100)}%
              </strong>
              <span className="kpi-subtext">Multiplicative Mapping</span>
            </div>
          </div>

          {/* SECTION 1: AI SUMMARY & REMEDIATION PLAN */}
          <div className="dashboard-top-row">
            {/* AI Narrative assessment */}
            <div className="surface details-report-card" style={{ flex: '1 1 50%' }}>
              <div className="panel-heading">
                <div>
                  <p className="eyebrow">Explainability Summary</p>
                  <h2>AI Inspection Assessment</h2>
                </div>
                <button className="primary-button icon-button-text" onClick={downloadReportPdf}>
                  <FileDown size={16} />
                  Download PDF Report
                </button>
              </div>
              <p className="ai-narrative-summary mb-3" style={{ fontSize: '0.9rem', fontStyle: 'italic', color: '#4A5568' }}>
                "{record.summary_report}"
              </p>
              
              {/* Maintenance Cards Remediation */}
              <div className="maintenance-status-container mt-4 p-3 border-rounded" style={{ background: 'var(--surface-alt)', border: '1px solid var(--line)' }}>
                <div className="status-indicator-block">
                  {record.maintenance_action === 'Replace' && (
                    <div className="status-badge red">
                      <span className="dot">🔴</span>
                      <strong>REPLACE</strong>
                    </div>
                  )}
                  {record.maintenance_action === 'Repair' && (
                    <div className="status-badge orange">
                      <span className="dot">🟠</span>
                      <strong>REPAIR</strong>
                    </div>
                  )}
                  {record.maintenance_action === 'Inspect' && (
                    <div className="status-badge yellow">
                      <span className="dot">🟡</span>
                      <strong>INSPECT</strong>
                    </div>
                  )}
                  {record.maintenance_action === 'Monitor' && (
                    <div className="status-badge green">
                      <span className="dot">🟢</span>
                      <strong>MONITOR</strong>
                    </div>
                  )}
                </div>
                
                <div className="maintenance-details-grid">
                  <div>
                    <span>Repair Window</span>
                    <strong>{record.repair_window_days} Days</strong>
                  </div>
                  <div>
                    <span>Next Inspection</span>
                    <strong>{record.inspection_interval_days} Days</strong>
                  </div>
                  <div>
                    <span>Reasoning</span>
                    <small className="text-muted leading-tight">
                      Aggregated visual severity of {record.aggregate_results.maximum_severity} mapped to the {record.aggregate_results.most_damaged_structural_component} component.
                    </small>
                  </div>
                </div>
              </div>
            </div>

            {/* Asset Maintenance Timeline */}
            <div className="surface timeline-card" style={{ flex: '1 1 45%' }}>
              <div className="panel-heading">
                <h2>Asset Maintenance Timeline</h2>
              </div>
              <div className="timeline-flex mt-4">
                <div className="timeline-node active">
                  <div className="node-dot"></div>
                  <span>Today</span>
                  <small>Assessment</small>
                </div>
                <div className="timeline-line"></div>
                <div className="timeline-node warning">
                  <div className="node-dot"></div>
                  <span>Repair Window</span>
                  <small>Within {record.repair_window_days}d</small>
                </div>
                <div className="timeline-line"></div>
                <div className="timeline-node info">
                  <div className="node-dot"></div>
                  <span>Re-Inspect</span>
                  <small>At {record.inspection_interval_days}d</small>
                </div>
                <div className="timeline-line"></div>
                <div className="timeline-node danger">
                  <div className="node-dot"></div>
                  <span>Limit</span>
                  <small>RUL: {record.rul_days}d</small>
                </div>
              </div>
            </div>
          </div>

          {/* SECTION 2: EXPLAINABILITY & IMAGE QUALITY DASHBOARD */}
          <div className="dashboard-top-row mt-4">
            {/* SHAP explainability cards */}
            <div className="surface explainability-shap-card" style={{ flex: '1 1 50%' }}>
              <div className="panel-heading">
                <h2>Predictive Health Score Adjustments (SHAP)</h2>
              </div>
              <p className="text-muted mb-3" style={{ fontSize: '0.82rem' }}>
                The Prediction Engine evaluated a baseline score of <b>{record.aggregate_results.health_baseline_score}%</b>. Visually detected defects applied the following point deductions:
              </p>
              <div className="shap-deductions-grid">
                {record.aggregate_results.point_deductions?.map((ded: any, idx: number) => (
                  <div key={idx} className="shap-deduction-card">
                    <div className="shap-header">
                      <strong>{ded.feature}</strong>
                      <span>{ded.value}</span>
                    </div>
                    <div className="shap-arrow">↓</div>
                    <div className="shap-points">-{ded.deduction} points</div>
                  </div>
                ))}
                {(!record.aggregate_results.point_deductions || record.aggregate_results.point_deductions.length === 0) && (
                  <p className="text-success text-center w-full py-4">No point deductions applied. Structural components remain healthy.</p>
                )}
              </div>
            </div>

            {/* Ingestion Image Quality stats */}
            <div className="surface quality-stats-card" style={{ flex: '1 1 45%' }}>
              <div className="panel-heading">
                <h2>UAV Image Ingestion Quality Gate</h2>
              </div>
              <div className="quality-stats-grid mt-3">
                <div className="quality-metric-item">
                  <span>Images Uploaded</span>
                  <strong>{validImageResults.length + rejectedImageResults.length}</strong>
                </div>
                <div className="quality-metric-item">
                  <span>Accepted (Passed)</span>
                  <strong className="text-success">{validImageResults.length}</strong>
                </div>
                <div className="quality-metric-item">
                  <span>Rejected (Failed)</span>
                  <strong className={rejectedImageResults.length > 0 ? 'text-red' : 'text-muted'}>{rejectedImageResults.length}</strong>
                </div>
                <div className="quality-metric-item">
                  <span>Avg Quality Score</span>
                  <strong>{record.performance_metrics?.avg_image_quality ?? 0}%</strong>
                </div>
              </div>
              
              {rejectedImageResults.length > 0 ? (
                <div className="rejection-breakdown mt-4">
                  <h3>Quality Rejection Reasons Breakdown</h3>
                  <div className="breakdown-grid mt-2">
                    <div className="breakdown-item">
                      <span>Blurry Frames</span>
                      <strong>{rejectedImageResults.filter(img => img.warnings.some(w => w.toLowerCase().includes('blur'))).length}</strong>
                    </div>
                    <div className="breakdown-item">
                      <span>Dark / Lighting</span>
                      <strong>{rejectedImageResults.filter(img => img.warnings.some(w => w.toLowerCase().includes('dark') || w.toLowerCase().includes('light'))).length}</strong>
                    </div>
                    <div className="breakdown-item">
                      <span>Low Resolution</span>
                      <strong>{rejectedImageResults.filter(img => img.warnings.some(w => w.toLowerCase().includes('resolution') || w.toLowerCase().includes('res'))).length}</strong>
                    </div>
                    <div className="breakdown-item">
                      <span>Duplicate Frames</span>
                      <strong>{rejectedImageResults.filter(img => img.warnings.some(w => w.toLowerCase().includes('duplicate'))).length}</strong>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="mt-4 p-3 border-rounded text-success text-center" style={{ background: 'rgba(72, 187, 120, 0.05)', border: '1px solid rgba(72, 187, 120, 0.15)', fontSize: '0.8rem' }}>
                  🎉 All drone images successfully passed resolution, focus, and illumination quality gates.
                </div>
              )}
            </div>
          </div>

          {/* SECTION 3: INTERACTIVE SCHEMA COMPONENT MAP & DEFECT GALLERY */}
          <div className="dashboard-middle-row mt-4">
            {/* Interactive Bridge Schematic */}
            <div className="surface component-map-card" style={{ flex: '1 1 50%' }}>
              <div className="panel-heading">
                <div>
                  <p className="eyebrow">Structural Localization</p>
                  <h2>Interactive Bridge Schematic Map</h2>
                </div>
                {selectedComponent && (
                  <button className="clear-btn text-primary" onClick={() => setSelectedComponent(null)}>
                    Reset Filter
                  </button>
                )}
              </div>
              <p className="text-muted mb-3" style={{ fontSize: '0.82rem' }}>
                Select a structural element on the schematic to display dimensions, visual defect counts, and confidence.
              </p>
              
              <div className="bridge-visual-schematic">
                {['Deck', 'Girder', 'Pier', 'Bearing', 'Expansion Joint', 'Guard Rail'].map((comp, idx) => {
                  const compDefs = record.aggregate_results!.hierarchy?.[comp] ?? []
                  const hasDefects = compDefs.length > 0
                  const isActive = selectedComponent === comp
                  
                  return (
                    <div key={idx} className="schematic-node-wrapper">
                      <button
                        className={`schematic-node ${isActive ? 'active' : ''} ${hasDefects ? 'has-defects' : ''}`}
                        onClick={() => setSelectedComponent(isActive ? null : comp)}
                      >
                        <span className="node-title">{comp}</span>
                        <span className="node-meta" style={{ color: hasDefects ? '#E53E3E' : '#48BB78', fontWeight: 'bold' }}>
                          {hasDefects ? `${compDefs.length} Defects` : 'Healthy'}
                        </span>
                      </button>
                      {idx < 5 && <div className="schematic-arrow">➔</div>}
                    </div>
                  )
                })}
              </div>

              {/* Dynamic details for the selected component */}
              {selectedComponent ? (() => {
                const compDefs = record.aggregate_results.hierarchy?.[selectedComponent] ?? []
                const findings = record.aggregate_results.component_findings?.find((f: any) => f.component === selectedComponent)
                return (
                  <div className="selected-component-info p-3 border-rounded mt-3">
                    <h3 style={{ fontSize: '0.9rem', color: '#1A365D' }}>Selected: <b>{selectedComponent}</b></h3>
                    <div className="info-grid mt-2" style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px', fontSize: '0.8rem' }}>
                      <div>
                        <span>Status Recommendation:</span>
                        <strong style={{ display: 'block', fontSize: '0.95rem' }} className={findings?.status === 'Repair' || findings?.status === 'Replace' ? 'text-red' : 'text-success'}>
                          {findings?.status ?? 'OK'}
                        </strong>
                      </div>
                      <div>
                        <span>Worst Defect Severity:</span>
                        <strong style={{ display: 'block', fontSize: '0.95rem' }}>{findings?.severity ?? 'Healthy'}</strong>
                      </div>
                      <div>
                        <span>Defect Occurrences:</span>
                        <strong style={{ display: 'block', fontSize: '0.95rem' }}>{compDefs.length} unique items</strong>
                      </div>
                    </div>
                    {compDefs.length > 0 && (
                      <div className="comp-measurements mt-3" style={{ fontSize: '0.78rem' }}>
                        <span className="text-muted">Detected Defect Parameters:</span>
                        <ul className="pl-4 mt-1">
                          {compDefs.map((d: any, idx: number) => (
                            <li key={idx} className="mb-1 text-muted">
                              <b>{d.type}</b> (Severity: {d.severity}) | Confidence: {Math.round(d.confidence * 100)}%
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )
              })() : (
                <div className="p-3 border-rounded mt-3 text-center text-muted" style={{ background: 'var(--surface-alt)', border: '1px dashed var(--line)', fontSize: '0.8rem' }}>
                  💡 Click any component node above to inspect defect logs and severity metrics.
                </div>
              )}
            </div>

            {/* Redesigned Defect Gallery */}
            <div className="surface defects-gallery-card" style={{ flex: '1 1 45%' }}>
              <div className="panel-heading flex-column flex-start">
                <h2>Defect Gallery</h2>
                <div className="gallery-search-bar mt-2 w-full">
                  <div className="search-input-wrapper">
                    <Search size={14} className="search-icon" />
                    <input
                      type="text"
                      placeholder="Search defects..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                    />
                  </div>
                  <select
                    value={selectedDefectCategory}
                    onChange={(e) => setSelectedDefectCategory(e.target.value)}
                  >
                    {defectCategories.map((c) => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="defects-gallery-list mt-3">
                {filteredUniqueDefects.map((def: DefectDetail) => {
                  const matchedImg = validImageResults.find((img) => img.image_name === def.images[0])
                  const thumbnailSrc = matchedImg?.visualizations?.bboxes ?? matchedImg?.visualizations?.original
                  
                  return (
                    <div
                      key={def.defect_id}
                      className={`defect-gallery-item severity-${def.severity.toLowerCase()}`}
                      onClick={() => setSelectedDefectModal(def)}
                    >
                      {thumbnailSrc && (
                        <div className="defect-thumbnail">
                          <img src={thumbnailSrc} alt={def.type} />
                        </div>
                      )}
                      <div className="defect-details-content">
                        <div className="defect-item-header d-flex justify-between align-center">
                          <strong>{def.defect_id}</strong>
                          <span className="defect-badge" style={{ background: 'var(--line)', padding: '2px 6px', borderRadius: '4px', fontSize: '0.65rem' }}>{def.type}</span>
                        </div>
                        <p className="defect-desc-text text-muted" style={{ fontSize: '0.72rem', margin: '4px 0' }}>
                          Zone: <b>{def.component}</b> | Severity: <b className={`severity-text-${def.severity.toLowerCase()}`}>{def.severity}</b>
                        </p>
                        <div className="defect-item-footer d-flex justify-between" style={{ fontSize: '0.68rem', color: 'var(--muted)' }}>
                          <span>Matches: {def.occurrences} images</span>
                          <span>Confidence: {Math.round(def.confidence * 100)}%</span>
                        </div>
                      </div>
                    </div>
                  )
                })}
                {filteredUniqueDefects.length === 0 && (
                  <p className="text-center text-muted mt-5" style={{ fontSize: '0.8rem' }}>No defects match the selected filters.</p>
                )}
              </div>
            </div>
          </div>

          {/* SECTION 4: Component findings matrix table */}
          <div className="surface component-findings-table-card mt-4">
            <div className="panel-heading">
              <h2>Structural Component Findings Matrix</h2>
            </div>
            <table className="component-findings-table mt-3">
              <thead>
                <tr>
                  <th>Component</th>
                  <th>Cracks Present</th>
                  <th>Rust / Corrosion</th>
                  <th>Worst Defect Severity</th>
                  <th>Status Action</th>
                </tr>
              </thead>
              <tbody>
                {record.aggregate_results.component_findings?.map((item: any, idx: number) => {
                  const statusClass = item.status === 'Replace' ? 'status-replace' : 
                                    item.status === 'Repair' ? 'status-repair' : 
                                    item.status === 'Inspect' ? 'status-inspect' : 'status-ok'
                  return (
                    <tr key={idx}>
                      <td><b>{item.component}</b></td>
                      <td>{item.cracks}</td>
                      <td>{item.rust}</td>
                      <td>
                        <span className={`badge-severity ${item.severity.toLowerCase()}`}>
                          {item.severity}
                        </span>
                      </td>
                      <td>
                        <span className={`badge-status ${statusClass}`}>{item.status}</span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          {/* SECTION 5: IMAGE VISUALIZATION WORKSPACE WITH DETECTION LOG PANEL */}
          <div className="dashboard-bottom-row mt-4">
            <div className="surface visualizer-full-card" style={{ flex: '1 1 100%' }}>
              <div className="panel-heading">
                <div>
                  <p className="eyebrow">Drone Campaign Ingestion</p>
                  <h2>Campaign Visualizer</h2>
                </div>
                {activeImage && (
                  <span className="active-img-badge" style={{ background: 'var(--line)', padding: '4px 8px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 'bold' }}>
                    {activeImage.image_name}
                  </span>
                )}
              </div>

              {activeImage ? (
                <div className="campaign-visualizer-workspace mt-3">
                  {/* Left screen view */}
                  <div className="visualizer-display-container">
                    <div className="visualizer-display" style={{ background: '#000', borderRadius: '8px', overflow: 'hidden', textAlign: 'center', height: '340px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <img
                        src={activeImage.visualizations?.[activeOverlay]}
                        alt="Visualizer Feed"
                        style={{ maxHeight: '100%', maxWidth: '100%', objectFit: 'contain' }}
                      />
                    </div>
                    <div className="visualizer-controls mt-2">
                      <div className="overlay-tabs-row d-flex gap-2">
                        {[
                          { key: 'original', label: 'Original' },
                          { key: 'bboxes', label: 'Detections' },
                          { key: 'segmentation', label: 'SAM Segmentation' },
                          { key: 'heatmap', label: 'Damage Heatmap' },
                        ].map((ov) => (
                          <button
                            key={ov.key}
                            className={`tab-btn ${activeOverlay === ov.key ? 'active' : ''}`}
                            onClick={() => setActiveOverlay(ov.key as any)}
                            style={{ flexGrow: 1, padding: '6px 12px', fontSize: '0.78rem' }}
                          >
                            {ov.label}
                          </button>
                        ))}
                      </div>
                      
                      {/* Left/Right image switcher */}
                      <div className="image-switcher-row d-flex justify-between align-center mt-2" style={{ fontSize: '0.8rem' }}>
                        <button
                          className="secondary-button"
                          style={{ padding: '4px 10px', fontSize: '0.75rem' }}
                          disabled={selectedImageIndex === 0}
                          onClick={() => setSelectedImageIndex((prev) => prev - 1)}
                        >
                          Prev Image
                        </button>
                        <span>Image {selectedImageIndex + 1} of {validImageResults.length}</span>
                        <button
                          className="secondary-button"
                          style={{ padding: '4px 10px', fontSize: '0.75rem' }}
                          disabled={selectedImageIndex === validImageResults.length - 1}
                          onClick={() => setSelectedImageIndex((prev) => prev + 1)}
                        >
                          Next Image
                        </button>
                      </div>
                    </div>
                  </div>

                  {/* Right side metadata panel */}
                  <div className="visualizer-metadata-panel p-3 border-rounded">
                    <h3 style={{ fontSize: '0.82rem', textTransform: 'uppercase', color: 'var(--muted)' }}>Frame Detection Logs</h3>
                    <div className="image-detections-list mt-2" style={{ maxHeight: '280px', overflowY: 'auto' }}>
                      {activeImage.features?.defects?.map((det: any, idx: number) => (
                        <div key={idx} className="det-item border-bottom py-2" style={{ borderBottom: '1px solid var(--line)' }}>
                          <div className="d-flex justify-between align-center">
                            <strong style={{ color: 'var(--ink)' }}>{det.type}</strong>
                            <span className={`badge-severity ${det.severity.toLowerCase()}`}>{det.severity}</span>
                          </div>
                          <div className="det-item-details text-muted mt-1" style={{ fontSize: '0.7rem' }}>
                            <span>Conf: {Math.round(det.confidence * 100)}%</span>
                            {det.type === 'Crack' && (
                              <span> | W: {det.measurements?.width_mm}mm, L: {det.measurements?.length_mm}mm</span>
                            )}
                            <span> | Area: {det.measurements?.area_pct}%</span>
                          </div>
                        </div>
                      ))}
                      {(!activeImage.features?.defects || activeImage.features.defects.length === 0) && (
                        <p className="text-muted text-center py-5" style={{ fontSize: '0.75rem' }}>No surface defects registered in this frame.</p>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-center mt-5 text-muted">No valid image overlays generated.</p>
              )}
            </div>
          </div>

          {/* SECTION 6: MODEL COMPILER SPECIFICATIONS AUDIT */}
          <div className="dashboard-bottom-row mt-4">
            {/* Performance and Vision Audit Panel */}
            <div className="surface performance-audit-card" style={{ flex: '1 1 100%' }}>
              <div className="panel-heading">
                <h2>Vision Engine &amp; Performance Audit</h2>
              </div>
              
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                {record.model_metadata && (
                  <div className="audit-section">
                    <h3>Vision Model Specifications</h3>
                    <div className="audit-grid">
                      <div>
                        <span>Detector Engine</span>
                        <strong>{record.model_metadata.model_name}</strong>
                      </div>
                      <div>
                        <span>Model Version</span>
                        <strong>{record.model_metadata.version}</strong>
                      </div>
                      <div>
                        <span>Confidence Threshold</span>
                        <strong>{record.model_metadata.threshold * 100}%</strong>
                      </div>
                      <div>
                        <span>Inference Device</span>
                        <strong>{record.model_metadata.device}</strong>
                      </div>
                    </div>
                  </div>
                )}

                {record.performance_metrics && (
                  <div className="audit-section">
                    <h3>Runtime Performance Diagnostics</h3>
                    <div className="audit-grid">
                      <div>
                        <span>Total Time</span>
                        <strong>{record.performance_metrics.total_processing_time_sec}s</strong>
                      </div>
                      <div>
                        <span>Ingestion Rate</span>
                        <strong>{record.performance_metrics.images_per_second} frames/s</strong>
                      </div>
                      <div>
                        <span>Avg Frame Quality</span>
                        <strong>{record.performance_metrics.avg_image_quality}%</strong>
                      </div>
                      <div>
                        <span>RAM Allocation</span>
                        <strong>{record.performance_metrics.memory_usage_mb} MB</strong>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* SECTION 7: SCIENTIFIC TRANSPARENCY & LIMITATIONS WARNINGS */}
          <div className="surface limitations-footer-card mt-4 border-dashed border-red" style={{ background: '#FFFDFD', border: '1px dashed #E53E3E', borderRadius: '8px', padding: '16px' }}>
            <h3 style={{ color: '#C53030', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <AlertTriangle size={16} />
              Scientific Transparency &amp; Model Limitations
            </h3>
            <p className="text-muted mt-1 leading-normal" style={{ fontSize: '0.8rem' }}>
              <b>Measurement vs Prediction Calibration:</b> Defect widths, lengths, and corrosion surface coverage percentages represent direct pixel measurements extracted from UAV drone imagery. The resulting Health Score (SHI), Failure Probability, and Remaining Useful Life are model-predicted values calculated using a calibrated tabular baseline modified by worst visual defect parameters.
            </p>
            <p className="text-muted mt-2 leading-normal" style={{ fontSize: '0.8rem' }}>
              <b>Core Model Limitation Disclosure:</b> Standard RGB cameras cannot detect subsurface delamination, core material fatigue, internal stresses, or hidden structural failures. Predictions assume normal structural status for unobserved internal variables. Drone imagery campaigns should be verified by periodic non-destructive ultrasonic or magnetic induction inspections.
            </p>
          </div>
        </div>
      )}

      {/* DEFECT SPECIFIC MODAL DETAILED VIEWER */}
      {selectedDefectModal && (
        <div className="modal-backdrop" onClick={() => setSelectedDefectModal(null)}>
          <div className="modal-surface" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>Defect File Card: {selectedDefectModal.defect_id}</h2>
              <button className="clear-btn" onClick={() => setSelectedDefectModal(null)}>Close</button>
            </div>
            <div className="modal-body">
              <div className="modal-metrics-grid">
                <div>
                  <span>Defect Class</span>
                  <strong>{selectedDefectModal.type}</strong>
                </div>
                <div>
                  <span>Severity Level</span>
                  <strong className={`severity-text-${selectedDefectModal.severity.toLowerCase()}`}>
                    {selectedDefectModal.severity}
                  </strong>
                </div>
                <div>
                  <span>Parent Component</span>
                  <strong>{selectedDefectModal.component}</strong>
                </div>
                <div>
                  <span>Detection Confidence</span>
                  <strong>{Math.round(selectedDefectModal.confidence * 100)}%</strong>
                </div>
              </div>

              <div className="modal-section-box mt-3">
                <h3>OpenCV Metrics Measurements</h3>
                <ul>
                  {selectedDefectModal.type === 'Crack' && (
                    <>
                      <li>Estimated Crack Length: <b>{selectedDefectModal.measurements.length_mm} mm</b></li>
                      <li>Estimated Crack Width: <b>{selectedDefectModal.measurements.width_mm} mm</b></li>
                    </>
                  )}
                  <li>Image Area Coverage Ratio: <b>{selectedDefectModal.measurements.area_pct}%</b></li>
                </ul>
              </div>

              <div className="modal-section-box mt-3">
                <h3>Mapped Campaign Images ({selectedDefectModal.images.length})</h3>
                <div className="mapped-images-badges">
                  {selectedDefectModal.images.map((imgName, index) => (
                    <span key={index} className="img-badge">{imgName}</span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
