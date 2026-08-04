import React, { useEffect, useMemo, useState } from 'react'
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  BarChart3,
  Bell,
  Camera,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  CircleGauge,
  Database,
  Eye,
  FileDown,
  Filter,
  History,
  Image as ImageIcon,
  Info,
  Layers,
  Plane,
  RefreshCw,
  Route,
  Search,
  Send,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Upload,
  User,
  Waves,
  X,
} from 'lucide-react'
import './App.css'
import DroneInspection from './components/DroneInspection'
import SplashScreen from './components/SplashScreen'
import { API_BASE, getStaticUrl } from './lib/api'
import { compressImage } from './lib/imageUtils'

type SensorPayload = Record<string, number | string | null>

type HealthResponse = {
  status: string
  version: string
  model_ready: boolean
  database_ok: boolean
  timestamp: string
}

type ModelInfoResponse = {
  is_ready: boolean
  model_version: string
  models_available: string[]
  feature_count: number
  training_results: Record<string, unknown> | null
}

type PredictionResponse = {
  prediction_id: number | null
  timestamp: string
  health_score: number
  health_score_raw: number
  failure_probability: number
  failure_probability_raw: number
  rul_days: number
  rul_degradation_rate: number
  rul_confidence: string
  rul_message: string
  risk_category: string
  maintenance_priority: string
  maintenance_recommendation: string
  maintenance_alert: boolean
  prediction_confidence: number
  model_version: string
}

type FeatureImportance = {
  feature: string
  shap_value: number
  direction: string
}

type ExplainResponse = {
  target: string
  explanation: {
    base_value: number
    feature_importances: FeatureImportance[]
    top_positive_features: FeatureImportance[]
    top_negative_features: FeatureImportance[]
    prediction_contribution: number
    note?: string | null
  }
}

type HistoryItem = {
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

type HistoryResponse = {
  items: HistoryItem[]
  total: number
}

type SensorField = {
  key: string
  label: string
  unit?: string
  group: FieldGroup
  step?: number
  options?: string[]
}

type FieldGroup = 'structure' | 'environment' | 'load' | 'diagnostics'

type ApiState = 'checking' | 'online' | 'degraded' | 'offline'

const DEFAULT_INPUT: SensorPayload = {
  Strain_microstrain: 734.5,
  Deflection_mm: 14.99,
  Vibration_ms2: 1.2,
  Tilt_deg: 0.72,
  Displacement_mm: 22.36,
  Crack_Propagation_mm: 0.015,
  Corrosion_Level_percent: 0.15,
  Cable_Member_Tension_kN: 447.9,
  Bearing_Joint_Forces_kN: 260.1,
  Fatigue_Accumulation_au: 0.3,
  Modal_Frequency_Hz: 1.9,
  Temperature_C: 15,
  Humidity_percent: 60.3,
  Wind_Speed_ms: 6.5,
  Wind_Direction_deg: 180,
  Precipitation_mmh: 0,
  Water_Level_m: 2,
  Seismic_Activity_ms2: 0,
  Solar_Radiation_Wm2: 446.5,
  Air_Quality_Index_AQI: 55,
  Soil_Settlement_mm: 0.3,
  Vehicle_Load_tons: 16.4,
  Traffic_Volume_vph: 853.2,
  Pedestrian_Load_pph: 96.3,
  Impact_Events_g: 0,
  Dynamic_Load_Distribution_percent: 90.1,
  Axle_Counts_pmin: 43.4,
  Anomaly_Detection_Score: 0,
  Energy_Dissipation_au: 0.156,
  Acoustic_Emissions_levels: 10.45,
  Visual_Analysis_Defect_Score: 0.004,
  Electrical_Resistance_ohms: 0.282,
  Localized_Strain_Hotspot: 0,
  Bridge_Mood_Meter: 'Healthy',
  Vibration_Anomaly_Location: 'Unknown',
  Flood_Event_Flag: 0,
  High_Winds_Storms: 0,
  Landslide_Ground_Movement: 0,
  Abnormal_Traffic_Load_Surges: 0,
  SHI_Predicted_7d_Ahead: 0.8,
  SHI_Predicted_30d_Ahead: 0.75,
}

const CRITICAL_PRESET: SensorPayload = {
  ...DEFAULT_INPUT,
  Strain_microstrain: 1850.0,
  Deflection_mm: 48.5,
  Vibration_ms2: 5.8,
  Tilt_deg: 3.2,
  Crack_Propagation_mm: 4.8,
  Corrosion_Level_percent: 12.5,
  Bridge_Mood_Meter: 'Critical',
  Anomaly_Detection_Score: 0.85,
}

const SENSOR_FIELDS: SensorField[] = [
  { key: 'Strain_microstrain', label: 'Strain', unit: 'microstrain', group: 'structure', step: 0.1 },
  { key: 'Deflection_mm', label: 'Deflection', unit: 'mm', group: 'structure', step: 0.01 },
  { key: 'Vibration_ms2', label: 'Vibration', unit: 'm/s2', group: 'structure', step: 0.01 },
  { key: 'Tilt_deg', label: 'Tilt', unit: 'deg', group: 'structure', step: 0.01 },
  { key: 'Displacement_mm', label: 'Displacement', unit: 'mm', group: 'structure', step: 0.01 },
  { key: 'Crack_Propagation_mm', label: 'Crack growth', unit: 'mm', group: 'structure', step: 0.001 },
  { key: 'Corrosion_Level_percent', label: 'Corrosion', unit: '%', group: 'structure', step: 0.01 },
  { key: 'Cable_Member_Tension_kN', label: 'Cable tension', unit: 'kN', group: 'structure', step: 0.1 },
  { key: 'Bearing_Joint_Forces_kN', label: 'Bearing forces', unit: 'kN', group: 'structure', step: 0.1 },
  { key: 'Fatigue_Accumulation_au', label: 'Fatigue', unit: 'a.u.', group: 'structure', step: 0.01 },
  { key: 'Modal_Frequency_Hz', label: 'Modal frequency', unit: 'Hz', group: 'structure', step: 0.01 },
  { key: 'Temperature_C', label: 'Temperature', unit: 'C', group: 'environment', step: 0.1 },
  { key: 'Humidity_percent', label: 'Humidity', unit: '%', group: 'environment', step: 0.1 },
  { key: 'Wind_Speed_ms', label: 'Wind speed', unit: 'm/s', group: 'environment', step: 0.1 },
  { key: 'Wind_Direction_deg', label: 'Wind direction', unit: 'deg', group: 'environment', step: 1 },
  { key: 'Precipitation_mmh', label: 'Precipitation', unit: 'mm/h', group: 'environment', step: 0.1 },
  { key: 'Water_Level_m', label: 'Water level', unit: 'm', group: 'environment', step: 0.1 },
  { key: 'Seismic_Activity_ms2', label: 'Seismic activity', unit: 'm/s2', group: 'environment', step: 0.001 },
  { key: 'Solar_Radiation_Wm2', label: 'Solar radiation', unit: 'W/m2', group: 'environment', step: 1 },
  { key: 'Air_Quality_Index_AQI', label: 'Air quality', unit: 'AQI', group: 'environment', step: 1 },
  { key: 'Soil_Settlement_mm', label: 'Soil settlement', unit: 'mm', group: 'environment', step: 0.01 },
  { key: 'Vehicle_Load_tons', label: 'Vehicle load', unit: 'tons', group: 'load', step: 0.1 },
  { key: 'Traffic_Volume_vph', label: 'Traffic volume', unit: 'veh/h', group: 'load', step: 1 },
  { key: 'Pedestrian_Load_pph', label: 'Pedestrian load', unit: 'people/h', group: 'load', step: 1 },
  { key: 'Impact_Events_g', label: 'Impact events', unit: 'g', group: 'load', step: 0.001 },
  { key: 'Dynamic_Load_Distribution_percent', label: 'Load distribution', unit: '%', group: 'load', step: 0.1 },
  { key: 'Axle_Counts_pmin', label: 'Axle count', unit: '/min', group: 'load', step: 0.1 },
  { key: 'Anomaly_Detection_Score', label: 'Anomaly score', group: 'diagnostics', step: 0.01 },
  { key: 'Energy_Dissipation_au', label: 'Energy dissipation', unit: 'a.u.', group: 'diagnostics', step: 0.001 },
  { key: 'Acoustic_Emissions_levels', label: 'Acoustic emissions', group: 'diagnostics', step: 0.01 },
  { key: 'Visual_Analysis_Defect_Score', label: 'Visual defect score', group: 'diagnostics', step: 0.001 },
  { key: 'Electrical_Resistance_ohms', label: 'Electrical resistance', unit: 'ohms', group: 'diagnostics', step: 0.001 },
  { key: 'Localized_Strain_Hotspot', label: 'Localized hotspot', group: 'diagnostics', step: 1 },
  {
    key: 'Bridge_Mood_Meter',
    label: 'Bridge mood',
    group: 'diagnostics',
    options: ['Healthy', 'Stressed', 'Critical'],
  },
  {
    key: 'Vibration_Anomaly_Location',
    label: 'Vibration location',
    group: 'diagnostics',
    options: ['Unknown', 'Deck', 'Cables', 'Piers'],
  },
  { key: 'Flood_Event_Flag', label: 'Flood event', group: 'diagnostics', step: 1 },
  { key: 'High_Winds_Storms', label: 'High winds', group: 'diagnostics', step: 1 },
  { key: 'Landslide_Ground_Movement', label: 'Ground movement', group: 'diagnostics', step: 1 },
  { key: 'Abnormal_Traffic_Load_Surges', label: 'Traffic surge', group: 'diagnostics', step: 1 },
  { key: 'SHI_Predicted_7d_Ahead', label: 'SHI 7d forecast', group: 'diagnostics', step: 0.01 },
  { key: 'SHI_Predicted_30d_Ahead', label: 'SHI 30d forecast', group: 'diagnostics', step: 0.01 },
]

const FIELD_GROUPS: { id: FieldGroup; label: string; icon: typeof Activity }[] = [
  { id: 'structure', label: 'Structure', icon: Layers },
  { id: 'environment', label: 'Environment', icon: Waves },
  { id: 'load', label: 'Loads', icon: Route },
  { id: 'diagnostics', label: 'Diagnostics', icon: SlidersHorizontal },
]

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...init?.headers,
    },
    ...init,
  })

  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `${response.status} ${response.statusText}`)
  }

  return response.json() as Promise<T>
}

function formatNumber(value: number | null | undefined, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '--'
  }

  return value.toLocaleString(undefined, {
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

function normalizeError(error: unknown) {
  if (error instanceof Error) {
    try {
      const parsed = JSON.parse(error.message) as { detail?: string }
      return parsed.detail ?? error.message
    } catch {
      return error.message
    }
  }

  return 'Request failed'
}

function riskTone(risk: string | null | undefined) {
  const value = risk?.toLowerCase() ?? ''
  if (value.includes('critical') || value.includes('poor')) {
    return 'tone-danger'
  }
  if (value.includes('fair') || value.includes('medium') || value.includes('warn')) {
    return 'tone-warning'
  }
  return 'tone-good'
}

function App() {
  const [apiState, setApiState] = useState<ApiState>('checking')
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null)
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [historyTotal, setHistoryTotal] = useState(0)
  const [form, setForm] = useState<SensorPayload>(DEFAULT_INPUT)
  const [activeGroup, setActiveGroup] = useState<FieldGroup>('structure')
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)
  const [explanation, setExplanation] = useState<ExplainResponse | null>(null)
  const [isPredicting, setIsPredicting] = useState(false)
  const [isExplaining, setIsExplaining] = useState(false)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [message, setMessage] = useState<string | null>(null)

  // ── Navigation & Audit Search States ───────────────────────── //
  const [activeTab, setActiveTab] = useState<'drone' | 'console' | 'vision' | 'history'>('drone')
  const [historySearch, setHistorySearch] = useState('')
  const [historyFilter, setHistoryFilter] = useState<string>('all')
  const [historyPage, setHistoryPage] = useState(1)
  const itemsPerPage = 8

  // ── Vision Inspection States ────────────────────────── //
  const [visionImageId, setVisionImageId] = useState<string | null>(null)
  const [visionImageUrl, setVisionImageUrl] = useState<string | null>(null)
  const [visionFilename, setVisionFilename] = useState<string | null>(null)
  const [visionPrediction, setVisionPrediction] = useState<any | null>(null)
  const [activeOverlay, setActiveOverlay] = useState<string>('original')
  const [isUploading, setIsUploading] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isGeneratingReport, setIsGeneratingReport] = useState(false)

  const visibleFields = useMemo(
    () => SENSOR_FIELDS.filter((field) => field.group === activeGroup),
    [activeGroup],
  )

  const systemLabel = health?.model_ready ? 'Model Ready' : 'Model Standby'
  const latestHistory = history[0]
  const healthScore = prediction?.health_score ?? latestHistory?.health_score ?? 85.8
  const gaugeValue = Math.max(0, Math.min(100, healthScore ?? 0))

  async function refreshSystem() {
    setIsRefreshing(true)
    try {
      const [healthData, modelData, historyData] = await Promise.all([
        fetchJson<HealthResponse>('/health'),
        fetchJson<ModelInfoResponse>('/model-info'),
        fetchJson<HistoryResponse>('/history?limit=50&offset=0'),
      ])

      setHealth(healthData)
      setModelInfo(modelData)
      setHistory(historyData.items)
      setHistoryTotal(historyData.total)
      setApiState(healthData.status === 'healthy' ? 'online' : 'degraded')
      setMessage(null)
    } catch (error) {
      setApiState('offline')
      setMessage(normalizeError(error))
    } finally {
      setIsRefreshing(false)
    }
  }

  useEffect(() => {
    void refreshSystem()
  }, [])

  const filteredHistory = useMemo(() => {
    return history.filter((item) => {
      const matchType =
        historyFilter === 'all' ||
        (item.analysis_type || '').toLowerCase().includes(historyFilter.toLowerCase())
      const searchLower = historySearch.toLowerCase()
      const matchSearch =
        !historySearch ||
        String(item.id).includes(searchLower) ||
        (item.model_version || '').toLowerCase().includes(searchLower) ||
        (item.analysis_type || '').toLowerCase().includes(searchLower) ||
        (item.risk_category || '').toLowerCase().includes(searchLower)
      return matchType && matchSearch
    })
  }, [history, historyFilter, historySearch])

  const paginatedHistory = useMemo(() => {
    const start = (historyPage - 1) * itemsPerPage
    return filteredHistory.slice(start, start + itemsPerPage)
  }, [filteredHistory, historyPage])

  const totalPages = Math.ceil(filteredHistory.length / itemsPerPage) || 1

  function updateField(key: string, value: string, isSelect: boolean) {
    setForm((current) => ({
      ...current,
      [key]: isSelect ? value : value === '' ? null : Number(value),
    }))
  }

  async function runPrediction() {
    setIsPredicting(true)
    setMessage(null)
    try {
      const result = await fetchJson<PredictionResponse>('/predict', {
        method: 'POST',
        body: JSON.stringify(form),
      })
      setPrediction(result)
      setExplanation(null)
      await refreshSystem()
    } catch (error) {
      setMessage(normalizeError(error))
    } finally {
      setIsPredicting(false)
    }
  }

  async function explainPrediction() {
    setIsExplaining(true)
    setMessage(null)
    try {
      const result = await fetchJson<ExplainResponse>('/explain', {
        method: 'POST',
        body: JSON.stringify({
          input_data: form,
          target: 'health_score',
        }),
      })
      setExplanation(result)
    } catch (error) {
      setMessage(normalizeError(error))
    } finally {
      setIsExplaining(false)
    }
  }

  function resetSample() {
    setForm(DEFAULT_INPUT)
    setPrediction(null)
    setExplanation(null)
    setMessage(null)
  }

  function applyPreset(preset: SensorPayload) {
    setForm(preset)
    setPrediction(null)
    setExplanation(null)
    setMessage(null)
  }

  // ── Vision Inspection Handlers ────────────────────────── //
  async function handleImageUpload(e: React.ChangeEvent<HTMLInputElement>) {
    if (!e.target.files || e.target.files.length === 0) return
    const file = e.target.files[0]
    setIsUploading(true)
    setMessage(null)
    setVisionPrediction(null)
    
    try {
      const optimizedFile = await compressImage(file, 1920, 1920, 0.85)

      const formData = new FormData()
      formData.append('files', optimizedFile)

      const response = await fetch(`${API_BASE}/vision/upload-image`, {
        method: 'POST',
        body: formData,
      })
      if (!response.ok) {
        throw new Error('Upload failed')
      }
      const data = await response.json() as any[]
      if (data && data.length > 0) {
        setVisionImageId(data[0].image_id)
        setVisionImageUrl(getStaticUrl(data[0].url))
        setVisionFilename(data[0].filename)
        setActiveOverlay('original')
      }
    } catch (err: any) {
      setMessage(err.message || 'Failed to upload image')
    } finally {
      setIsUploading(false)
    }
  }

  async function runVisionPredict() {
    if (!visionImageId) return
    setIsAnalyzing(true)
    setMessage(null)
    try {
      const response = await fetch(`${API_BASE}/vision/vision-predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_id: visionImageId,
          pixel_to_mm: 0.5,
        }),
      })
      if (!response.ok) {
        throw new Error('Vision analysis failed')
      }
      const data = await response.json()
      setVisionPrediction(data)
      setActiveOverlay('segmentation')
      await refreshSystem()
    } catch (err: any) {
      setMessage(err.message || 'Vision inspection failed')
    } finally {
      setIsAnalyzing(false)
    }
  }

  async function downloadReport() {
    if (!visionImageId || !visionPrediction) return
    setIsGeneratingReport(true)
    setMessage(null)
    try {
      const response = await fetch(
        `${API_BASE}/vision/generate-report?image_id=${visionImageId}&prediction_id=${visionPrediction.prediction_id}`
      )
      if (!response.ok) {
        throw new Error('Report generation failed')
      }
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `Inspection_Report_${visionPrediction.prediction_id}.pdf`
      document.body.appendChild(a)
      a.click()
      a.remove()
      window.URL.revokeObjectURL(url)
    } catch (err: any) {
      setMessage(err.message || 'Failed to download report')
    } finally {
      setIsGeneratingReport(false)
    }
  }

  function clearImage() {
    setVisionImageId(null)
    setVisionImageUrl(null)
    setVisionFilename(null)
    setVisionPrediction(null)
    setActiveOverlay('original')
  }

  return (
    <>
      <SplashScreen />
      <main className="app-shell">
        {/* Left Navigation Sidebar */}
        <aside className="sidebar">
          <div className="brand-mark" title="BridgeGuardian AI Enterprise">
            <img src="/logo-icon.svg" alt="BridgeGuardian AI Logo" />
          </div>

          <nav className="rail" aria-label="Primary Navigation">
            <button
              type="button"
              className={`rail-button ${activeTab === 'drone' ? 'active' : ''}`}
              onClick={() => setActiveTab('drone')}
              title="Drone Campaign Inspection"
            >
              <Plane size={20} aria-hidden="true" />
            </button>

            <button
              type="button"
              className={`rail-button ${activeTab === 'console' ? 'active' : ''}`}
              onClick={() => setActiveTab('console')}
              title="Telemetry Sensor Console"
            >
              <Activity size={20} aria-hidden="true" />
            </button>

            <button
              type="button"
              className={`rail-button ${activeTab === 'vision' ? 'active' : ''}`}
              onClick={() => setActiveTab('vision')}
              title="Single Image Vision AI"
            >
              <Camera size={20} aria-hidden="true" />
            </button>

            <button
              type="button"
              className={`rail-button ${activeTab === 'history' ? 'active' : ''}`}
              onClick={() => setActiveTab('history')}
              title="Audit Trail & History"
            >
              <History size={20} aria-hidden="true" />
            </button>
          </nav>
        </aside>

        {/* Main Workspace Surface */}
        <section className="workspace">
          {/* Topbar Header */}
          <header className="topbar">
            <div className="nav-breadcrumbs">
              <span>BridgeGuardian AI</span>
              <ChevronRight size={14} />
              <span>East Span #04</span>
              <ChevronRight size={14} />
              <span className="active">
                {activeTab === 'drone'
                  ? 'Drone Inspection Campaign'
                  : activeTab === 'console'
                  ? 'Structural Telemetry Console'
                  : activeTab === 'vision'
                  ? 'Computer Vision Analysis'
                  : 'Audit Trail & History'}
              </span>
            </div>

            <div className="search-shortcut">
              <Search size={16} />
              <input
                type="text"
                placeholder="Search campaigns, models, logs..."
                value={historySearch}
                onChange={(e) => {
                  setHistorySearch(e.target.value)
                  if (activeTab !== 'history') setActiveTab('history')
                }}
              />
              <span className="kbd-badge">Ctrl K</span>
            </div>

            <div className="status-cluster">
              <span className={`status-pill ${apiState}`}>
                <span className="status-dot" />
                {apiState === 'online' ? 'System Online' : apiState}
              </span>
              <span className={`status-pill ${health?.model_ready ? 'online' : 'degraded'}`}>
                <Database size={15} aria-hidden="true" />
                {systemLabel}
              </span>
              <button
                className="btn btn-secondary"
                type="button"
                onClick={refreshSystem}
                disabled={isRefreshing}
                style={{ padding: '6px 12px', minHeight: '36px' }}
              >
                <RefreshCw size={15} className={isRefreshing ? 'spinning' : ''} aria-hidden="true" />
              </button>
            </div>
          </header>

          {/* Banner Notifications */}
          {message && (
            <div className="banner" role="status" style={{ borderRadius: '12px', marginBottom: '24px' }}>
              <Info size={18} aria-hidden="true" />
              <span>{message}</span>
            </div>
          )}

          {/* Enterprise Hero Banner */}
          <div className="enterprise-hero">
            <div className="hero-glow-backdrop" />
            <div className="hero-content-wrapper">
              <div className="hero-title-group">
                <h1>
                  <ShieldCheck size={28} style={{ color: '#0EA5E9' }} />
                  Golden Gate Bridge — East Span #04
                </h1>
                <p className="hero-subtitle">
                  <span>● Operational Status: Normal</span>
                  <span>|</span>
                  <span>18 Active Telemetry Sensors</span>
                  <span>|</span>
                  <span>YOLOv11 & SAM2 Vision AI Engine</span>
                </p>
              </div>

              <div className="hero-metrics-pill-bar">
                <div className="hero-stat-block">
                  <span className="hero-stat-label">Structural Health</span>
                  <span className="hero-stat-value" style={{ color: '#0EA5E9' }}>
                    {formatNumber(healthScore, 1)} / 100
                  </span>
                </div>
                <div style={{ width: '1px', height: '32px', background: 'rgba(255, 255, 255, 0.15)' }} />
                <div className="hero-stat-block">
                  <span className="hero-stat-label">Failure Prob (PoF)</span>
                  <span className="hero-stat-value" style={{ color: '#22C55E' }}>
                    {formatNumber(prediction?.failure_probability ?? latestHistory?.failure_probability, 2)}%
                  </span>
                </div>
                <div style={{ width: '1px', height: '32px', background: 'rgba(255, 255, 255, 0.15)' }} />
                <div className="hero-stat-block">
                  <span className="hero-stat-label">Est. RUL</span>
                  <span className="hero-stat-value">
                    {formatNumber(prediction?.rul_days ?? latestHistory?.rul_days, 0)} d
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* ────────────────── Active View Tab Switch ────────────────── */}

          {activeTab === 'drone' ? (
            <DroneInspection onCampaignComplete={refreshSystem} />
          ) : activeTab === 'console' ? (
            <>
              {/* Telemetry Quick Stat Cards */}
              <div className="stats-card-grid">
                <div className="stat-card">
                  <div className="stat-card-accent-line" />
                  <div className="stat-header">
                    <span className="stat-title">Structural Health Index</span>
                    <div className="stat-icon-box"><CircleGauge size={18} /></div>
                  </div>
                  <div className="stat-main">
                    <span className="stat-value">{formatNumber(healthScore, 1)}</span>
                    <span className="stat-trend positive">High Integrity</span>
                  </div>
                </div>

                <div className="stat-card">
                  <div className="stat-card-accent-line" style={{ background: 'linear-gradient(90deg, #22C55E 0%, #10B981 100%)' }} />
                  <div className="stat-header">
                    <span className="stat-title">Failure Probability</span>
                    <div className="stat-icon-box" style={{ color: '#22C55E', background: 'rgba(34, 197, 94, 0.12)' }}><BarChart3 size={18} /></div>
                  </div>
                  <div className="stat-main">
                    <span className="stat-value">{formatNumber(prediction?.failure_probability ?? latestHistory?.failure_probability, 2)}%</span>
                    <span className="stat-trend positive">Low Risk</span>
                  </div>
                </div>

                <div className="stat-card">
                  <div className="stat-card-accent-line" style={{ background: 'linear-gradient(90deg, #F59E0B 0%, #D97706 100%)' }} />
                  <div className="stat-header">
                    <span className="stat-title">Remaining Useful Life</span>
                    <div className="stat-icon-box" style={{ color: '#F59E0B', background: 'rgba(245, 158, 11, 0.12)' }}><History size={18} /></div>
                  </div>
                  <div className="stat-main">
                    <span className="stat-value">{formatNumber(prediction?.rul_days ?? latestHistory?.rul_days, 0)} d</span>
                    <span className="stat-trend warning">Optimal</span>
                  </div>
                </div>

                <div className="stat-card">
                  <div className="stat-card-accent-line" style={{ background: 'linear-gradient(90deg, #06B6D4 0%, #0891B2 100%)' }} />
                  <div className="stat-header">
                    <span className="stat-title">AI Model Confidence</span>
                    <div className="stat-icon-box" style={{ color: '#06B6D4', background: 'rgba(6, 182, 212, 0.12)' }}><Sparkles size={18} /></div>
                  </div>
                  <div className="stat-main">
                    <span className="stat-value">{formatNumber(prediction?.prediction_confidence ?? 0.96, 1)}%</span>
                    <span className="stat-trend positive">XGBoost & RF</span>
                  </div>
                </div>
              </div>

              {/* Sensor Controls & Output Panel */}
              <div className="content-grid">
                <section className="surface">
                  <div className="panel-heading">
                    <div>
                      <p className="eyebrow">Sensor Telemetry Frame</p>
                      <h2>Inference Parameters Input</h2>
                    </div>
                    <div className="preset-bar">
                      <button type="button" className="preset-pill" onClick={resetSample}>
                        <RefreshCw size={14} /> Reset
                      </button>
                      <button type="button" className="preset-pill" onClick={() => applyPreset(CRITICAL_PRESET)}>
                        <AlertTriangle size={14} style={{ color: '#EF4444' }} /> Severe Defect Preset
                      </button>
                    </div>
                  </div>

                  {/* Group Selector Pills */}
                  <div className="preset-bar" style={{ marginBottom: '20px' }}>
                    {FIELD_GROUPS.map((group) => {
                      const Icon = group.icon
                      return (
                        <button
                          key={group.id}
                          type="button"
                          className={`btn ${activeGroup === group.id ? 'btn-primary' : 'btn-secondary'}`}
                          style={{ padding: '6px 16px', fontSize: '0.82rem' }}
                          onClick={() => setActiveGroup(group.id)}
                        >
                          <Icon size={15} />
                          {group.label}
                        </button>
                      )
                    })}
                  </div>

                  <div className="form-grid">
                    {visibleFields.map((field) => {
                      const value = form[field.key]
                      return (
                        <div className="form-group" key={field.key}>
                          <label>
                            {field.label}
                            {field.unit && <small style={{ color: 'var(--muted)', fontSize: '0.75rem' }}>({field.unit})</small>}
                          </label>
                          {field.options ? (
                            <select
                              className="form-control"
                              value={String(value ?? '')}
                              onChange={(e) => updateField(field.key, e.target.value, true)}
                            >
                              {field.options.map((opt) => (
                                <option key={opt} value={opt}>{opt}</option>
                              ))}
                            </select>
                          ) : (
                            <input
                              type="number"
                              className="form-control"
                              step={field.step ?? 0.1}
                              value={value ?? ''}
                              onChange={(e) => updateField(field.key, e.target.value, false)}
                            />
                          )}
                        </div>
                      )
                    })}
                  </div>

                  <div style={{ marginTop: '28px', display: 'flex', justifyContent: 'flex-end' }}>
                    <button
                      type="button"
                      className="btn btn-primary btn-pulse"
                      style={{ padding: '12px 28px', fontSize: '0.95rem' }}
                      onClick={runPrediction}
                      disabled={isPredicting}
                    >
                      <Send size={18} />
                      {isPredicting ? 'Running AI Models...' : 'Run Telemetry AI Prediction'}
                    </button>
                  </div>
                </section>

                {/* Inference Outputs Sidebar */}
                <section className="surface">
                  <div className="panel-heading">
                    <div>
                      <p className="eyebrow">AI Inference Output</p>
                      <h2>Prediction Analysis</h2>
                    </div>
                    <button
                      type="button"
                      className="btn btn-secondary"
                      onClick={explainPrediction}
                      disabled={isExplaining || !prediction}
                      style={{ padding: '6px 14px', fontSize: '0.82rem' }}
                    >
                      <Sparkles size={15} />
                      {isExplaining ? 'Computing SHAP...' : 'Explain'}
                    </button>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', marginBottom: '20px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '12px 16px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-md)' }}>
                      <span style={{ fontSize: '0.88rem', color: 'var(--muted)' }}>Risk Category</span>
                      <span className={`badge ${riskTone(prediction?.risk_category ?? latestHistory?.risk_category)}`}>
                        {prediction?.risk_category ?? latestHistory?.risk_category ?? 'Awaiting Run'}
                      </span>
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '12px 16px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-md)' }}>
                      <span style={{ fontSize: '0.88rem', color: 'var(--muted)' }}>Maintenance Recommendation</span>
                      <strong style={{ fontSize: '0.88rem' }}>{prediction?.maintenance_priority ?? 'Routine Monitoring'}</strong>
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '12px 16px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-md)' }}>
                      <span style={{ fontSize: '0.88rem', color: 'var(--muted)' }}>Model Version</span>
                      <strong style={{ fontSize: '0.88rem' }}>{prediction?.model_version ?? modelInfo?.model_version ?? 'RandomForest / XGBoost'}</strong>
                    </div>
                  </div>

                  {explanation && (
                    <div style={{ marginTop: '20px', padding: '16px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-md)' }}>
                      <h3 style={{ fontSize: '0.9rem', margin: '0 0 12px', color: 'var(--ink)' }}>Top SHAP Drivers</h3>
                      {explanation.explanation.feature_importances.slice(0, 5).map((item) => (
                        <div key={item.feature} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--border-line)', fontSize: '0.82rem' }}>
                          <span>{item.feature.replaceAll('_', ' ')}</span>
                          <strong style={{ color: item.direction === 'positive' ? 'var(--danger)' : 'var(--success)' }}>
                            {formatNumber(item.shap_value, 3)}
                          </strong>
                        </div>
                      ))}
                    </div>
                  )}
                </section>
              </div>
            </>
          ) : activeTab === 'vision' ? (
            /* Single Image Vision Analysis Page */
            <div className="content-grid">
              <section className="surface">
                <div className="panel-heading">
                  <div>
                    <p className="eyebrow">Single Image Computer Vision</p>
                    <h2>Visual Inspection AI</h2>
                  </div>
                  {visionImageId && (
                    <button type="button" className="btn btn-secondary" onClick={clearImage}>
                      <X size={15} /> Clear
                    </button>
                  )}
                </div>

                {!visionImageId ? (
                  <div className="dropzone-container">
                    <input
                      type="file"
                      id="vision-file-input"
                      accept="image/*"
                      onChange={handleImageUpload}
                      disabled={isUploading}
                      style={{ display: 'none' }}
                    />
                    <label htmlFor="vision-file-input" style={{ cursor: 'pointer', display: 'block' }}>
                      <div className="dropzone-icon-ring">
                        {isUploading ? <RefreshCw size={32} className="spinning" /> : <Upload size={32} />}
                      </div>
                      <h3 style={{ margin: '0 0 8px', fontSize: '1.1rem', fontWeight: 700 }}>
                        {isUploading ? 'Uploading & Processing Image...' : 'Upload Drone Inspection Image'}
                      </h3>
                      <p style={{ margin: 0, color: 'var(--muted)', fontSize: '0.9rem' }}>
                        Drag & drop a high-res photo or click to browse (JPEG, PNG, WEBP)
                      </p>
                    </label>
                  </div>
                ) : (
                  <div>
                    <div style={{ position: 'relative', borderRadius: 'var(--radius-lg)', overflow: 'hidden', border: '1px solid var(--border-line)', minHeight: '360px', background: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      {isAnalyzing ? (
                        <div style={{ textAlign: 'center', color: '#FFF', padding: '40px' }}>
                          <RefreshCw size={40} className="spinning" style={{ color: 'var(--primary)', marginBottom: '16px' }} />
                          <h3>Running YOLOv11 & SAM2 Segmenter...</h3>
                        </div>
                      ) : (
                        <img
                          src={visionPrediction ? visionPrediction.visualizations[activeOverlay] : visionImageUrl || ''}
                          alt="Bridge Footprint"
                          style={{ maxWidth: '100%', maxHeight: '520px', objectFit: 'contain' }}
                        />
                      )}
                    </div>

                    {!visionPrediction ? (
                      <div style={{ marginTop: '20px', display: 'flex', justifyContent: 'center' }}>
                        <button
                          type="button"
                          className="btn btn-primary btn-pulse"
                          onClick={runVisionPredict}
                          disabled={isAnalyzing}
                          style={{ padding: '12px 28px' }}
                        >
                          <Sparkles size={18} />
                          {isAnalyzing ? 'Analyzing Defect Footprint...' : 'Run Vision Defect Analysis'}
                        </button>
                      </div>
                    ) : (
                      <div style={{ marginTop: '20px', display: 'flex', gap: '8px', flexWrap: 'wrap', justifyContent: 'center' }}>
                        {[
                          { key: 'original', label: 'Original Photo' },
                          { key: 'bboxes', label: 'Bounding Boxes' },
                          { key: 'segmentation', label: 'SAM2 Masks' },
                          { key: 'heatmap', label: 'Heatmap Overlay' },
                        ].map((ov) => (
                          <button
                            key={ov.key}
                            type="button"
                            className={`btn ${activeOverlay === ov.key ? 'btn-primary' : 'btn-secondary'}`}
                            onClick={() => setActiveOverlay(ov.key)}
                            style={{ padding: '6px 14px', fontSize: '0.82rem' }}
                          >
                            {ov.label}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </section>

              {/* Single Image Results Sidebar */}
              <section className="surface">
                <div className="panel-heading">
                  <div>
                    <p className="eyebrow">Vision AI Results</p>
                    <h2>Inspection Metrics</h2>
                  </div>
                </div>

                {visionPrediction ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <div style={{ padding: '16px', background: 'var(--accent-light)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(14, 165, 233, 0.3)' }}>
                      <span style={{ fontSize: '0.78rem', fontWeight: 800, color: 'var(--primary)', textTransform: 'uppercase' }}>Health Score</span>
                      <div style={{ fontSize: '2.2rem', fontWeight: 800, color: 'var(--ink)', margin: '4px 0' }}>
                        {formatNumber(visionPrediction.predictions.health_score, 1)} / 100
                      </div>
                      <span className={`badge ${riskTone(visionPrediction.predictions.risk_category)}`}>
                        {visionPrediction.predictions.risk_category}
                      </span>
                    </div>

                    <button
                      type="button"
                      className="btn btn-primary"
                      onClick={downloadReport}
                      disabled={isGeneratingReport}
                      style={{ width: '100%' }}
                    >
                      <FileDown size={18} />
                      {isGeneratingReport ? 'Generating Report...' : 'Download PDF Inspection Report'}
                    </button>

                    <div style={{ padding: '16px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-md)' }}>
                      <h4 style={{ margin: '0 0 12px', fontSize: '0.9rem' }}>CV Extracted Defect Features</h4>
                      {[
                        { label: 'Crack Density', val: `${visionPrediction.features.crack_density}%` },
                        { label: 'Max Crack Width', val: `${visionPrediction.features.crack_width} mm` },
                        { label: 'Corrosion Area', val: `${visionPrediction.features.corrosion_percent}%` },
                        { label: 'Concrete Spalling', val: `${visionPrediction.features.spalling_percent}%` },
                      ].map((row) => (
                        <div key={row.label} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--border-line)', fontSize: '0.85rem' }}>
                          <span style={{ color: 'var(--muted)' }}>{row.label}</span>
                          <strong>{row.val}</strong>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div style={{ textAlign: 'center', padding: '40px 20px', color: 'var(--muted)' }}>
                    <ImageIcon size={48} style={{ opacity: 0.4, marginBottom: '12px' }} />
                    <p style={{ margin: 0, fontSize: '0.9rem' }}>Upload an inspection photo to analyze defect bounding boxes and health metrics.</p>
                  </div>
                )}
              </section>
            </div>
          ) : (
            /* Audit Trail & History Page */
            <section className="surface">
              <div className="panel-heading" style={{ flexWrap: 'wrap', gap: '16px' }}>
                <div>
                  <p className="eyebrow">Enterprise Audit Trail</p>
                  <h2>Prediction & Inspection History</h2>
                </div>

                {/* Workflow Filter Chips */}
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                  {[
                    { id: 'all', label: 'All Workflows' },
                    { id: 'Drone Campaign', label: 'Drone Campaign' },
                    { id: 'Single Image', label: 'Single Image' },
                    { id: 'Structural Health', label: 'Structural Health' },
                  ].map((chip) => (
                    <button
                      key={chip.id}
                      type="button"
                      className={`btn ${historyFilter === chip.id ? 'btn-primary' : 'btn-secondary'}`}
                      style={{ padding: '6px 14px', fontSize: '0.8rem' }}
                      onClick={() => {
                        setHistoryFilter(chip.id)
                        setHistoryPage(1)
                      }}
                    >
                      {chip.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Data Table */}
              <div style={{ overflowX: 'auto' }}>
                <div className="history-table">
                  <div className="history-row header">
                    <span>Timestamp</span>
                    <span>Workflow Type</span>
                    <span>Health (SHI)</span>
                    <span>Failure Prob</span>
                    <span>Priority</span>
                  </div>
                  {paginatedHistory.length === 0 ? (
                    <div style={{ padding: '40px', textAlign: 'center', color: 'var(--muted)' }}>
                      No history records found matching your filters.
                    </div>
                  ) : (
                    paginatedHistory.map((item) => (
                      <div className="history-row" key={item.id}>
                        <span>{compactDate(item.created_at)}</span>
                        <div>
                          <span className="workflow-tag" title={item.model_version || ''}>
                            {item.analysis_type ?? 'Structural Health'}
                          </span>
                        </div>
                        <strong style={{ fontSize: '0.95rem' }}>{formatNumber(item.health_score, 1)}</strong>
                        <span style={{ fontSize: '0.9rem', color: 'var(--ink-subtle)' }}>
                          {formatNumber(item.failure_probability, 2)}%
                        </span>
                        <div>
                          <span className={`badge ${riskTone(item.risk_category)}`}>
                            {item.maintenance_priority ?? 'Low'}
                          </span>
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
                    disabled={historyPage <= 1}
                    onClick={() => setHistoryPage((p) => Math.max(1, p - 1))}
                    style={{ padding: '6px 12px' }}
                  >
                    <ChevronLeft size={16} /> Prev
                  </button>
                  <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>
                    Page {historyPage} of {totalPages}
                  </span>
                  <button
                    type="button"
                    className="btn btn-secondary"
                    disabled={historyPage >= totalPages}
                    onClick={() => setHistoryPage((p) => Math.min(totalPages, p + 1))}
                    style={{ padding: '6px 12px' }}
                  >
                    Next <ChevronRight size={16} />
                  </button>
                </div>
              </div>
            </section>
          )}

          {/* Model Inventory Footer Panel */}
          <section className="surface" style={{ marginTop: '28px' }}>
            <div className="panel-heading">
              <div>
                <p className="eyebrow">AI Runtime Inventory</p>
                <h2>Model Architecture & Status</h2>
              </div>
              <span className="badge badge-minor">
                <CheckCircle2 size={14} /> Production Ready
              </span>
            </div>

            <div className="form-grid">
              <div style={{ padding: '16px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-md)' }}>
                <span style={{ fontSize: '0.78rem', color: 'var(--muted)', textTransform: 'uppercase', fontWeight: 700 }}>API Engine Version</span>
                <div style={{ fontSize: '1.2rem', fontWeight: 800, marginTop: '4px' }}>{health?.version ?? 'v1.0.0'}</div>
              </div>

              <div style={{ padding: '16px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-md)' }}>
                <span style={{ fontSize: '0.78rem', color: 'var(--muted)', textTransform: 'uppercase', fontWeight: 700 }}>Database Connection</span>
                <div style={{ fontSize: '1.2rem', fontWeight: 800, marginTop: '4px', color: health?.database_ok ? 'var(--success)' : 'var(--danger)' }}>
                  {health?.database_ok ? 'Connected' : 'Offline'}
                </div>
              </div>

              <div style={{ padding: '16px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-md)' }}>
                <span style={{ fontSize: '0.78rem', color: 'var(--muted)', textTransform: 'uppercase', fontWeight: 700 }}>Telemetry Feature Count</span>
                <div style={{ fontSize: '1.2rem', fontWeight: 800, marginTop: '4px' }}>{modelInfo?.feature_count ?? 42} Features</div>
              </div>

              <div style={{ padding: '16px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-md)' }}>
                <span style={{ fontSize: '0.78rem', color: 'var(--muted)', textTransform: 'uppercase', fontWeight: 700 }}>Loaded Vision AI Models</span>
                <div style={{ fontSize: '1.2rem', fontWeight: 800, marginTop: '4px' }}>YOLOv11 & SAM2</div>
              </div>
            </div>
          </section>
        </section>
      </main>
    </>
  )
}

export default App
