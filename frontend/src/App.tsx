import { useEffect, useMemo, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Bell,
  Camera,
  CircleGauge,
  Database,
  FileDown,
  History,
  Image,
  Info,
  Layers,
  RefreshCw,
  Route,
  Send,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Upload,
  Waves,
  X,
} from 'lucide-react'
import './App.css'
import DroneInspection from './components/DroneInspection'
import { API_BASE, getStaticUrl } from './lib/api'

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
  model_version: string | null
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
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(value))
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
  if (value.includes('fair') || value.includes('medium')) {
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

  // ── Vision Inspection States ────────────────────────────────────────── //
  const [activeTab, setActiveTab] = useState<'console' | 'vision' | 'drone'>('drone')
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

  const systemLabel = health?.model_ready ? 'Model ready' : 'Model not ready'
  const latestHistory = history[0]
  const healthScore = prediction?.health_score ?? latestHistory?.health_score ?? null
  const gaugeValue = Math.max(0, Math.min(100, healthScore ?? 0))

  async function refreshSystem() {
    setIsRefreshing(true)
    try {
      const [healthData, modelData, historyData] = await Promise.all([
        fetchJson<HealthResponse>('/health'),
        fetchJson<ModelInfoResponse>('/model-info'),
        fetchJson<HistoryResponse>('/history?limit=6&offset=0'),
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

  // ── Vision Inspection Event Handlers ────────────────────────────────── //
  async function handleImageUpload(e: React.ChangeEvent<HTMLInputElement>) {
    if (!e.target.files || e.target.files.length === 0) return
    const file = e.target.files[0]
    setIsUploading(true)
    setMessage(null)
    setVisionPrediction(null)
    
    const formData = new FormData()
    formData.append('files', file)
    
    try {
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
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand-mark">
          <ShieldCheck size={28} aria-hidden="true" />
        </div>
        <nav className="rail" aria-label="Primary">
          <button
            type="button"
            className={activeTab === 'console' ? 'active' : ''}
            onClick={() => setActiveTab('console')}
            title="Sensor Console"
          >
            <CircleGauge size={20} aria-hidden="true" />
          </button>
          <button
            type="button"
            className={activeTab === 'vision' ? 'active' : ''}
            onClick={() => setActiveTab('vision')}
            title="Vision Inspection"
          >
            <Camera size={20} aria-hidden="true" />
          </button>
          <button
            type="button"
            className={activeTab === 'drone' ? 'active' : ''}
            onClick={() => setActiveTab('drone')}
            title="Drone Inspection"
          >
            <Layers size={20} aria-hidden="true" />
          </button>
        </nav>
      </aside>

      <section className="workspace">
        {activeTab === 'console' ? (
          <>
            <header className="topbar">
              <div>
                <p className="eyebrow">BridgeGuardian AI</p>
                <h1>Structural health console</h1>
              </div>
              <div className="status-cluster">
                <span className={`status-pill ${apiState}`}>
                  <Activity size={15} aria-hidden="true" />
                  {apiState}
                </span>
                <span className={`status-pill ${health?.model_ready ? 'online' : 'degraded'}`}>
                  <Database size={15} aria-hidden="true" />
                  {systemLabel}
                </span>
                <button className="icon-button" type="button" onClick={refreshSystem} disabled={isRefreshing}>
                  <RefreshCw size={18} aria-hidden="true" />
                  <span className="sr-only">Refresh status</span>
                </button>
              </div>
            </header>

            {message && (
              <div className="banner" role="status">
                <Info size={18} aria-hidden="true" />
                <span>{message}</span>
              </div>
            )}

            <section className="dashboard-grid" id="overview">
              <div className="surface bridge-panel">
                <div className="panel-heading">
                  <div>
                    <p className="eyebrow">Live span</p>
                    <h2>North approach bridge</h2>
                  </div>
                  <span className={`risk-badge ${riskTone(prediction?.risk_category ?? latestHistory?.risk_category)}`}>
                    {prediction?.risk_category ?? latestHistory?.risk_category ?? 'Awaiting run'}
                  </span>
                </div>

                <div className="bridge-visual" aria-hidden="true">
                  <div className="bridge-deck">
                    <span />
                    <span />
                    <span />
                    <span />
                  </div>
                  <div className="bridge-arch" />
                  <div className="sensor-dot dot-a" />
                  <div className="sensor-dot dot-b" />
                  <div className="sensor-dot dot-c" />
                </div>

                <div className="telemetry-strip">
                  <span>Strain {formatNumber(Number(form.Strain_microstrain), 1)}</span>
                  <span>Vibration {formatNumber(Number(form.Vibration_ms2), 2)}</span>
                  <span>Traffic {formatNumber(Number(form.Traffic_Volume_vph), 0)}</span>
                </div>
              </div>

              <div className="surface score-panel">
                <div className="score-ring" style={{ '--score': `${gaugeValue}%` } as React.CSSProperties}>
                  <div>
                    <strong>{formatNumber(healthScore, 1)}</strong>
                    <span>SHI</span>
                  </div>
                </div>
                <div className="score-copy">
                  <p className="eyebrow">Current assessment</p>
                  <h2>{prediction?.maintenance_priority ?? latestHistory?.maintenance_priority ?? 'No prediction'}</h2>
                  <p>{prediction?.maintenance_recommendation ?? 'Prediction output will appear after inference.'}</p>
                </div>
              </div>

              <div className="metrics-row">
                <article className="metric-tile">
                  <BarChart3 size={19} aria-hidden="true" />
                  <span>Failure probability</span>
                  <strong>{formatNumber(prediction?.failure_probability ?? latestHistory?.failure_probability, 2)}%</strong>
                </article>
                <article className="metric-tile">
                  <History size={19} aria-hidden="true" />
                  <span>Remaining life</span>
                  <strong>{formatNumber(prediction?.rul_days ?? latestHistory?.rul_days, 0)} d</strong>
                </article>
                <article className="metric-tile">
                  <Sparkles size={19} aria-hidden="true" />
                  <span>Confidence</span>
                  <strong>{formatNumber(prediction?.prediction_confidence, 1)}%</strong>
                </article>
              </div>
            </section>

            <section className="content-grid">
              <section className="surface sensor-panel" id="sensors">
                <div className="panel-heading">
                  <div>
                    <p className="eyebrow">Sensor frame</p>
                    <h2>Inference input</h2>
                  </div>
                  <div className="button-group">
                    <button type="button" className="secondary-button" onClick={resetSample}>
                      <RefreshCw size={16} aria-hidden="true" />
                      Reset
                    </button>
                    <button type="button" className="primary-button" onClick={runPrediction} disabled={isPredicting}>
                      <Send size={16} aria-hidden="true" />
                      {isPredicting ? 'Running' : 'Predict'}
                    </button>
                  </div>
                </div>

                <div className="segmented-control">
                  {FIELD_GROUPS.map((group) => {
                    const Icon = group.icon
                    return (
                      <button
                        key={group.id}
                        type="button"
                        className={activeGroup === group.id ? 'active' : ''}
                        onClick={() => setActiveGroup(group.id)}
                      >
                        <Icon size={16} aria-hidden="true" />
                        {group.label}
                      </button>
                    )
                  })}
                </div>

                <div className="sensor-grid">
                  {visibleFields.map((field) => {
                    const value = form[field.key]
                    return (
                      <label className="field-control" key={field.key}>
                        <span>
                          {field.label}
                          {field.unit && <small>{field.unit}</small>}
                        </span>
                        {field.options ? (
                          <select
                            value={String(value ?? '')}
                            onChange={(event) => updateField(field.key, event.target.value, true)}
                          >
                            {field.options.map((option) => (
                              <option value={option} key={option}>
                                {option}
                              </option>
                            ))}
                          </select>
                        ) : (
                          <input
                            type="number"
                            inputMode="decimal"
                            step={field.step ?? 0.1}
                            value={value ?? ''}
                            onChange={(event) => updateField(field.key, event.target.value, false)}
                          />
                        )}
                      </label>
                    )
                  })}
                </div>
              </section>

              <section className="surface result-panel">
                <div className="panel-heading">
                  <div>
                    <p className="eyebrow">Model output</p>
                    <h2>Prediction detail</h2>
                  </div>
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={explainPrediction}
                    disabled={isExplaining || !prediction}
                  >
                    <Sparkles size={16} aria-hidden="true" />
                    {isExplaining ? 'Explaining' : 'Explain'}
                  </button>
                </div>

                <div className="detail-stack">
                  <div className="detail-row">
                    <span>Risk category</span>
                    <strong className={riskTone(prediction?.risk_category)}>{prediction?.risk_category ?? '--'}</strong>
                  </div>
                  <div className="detail-row">
                    <span>Maintenance alert</span>
                    <strong>{prediction?.maintenance_alert ? 'Active' : prediction ? 'Clear' : '--'}</strong>
                  </div>
                  <div className="detail-row">
                    <span>RUL confidence</span>
                    <strong>{prediction?.rul_confidence ?? '--'}</strong>
                  </div>
                  <div className="detail-row">
                    <span>Model version</span>
                    <strong>{prediction?.model_version ?? modelInfo?.model_version ?? '--'}</strong>
                  </div>
                </div>

                <div className={`alert-copy ${prediction?.maintenance_alert ? 'active' : ''}`}>
                  <AlertTriangle size={18} aria-hidden="true" />
                  <span>{prediction?.rul_message ?? 'No active RUL message.'}</span>
                </div>

                {explanation && (
                  <div className="explain-box">
                    <div className="explain-head">
                      <span>Top SHAP drivers</span>
                      <strong>{formatNumber(explanation.explanation.prediction_contribution, 3)}</strong>
                    </div>
                    <div className="importance-list">
                      {explanation.explanation.feature_importances.slice(0, 5).map((item) => (
                        <div className="importance-item" key={`${item.feature}-${item.shap_value}`}>
                          <span>{item.feature.replaceAll('_', ' ')}</span>
                          <strong className={item.direction === 'positive' ? 'tone-danger' : 'tone-good'}>
                            {formatNumber(item.shap_value, 3)}
                          </strong>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </section>
            </section>

            <section className="content-grid bottom-grid" id="history">
              <section className="surface">
                <div className="panel-heading">
                  <div>
                    <p className="eyebrow">Audit trail</p>
                    <h2>Recent predictions</h2>
                  </div>
                  <span className="subtle-count">{historyTotal} total</span>
                </div>

                <div className="history-table">
                  <div className="history-row header">
                    <span>Time</span>
                    <span>SHI</span>
                    <span>PoF</span>
                    <span>Priority</span>
                  </div>
                  {history.length === 0 ? (
                    <div className="empty-state">No prediction history</div>
                  ) : (
                    history.map((item) => (
                      <div className="history-row" key={item.id}>
                        <span>{compactDate(item.created_at)}</span>
                        <strong>{formatNumber(item.health_score, 1)}</strong>
                        <span>{formatNumber(item.failure_probability, 2)}%</span>
                        <span className={riskTone(item.risk_category)}>{item.maintenance_priority ?? '--'}</span>
                      </div>
                    ))
                  )}
                </div>
              </section>

              <section className="surface model-panel">
                <div className="panel-heading">
                  <div>
                    <p className="eyebrow">Runtime</p>
                    <h2>Model inventory</h2>
                  </div>
                  <Bell size={18} aria-hidden="true" />
                </div>

                <div className="model-grid">
                  <div>
                    <span>API version</span>
                    <strong>{health?.version ?? '--'}</strong>
                  </div>
                  <div>
                    <span>Database</span>
                    <strong>{health?.database_ok ? 'Connected' : '--'}</strong>
                  </div>
                  <div>
                    <span>Features</span>
                    <strong>{modelInfo?.feature_count ?? '--'}</strong>
                  </div>
                  <div>
                    <span>Targets</span>
                    <strong>{modelInfo?.models_available.length ?? '--'}</strong>
                  </div>
                </div>

                <div className="model-list">
                  {(modelInfo?.models_available ?? []).map((model) => (
                    <span key={model}>{model.replaceAll('_', ' ')}</span>
                  ))}
                  {modelInfo?.models_available.length === 0 && <span>No loaded models</span>}
                </div>
              </section>
            </section>
          </>
        ) : activeTab === 'drone' ? (
          <>
            <header className="topbar">
              <div>
                <p className="eyebrow">BridgeGuardian AI</p>
                <h1>Drone Campaign Inspection</h1>
              </div>
              <div className="status-cluster">
                <span className={`status-pill ${apiState}`}>
                  <Activity size={15} aria-hidden="true" />
                  {apiState}
                </span>
                <span className={`status-pill ${health?.model_ready ? 'online' : 'degraded'}`}>
                  <Database size={15} aria-hidden="true" />
                  {systemLabel}
                </span>
                <button className="icon-button" type="button" onClick={refreshSystem} disabled={isRefreshing}>
                  <RefreshCw size={18} aria-hidden="true" />
                  <span className="sr-only">Refresh status</span>
                </button>
              </div>
            </header>
            <DroneInspection />
          </>
        ) : (
          <>
            <header className="topbar">
              <div>
                <p className="eyebrow">BridgeGuardian AI</p>
                <h1>Computer vision inspection</h1>
              </div>
              <div className="status-cluster">
                <span className={`status-pill ${apiState}`}>
                  <Activity size={15} aria-hidden="true" />
                  {apiState}
                </span>
                <span className={`status-pill ${health?.model_ready ? 'online' : 'degraded'}`}>
                  <Database size={15} aria-hidden="true" />
                  {systemLabel}
                </span>
                <button className="icon-button" type="button" onClick={refreshSystem} disabled={isRefreshing}>
                  <RefreshCw size={18} aria-hidden="true" />
                  <span className="sr-only">Refresh status</span>
                </button>
              </div>
            </header>

            {message && (
              <div className="banner" role="status">
                <Info size={18} aria-hidden="true" />
                <span>{message}</span>
              </div>
            )}

            <div className="vision-dashboard-container">
              {/* Image upload / visualizer panel */}
              <div className="surface vision-viewer-panel">
                <div className="panel-heading">
                  <div>
                    <p className="eyebrow">Drone Footage Analysis {visionFilename && ` - ${visionFilename}`}</p>
                    <h2>Visualizer</h2>
                  </div>
                  {visionImageId && (
                    <button type="button" className="secondary-button icon-button-text" onClick={clearImage}>
                      <X size={16} aria-hidden="true" />
                      Clear
                    </button>
                  )}
                </div>

                {!visionImageId ? (
                  <div className="upload-dropzone">
                    <input
                      type="file"
                      id="drone-file"
                      accept="image/*"
                      onChange={handleImageUpload}
                      disabled={isUploading}
                      style={{ display: 'none' }}
                    />
                    <label htmlFor="drone-file" className="dropzone-label">
                      <div className="dropzone-inner">
                        {isUploading ? (
                          <RefreshCw size={38} className="upload-icon spinner-icon spinning" />
                        ) : (
                          <Upload size={38} className="upload-icon" />
                        )}
                        <h3>{isUploading ? 'Uploading Image...' : 'Upload Bridge Image'}</h3>
                        <p>Drag and drop drone inspection photo, or click to browse</p>
                        <small>Supports JPEG, PNG, WEBP</small>
                      </div>
                    </label>
                  </div>
                ) : (
                  <div className="visualizer-content">
                    <div className="image-display-frame">
                      {isAnalyzing ? (
                        <div className="analysis-spinner">
                          <RefreshCw size={36} className="spinner-icon spinning" />
                          <h3>Analyzing Structural Details...</h3>
                          <p>Running object detection and morphological defect segmentation...</p>
                        </div>
                      ) : (
                        <img
                          src={visionPrediction ? visionPrediction.visualizations[activeOverlay] : visionImageUrl || ''}
                          alt="Bridge Footprint"
                          className="vision-display-img"
                        />
                      )}
                    </div>

                    {!visionPrediction ? (
                      <div className="analysis-trigger-bar">
                        <button
                          type="button"
                          className="primary-button run-analysis-btn"
                          onClick={runVisionPredict}
                          disabled={isAnalyzing}
                        >
                          <Sparkles size={16} aria-hidden="true" />
                          {isAnalyzing ? 'Analyzing Image...' : 'Run Defect Detection'}
                        </button>
                      </div>
                    ) : (
                      <div className="overlay-control-bar">
                        {[
                          { key: 'original', label: 'Original' },
                          { key: 'cracks', label: 'Cracks' },
                          { key: 'rust', label: 'Rust' },
                          { key: 'bboxes', label: 'Bounding Boxes' },
                          { key: 'heatmap', label: 'Heatmap' },
                          { key: 'segmentation', label: 'Segmentation Overlay' },
                        ].map((overlay) => (
                          <button
                            key={overlay.key}
                            type="button"
                            className={`overlay-tab ${activeOverlay === overlay.key ? 'active' : ''}`}
                            onClick={() => setActiveOverlay(overlay.key)}
                          >
                            {overlay.label}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Stats / Predictions side panel */}
              <div className="vision-results-sidebar">
                {visionPrediction ? (
                  <>
                    {/* Score Summary Card */}
                    <div className="surface vision-score-card">
                      <div className="vision-health-circle" style={{ '--score': `${visionPrediction.predictions.health_score}%` } as React.CSSProperties}>
                        <div>
                          <strong>{formatNumber(visionPrediction.predictions.health_score, 1)}</strong>
                          <span>SHI</span>
                        </div>
                      </div>
                      <div className="score-summary-details">
                        <p className="eyebrow">Risk Classification</p>
                        <h2 className={`risk-category-value ${riskTone(visionPrediction.predictions.risk_category)}`}>
                          {visionPrediction.predictions.risk_category}
                        </h2>
                        <div className="mini-metrics-row">
                          <div>
                            <span>PoF</span>
                            <strong>{formatNumber(visionPrediction.predictions.failure_probability, 2)}%</strong>
                          </div>
                          <div>
                            <span>RUL</span>
                            <strong>{formatNumber(visionPrediction.predictions.rul_days, 0)} d</strong>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* PDF Exporter Bar */}
                    <div className="action-bar-export">
                      <button
                        type="button"
                        className="primary-button pdf-download-btn"
                        onClick={downloadReport}
                        disabled={isGeneratingReport}
                        style={{ width: '100%', justifyContent: 'center' }}
                      >
                        <FileDown size={16} aria-hidden="true" />
                        {isGeneratingReport ? 'Generating Report...' : 'Download PDF Report'}
                      </button>
                    </div>

                    {/* Recommendations Warning Card */}
                    <div className="surface vision-recommendation-card">
                      <div className="card-heading-warning">
                        <AlertTriangle size={18} aria-hidden="true" />
                        <span>Recommendation Priority: <b>{visionPrediction.predictions.maintenance_priority}</b></span>
                      </div>
                      <p>{visionPrediction.predictions.maintenance_recommendation}</p>
                    </div>

                    {/* Extracted Features List */}
                    <div className="surface vision-features-list">
                      <div className="panel-heading">
                        <h2>Extracted CV Parameters</h2>
                      </div>
                      <div className="cv-features-table">
                        {[
                          { label: 'Crack Density', val: `${visionPrediction.features.crack_density}%` },
                          { label: 'Estimated Crack Length', val: `${visionPrediction.features.crack_length} mm` },
                          { label: 'Estimated Crack Width', val: `${visionPrediction.features.crack_width} mm` },
                          { label: 'Corrosion/Rust Area', val: `${visionPrediction.features.corrosion_percent}%` },
                          { label: 'Concrete Spalling', val: `${visionPrediction.features.spalling_percent}%` },
                          { label: 'Vegetation Area', val: `${visionPrediction.features.vegetation_percent}%` },
                          { label: 'Water Leakage Stains', val: `${visionPrediction.features.leakage_percent}%` },
                          { label: 'Bridge Tilt Angle', val: `${visionPrediction.features.tilt_angle}°` },
                          { label: 'Missing/Loose Components', val: `${visionPrediction.features.missing_components}` },
                          { label: 'Total Damage Area', val: `${visionPrediction.features.damage_area_percent}%` },
                        ].map((row) => (
                          <div className="cv-feature-row" key={row.label}>
                            <span>{row.label}</span>
                            <strong>{row.val}</strong>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* SHAP Drivers */}
                    {visionPrediction.shap?.explanation?.feature_importances && (
                      <div className="surface vision-shap-card">
                        <div className="panel-heading">
                          <h2>SHAP Impact Drivers</h2>
                        </div>
                        <div className="importance-list">
                          {visionPrediction.shap.explanation.feature_importances.slice(0, 5).map((item: any) => (
                            <div className="importance-item" key={`${item.feature}-${item.shap_value}`}>
                              <span>{item.feature.replaceAll('_', ' ')}</span>
                              <strong className={item.direction === 'positive' ? 'tone-danger' : 'tone-good'}>
                                {formatNumber(item.shap_value, 3)}
                              </strong>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="surface vision-placeholder-sidebar">
                    <Image size={40} className="placeholder-icon" style={{ opacity: 0.5, marginBottom: '8px' }} />
                    <h3>No Analysis Results</h3>
                    <p>Upload a drone photo and click "Run Defect Detection" to view structural health predictions, recommendations, and measurements.</p>
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </section>
    </main>
  )
}

export default App
