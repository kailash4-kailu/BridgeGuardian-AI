import React, { useEffect, useState } from 'react'
import './App.css'
import SplashScreen from './components/SplashScreen'
import Sidebar from './components/layout/Sidebar'
import type { TabType } from './components/layout/Sidebar'
import Navbar from './components/layout/Navbar'
import HeroHeader from './components/layout/HeroHeader'
import TelemetryConsole from './components/telemetry/TelemetryConsole'
import SingleVisionConsole from './components/vision/SingleVisionConsole'
import ModelInventory from './components/models/ModelInventory'
import DroneInspection from './components/DroneInspection'

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

type ExplainResponse = {
  target: string
  explanation: {
    base_value: number
    feature_importances: any[]
    prediction_contribution: number
  }
}

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
  SHI_Predicted_7d_Ahead: 0.35,
  SHI_Predicted_30d_Ahead: 0.20,
}

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

function App() {
  const [apiState, setApiState] = useState<ApiState>('checking')
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null)
  const [form, setForm] = useState<SensorPayload>(DEFAULT_INPUT)
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)
  const [explanation, setExplanation] = useState<ExplainResponse | null>(null)
  const [isPredicting, setIsPredicting] = useState(false)
  const [isExplaining, setIsExplaining] = useState(false)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [message, setMessage] = useState<string | null>(null)

  // Navigation Tabs: Drone, Console, Vision
  const [activeTab, setActiveTab] = useState<TabType>('drone')

  // Vision Inspection States
  const [visionImageId, setVisionImageId] = useState<string | null>(null)
  const [visionImageUrl, setVisionImageUrl] = useState<string | null>(null)
  const [visionFilename, setVisionFilename] = useState<string | null>(null)
  const [visionPrediction, setVisionPrediction] = useState<any | null>(null)
  const [activeOverlay, setActiveOverlay] = useState<string>('original')
  const [isUploading, setIsUploading] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isGeneratingReport, setIsGeneratingReport] = useState(false)

  const healthScore = prediction?.health_score ?? 85.8

  async function refreshSystem() {
    setIsRefreshing(true)
    try {
      const [healthData, modelData] = await Promise.all([
        fetchJson<HealthResponse>('/health'),
        fetchJson<ModelInfoResponse>('/model-info'),
      ])

      setHealth(healthData)
      setModelInfo(modelData)
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

  function applyPreset(preset: SensorPayload) {
    setForm(preset)
    setPrediction(null)
    setExplanation(null)
    setMessage(null)
  }

  // Vision Handlers
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
      if (!response.ok) throw new Error('Upload failed')
      
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
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_id: visionImageId, pixel_to_mm: 0.5 }),
      })
      if (!response.ok) throw new Error('Vision analysis failed')
      
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
      if (!response.ok) throw new Error('Report generation failed')
      
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
        <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />

        <section className="workspace">
          <Navbar
            activeTab={activeTab}
            health={health}
            apiState={apiState}
            onRefresh={refreshSystem}
            isRefreshing={isRefreshing}
          />

          {message && (
            <div className="banner" role="status" style={{ borderRadius: '12px', marginBottom: '24px' }}>
              <span>{message}</span>
            </div>
          )}

          <HeroHeader
            healthScore={healthScore}
            failureProbability={prediction?.failure_probability ?? 2.42}
            rulDays={prediction?.rul_days ?? 182.7}
            riskCategory={prediction?.risk_category ?? 'Good'}
            dbConnected={health?.database_ok}
            featureCount={modelInfo?.feature_count}
          />

          {activeTab === 'drone' ? (
            <DroneInspection onCampaignComplete={refreshSystem} />
          ) : activeTab === 'console' ? (
            <TelemetryConsole
              form={form}
              onFieldChange={updateField}
              onRunPrediction={runPrediction}
              onExplainPrediction={explainPrediction}
              onReset={resetSample}
              onApplyPreset={applyPreset}
              prediction={prediction}
              explanation={explanation}
              isPredicting={isPredicting}
              isExplaining={isExplaining}
              latestHistory={null}
              DEFAULT_INPUT={DEFAULT_INPUT}
              CRITICAL_PRESET={CRITICAL_PRESET}
            />
          ) : (
            <SingleVisionConsole
              visionImageId={visionImageId}
              visionImageUrl={visionImageUrl}
              visionFilename={visionFilename}
              visionPrediction={visionPrediction}
              activeOverlay={activeOverlay}
              setActiveOverlay={setActiveOverlay}
              isUploading={isUploading}
              isAnalyzing={isAnalyzing}
              isGeneratingReport={isGeneratingReport}
              onImageUpload={handleImageUpload}
              onRunVisionPredict={runVisionPredict}
              onDownloadReport={downloadReport}
              onClearImage={clearImage}
            />
          )}

          <ModelInventory health={health} modelInfo={modelInfo} />
        </section>
      </main>
    </>
  )
}

export default App
