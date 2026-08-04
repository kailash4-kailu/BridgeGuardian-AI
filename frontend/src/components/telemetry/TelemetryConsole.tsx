import React, { useState } from 'react'
import {
  CircleGauge,
  BarChart3,
  History as HistoryIcon,
  Sparkles,
  RefreshCw,
  AlertTriangle,
  Send,
  Layers,
  Waves,
  Route,
  SlidersHorizontal,
} from 'lucide-react'
import MetricCard from '../ui/MetricCard'
import StatusBadge from '../ui/StatusBadge'

type SensorPayload = Record<string, number | string | null>

type FieldGroup = 'structure' | 'environment' | 'load' | 'diagnostics'

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
    prediction_contribution: number
  }
}

interface TelemetryConsoleProps {
  form: SensorPayload
  onFieldChange: (key: string, value: string, isSelect: boolean) => void
  onRunPrediction: () => void
  onExplainPrediction: () => void
  onReset: () => void
  onApplyPreset: (preset: SensorPayload) => void
  prediction: PredictionResponse | null
  explanation: ExplainResponse | null
  isPredicting: boolean
  isExplaining: boolean
  latestHistory: any
  DEFAULT_INPUT: SensorPayload
  CRITICAL_PRESET: SensorPayload
}

const FIELD_GROUPS: { id: FieldGroup; label: string; icon: any }[] = [
  { id: 'structure', label: 'Structure', icon: Layers },
  { id: 'environment', label: 'Environment', icon: Waves },
  { id: 'load', label: 'Loads', icon: Route },
  { id: 'diagnostics', label: 'Diagnostics', icon: SlidersHorizontal },
]

const SENSOR_FIELDS = [
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

function formatNumber(val: number | null | undefined, digits = 1) {
  if (val === null || val === undefined || Number.isNaN(val)) return '--'
  return val.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  })
}

export const TelemetryConsole: React.FC<TelemetryConsoleProps> = ({
  form,
  onFieldChange,
  onRunPrediction,
  onExplainPrediction,
  onReset,
  onApplyPreset,
  prediction,
  explanation,
  isPredicting,
  isExplaining,
  latestHistory,
  CRITICAL_PRESET,
}) => {
  const [activeGroup, setActiveGroup] = useState<FieldGroup>('structure')

  const visibleFields = SENSOR_FIELDS.filter((f) => f.group === activeGroup)
  const healthScore = prediction?.health_score ?? latestHistory?.health_score ?? 85.8
  const pofValue = prediction?.failure_probability ?? latestHistory?.failure_probability ?? 2.42
  const rulValue = prediction?.rul_days ?? latestHistory?.rul_days ?? 182.7

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Primary KPI Metrics Grid */}
      <div className="stats-card-grid">
        <MetricCard
          title="Structural Health Index"
          value={formatNumber(healthScore, 1)}
          unit="/ 100"
          trendLabel="Optimal Integrity"
          trendTone="good"
          icon={CircleGauge}
        />
        <MetricCard
          title="Failure Probability"
          value={formatNumber(pofValue, 2)}
          unit="%"
          trendLabel="Low Risk"
          trendTone="good"
          icon={BarChart3}
        />
        <MetricCard
          title="Remaining Useful Life"
          value={formatNumber(rulValue, 0)}
          unit="Days"
          trendLabel="Target Life"
          trendTone="warning"
          icon={HistoryIcon}
        />
        <MetricCard
          title="Prediction Confidence"
          value={formatNumber(
            (prediction?.prediction_confidence ?? 0.96) <= 1
              ? (prediction?.prediction_confidence ?? 0.96) * 100
              : prediction?.prediction_confidence,
            1
          )}
          unit="%"
          trendLabel="RF / XGBoost Engine"
          trendTone="good"
          icon={Sparkles}
        />
      </div>

      {/* Main Form & Results Grid */}
      <div className="content-grid">
        {/* Form Inputs Surface */}
        <section className="surface">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Sensor Telemetry Input</p>
              <h2>Inference Parameters Frame</h2>
            </div>
            <div className="preset-bar">
              <button type="button" className="preset-pill" onClick={onReset}>
                <RefreshCw size={13} /> Reset Baseline
              </button>
              <button type="button" className="preset-pill" onClick={() => onApplyPreset(CRITICAL_PRESET)}>
                <AlertTriangle size={13} style={{ color: 'var(--danger)' }} /> Severe Damage Preset
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
                      onChange={(e) => onFieldChange(field.key, e.target.value, true)}
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
                      onChange={(e) => onFieldChange(field.key, e.target.value, false)}
                    />
                  )}
                </div>
              )
            })}
          </div>

          {/* Single Primary Call-To-Action */}
          <div style={{ marginTop: '28px', display: 'flex', justifyContent: 'flex-end' }}>
            <button
              type="button"
              className="btn btn-primary"
              style={{ padding: '12px 28px', fontSize: '0.95rem' }}
              onClick={onRunPrediction}
              disabled={isPredicting}
            >
              <Send size={18} />
              {isPredicting ? 'Executing AI Models...' : 'Run Telemetry AI Prediction'}
            </button>
          </div>
        </section>

        {/* Results Side Panel */}
        <section className="surface">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">AI Inference Output</p>
              <h2>Prediction Assessment</h2>
            </div>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={onExplainPrediction}
              disabled={isExplaining || !prediction}
              style={{ padding: '6px 14px', fontSize: '0.82rem' }}
            >
              <Sparkles size={14} />
              {isExplaining ? 'Computing SHAP...' : 'Explain'}
            </button>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', marginBottom: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 16px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-md)' }}>
              <span style={{ fontSize: '0.88rem', color: 'var(--muted)' }}>Risk Category</span>
              <StatusBadge status={prediction?.risk_category ?? latestHistory?.risk_category ?? 'Awaiting Run'} />
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 16px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-md)' }}>
              <span style={{ fontSize: '0.88rem', color: 'var(--muted)' }}>Action Recommendation</span>
              <strong style={{ fontSize: '0.88rem' }}>{prediction?.maintenance_priority ?? 'Routine Monitoring'}</strong>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 16px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-md)' }}>
              <span style={{ fontSize: '0.88rem', color: 'var(--muted)' }}>Model Architecture</span>
              <strong style={{ fontSize: '0.88rem' }}>{prediction?.model_version ?? 'RandomForest / XGBoost'}</strong>
            </div>
          </div>

          {explanation && (
            <div style={{ marginTop: '20px', padding: '16px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-md)' }}>
              <h4 style={{ fontSize: '0.88rem', margin: '0 0 12px', color: 'var(--ink)' }}>Top SHAP Drivers</h4>
              {explanation.explanation.feature_importances.slice(0, 5).map((item) => (
                <div key={item.feature} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--border-line)', fontSize: '0.82rem' }}>
                  <span style={{ color: 'var(--muted)' }}>{item.feature.replaceAll('_', ' ')}</span>
                  <strong style={{ color: item.direction === 'positive' ? 'var(--danger)' : 'var(--success)' }}>
                    {formatNumber(item.shap_value, 3)}
                  </strong>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  )
}

export default TelemetryConsole
