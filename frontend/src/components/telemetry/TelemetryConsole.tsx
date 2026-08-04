import React, { useState, useMemo } from 'react'
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
  ArrowRightLeft,
  ChevronDown,
  Info,
} from 'lucide-react'
import MetricCard from '../ui/MetricCard'
import StatusBadge from '../ui/StatusBadge'
import CompareDrawer from '../compare/CompareDrawer'

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
  { key: 'Strain_microstrain', label: 'Strain', unit: 'microstrain', group: 'structure', step: 0.1, min: -5000, max: 5000 },
  { key: 'Deflection_mm', label: 'Deflection', unit: 'mm', group: 'structure', step: 0.01, min: 0 },
  { key: 'Vibration_ms2', label: 'Vibration', unit: 'm/s2', group: 'structure', step: 0.01, min: 0 },
  { key: 'Tilt_deg', label: 'Tilt', unit: 'deg', group: 'structure', step: 0.01 },
  { key: 'Displacement_mm', label: 'Displacement', unit: 'mm', group: 'structure', step: 0.01 },
  { key: 'Crack_Propagation_mm', label: 'Crack growth', unit: 'mm', group: 'structure', step: 0.001, min: 0 },
  { key: 'Corrosion_Level_percent', label: 'Corrosion', unit: '%', group: 'structure', step: 0.01, min: 0, max: 100 },
  { key: 'Cable_Member_Tension_kN', label: 'Cable tension', unit: 'kN', group: 'structure', step: 0.1, min: 0 },
  { key: 'Bearing_Joint_Forces_kN', label: 'Bearing forces', unit: 'kN', group: 'structure', step: 0.1, min: 0 },
  { key: 'Fatigue_Accumulation_au', label: 'Fatigue', unit: 'a.u.', group: 'structure', step: 0.01, min: 0 },
  { key: 'Modal_Frequency_Hz', label: 'Modal frequency', unit: 'Hz', group: 'structure', step: 0.01, min: 0 },
  { key: 'Temperature_C', label: 'Temperature', unit: 'C', group: 'environment', step: 0.1, min: -50, max: 80 },
  { key: 'Humidity_percent', label: 'Humidity', unit: '%', group: 'environment', step: 0.1, min: 0, max: 100 },
  { key: 'Wind_Speed_ms', label: 'Wind speed', unit: 'm/s', group: 'environment', step: 0.1, min: 0 },
  { key: 'Wind_Direction_deg', label: 'Wind direction', unit: 'deg', group: 'environment', step: 1, min: 0, max: 360 },
  { key: 'Precipitation_mmh', label: 'Precipitation', unit: 'mm/h', group: 'environment', step: 0.1, min: 0 },
  { key: 'Water_Level_m', label: 'Water level', unit: 'm', group: 'environment', step: 0.1 },
  { key: 'Seismic_Activity_ms2', label: 'Seismic activity', unit: 'm/s2', group: 'environment', step: 0.001, min: 0 },
  { key: 'Solar_Radiation_Wm2', label: 'Solar radiation', unit: 'W/m2', group: 'environment', step: 1, min: 0 },
  { key: 'Air_Quality_Index_AQI', label: 'Air quality', unit: 'AQI', group: 'environment', step: 1, min: 0 },
  { key: 'Soil_Settlement_mm', label: 'Soil settlement', unit: 'mm', group: 'environment', step: 0.01 },
  { key: 'Vehicle_Load_tons', label: 'Vehicle load', unit: 'tons', group: 'load', step: 0.1, min: 0 },
  { key: 'Traffic_Volume_vph', label: 'Traffic volume', unit: 'veh/h', group: 'load', step: 1, min: 0 },
  { key: 'Pedestrian_Load_pph', label: 'Pedestrian load', unit: 'people/h', group: 'load', step: 1, min: 0 },
  { key: 'Impact_Events_g', label: 'Impact events', unit: 'g', group: 'load', step: 0.001, min: 0 },
  { key: 'Dynamic_Load_Distribution_percent', label: 'Load distribution', unit: '%', group: 'load', step: 0.1, min: 0, max: 100 },
  { key: 'Axle_Counts_pmin', label: 'Axle count', unit: '/min', group: 'load', step: 0.1, min: 0 },
  { key: 'Anomaly_Detection_Score', label: 'Anomaly score', group: 'diagnostics', step: 0.01, min: 0, max: 1 },
  { key: 'Energy_Dissipation_au', label: 'Energy dissipation', unit: 'a.u.', group: 'diagnostics', step: 0.001, min: 0 },
  { key: 'Acoustic_Emissions_levels', label: 'Acoustic emissions', group: 'diagnostics', step: 0.01, min: 0 },
  { key: 'Visual_Analysis_Defect_Score', label: 'Visual defect score', group: 'diagnostics', step: 0.001, min: 0 },
  { key: 'Electrical_Resistance_ohms', label: 'Electrical resistance', unit: 'ohms', group: 'diagnostics', step: 0.001, min: 0 },
  { key: 'Localized_Strain_Hotspot', label: 'Localized hotspot', group: 'diagnostics', step: 1, min: 0, max: 1 },
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
  { key: 'Flood_Event_Flag', label: 'Flood event', group: 'diagnostics', step: 1, min: 0, max: 1 },
  { key: 'High_Winds_Storms', label: 'High winds', group: 'diagnostics', step: 1, min: 0, max: 1 },
  { key: 'Landslide_Ground_Movement', label: 'Ground movement', group: 'diagnostics', step: 1, min: 0, max: 1 },
  { key: 'Abnormal_Traffic_Load_Surges', label: 'Traffic surge', group: 'diagnostics', step: 1, min: 0, max: 1 },
  { key: 'SHI_Predicted_7d_Ahead', label: 'SHI 7d forecast', group: 'diagnostics', step: 0.01, min: 0, max: 1 },
  { key: 'SHI_Predicted_30d_Ahead', label: 'SHI 30d forecast', group: 'diagnostics', step: 0.01, min: 0, max: 1 },
]

function formatNumber(val: number | null | undefined, digits = 1) {
  if (val === null || val === undefined || Number.isNaN(val)) return '--'
  return val.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  })
}

function getValidationError(form: SensorPayload): string | null {
  const corrosion = Number(form['Corrosion_Level_percent'])
  if (!isNaN(corrosion) && (corrosion < 0 || corrosion > 100)) {
    return 'Corrosion level must be between 0% and 100%.'
  }

  const humidity = Number(form['Humidity_percent'])
  if (!isNaN(humidity) && (humidity < 0 || humidity > 100)) {
    return 'Humidity percentage must be between 0% and 100%.'
  }

  const loadDist = Number(form['Dynamic_Load_Distribution_percent'])
  if (!isNaN(loadDist) && (loadDist < 0 || loadDist > 100)) {
    return 'Dynamic Load Distribution must be between 0% and 100%.'
  }

  const crack = Number(form['Crack_Propagation_mm'])
  if (!isNaN(crack) && crack < 0) {
    return 'Crack growth propagation cannot be a negative value.'
  }

  const vibration = Number(form['Vibration_ms2'])
  if (!isNaN(vibration) && vibration < 0) {
    return 'Vibration measurement cannot be a negative value.'
  }

  return null
}

function formatShapInsight(feature: string, val: number, direction: string): string {
  const cleanName = feature.replaceAll('_', ' ')
  const pct = Math.abs(val * 100).toFixed(1)
  if (direction === 'positive' || val > 0) {
    return `Elevated ${cleanName} increased structural failure risk by ${pct}%.`
  } else {
    return `Normal ${cleanName} reduced structural failure risk by ${pct}%.`
  }
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
  DEFAULT_INPUT,
  CRITICAL_PRESET,
}) => {
  const [activeGroup, setActiveGroup] = useState<FieldGroup>('structure')
  const [isCompareOpen, setIsCompareOpen] = useState(false)

  const validationError = useMemo(() => getValidationError(form), [form])
  const visibleFields = SENSOR_FIELDS.filter((f) => f.group === activeGroup)

  const healthScore = prediction?.health_score ?? null
  const pofValue = prediction?.failure_probability ?? null
  const rulValue = prediction?.rul_days ?? null
  const confidenceValue = prediction
    ? (prediction.prediction_confidence <= 1
        ? prediction.prediction_confidence * 100
        : prediction.prediction_confidence)
    : null

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Compare Side-by-Side Drawer Component */}
      <CompareDrawer
        isOpen={isCompareOpen}
        onClose={() => setIsCompareOpen(false)}
        runA={null}
        runB={prediction}
        labelA="Baseline Threshold"
        labelB="Current AI Prediction"
      />

      {/* Primary KPI Metrics Grid */}
      <div className="stats-card-grid">
        <MetricCard
          title="Structural Health Index"
          value={formatNumber(healthScore, 1)}
          unit={healthScore !== null ? '/ 100' : undefined}
          trendLabel={healthScore !== null ? (healthScore < 50 ? 'Critical Distress' : healthScore < 70 ? 'Moderate Risk' : 'Optimal Integrity') : 'Awaiting Analysis'}
          trendTone={healthScore !== null ? (healthScore < 50 ? 'danger' : healthScore < 70 ? 'warning' : 'good') : 'neutral'}
          icon={CircleGauge}
        />
        <MetricCard
          title="Failure Probability"
          value={formatNumber(pofValue, 2)}
          unit={pofValue !== null ? '%' : undefined}
          trendLabel={pofValue !== null ? (pofValue > 10 ? 'High Risk' : 'Low Risk') : 'Awaiting Analysis'}
          trendTone={pofValue !== null ? (pofValue > 10 ? 'danger' : 'good') : 'neutral'}
          icon={BarChart3}
        />
        <MetricCard
          title="Remaining Useful Life"
          value={formatNumber(rulValue, 0)}
          unit={rulValue !== null ? 'Days' : undefined}
          trendLabel={rulValue !== null ? 'Calibrated Model' : 'Awaiting Analysis'}
          trendTone={rulValue !== null ? 'warning' : 'neutral'}
          icon={HistoryIcon}
        />
        <MetricCard
          title="Prediction Confidence"
          value={formatNumber(confidenceValue, 1)}
          unit={confidenceValue !== null ? '%' : undefined}
          trendLabel={confidenceValue !== null ? 'Predictive Analytics' : 'Awaiting Analysis'}
          trendTone={confidenceValue !== null ? 'good' : 'neutral'}
          icon={Sparkles}
        />
      </div>

      {/* Inference Loading Banner */}
      {isPredicting && (
        <div className="banner" role="status" style={{ background: 'var(--surface-alt)', borderLeft: '4px solid var(--primary)', borderRadius: '12px', display: 'flex', alignItems: 'center', gap: '12px' }}>
          <RefreshCw size={20} className="spinning" style={{ color: 'var(--primary)' }} />
          <div>
            <strong>Running AI Analysis...</strong>
            <div style={{ fontSize: '0.82rem', color: 'var(--muted)', marginTop: '2px' }}>
              Calculating Structural Health Index • Estimating Failure Probability • Computing Remaining Useful Life • Generating Recommendation
            </div>
          </div>
        </div>
      )}

      {/* Input Validation Error Banner */}
      {validationError && (
        <div className="banner" role="alert" style={{ background: 'var(--danger-bg)', borderColor: 'rgba(239, 68, 68, 0.4)', color: 'var(--danger)', borderRadius: '12px' }}>
          <AlertTriangle size={18} />
          <span><b>Input Validation Error:</b> {validationError}</span>
        </div>
      )}

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
                      min={field.min}
                      max={field.max}
                      value={value ?? ''}
                      onChange={(e) => onFieldChange(field.key, e.target.value, false)}
                    />
                  )}
                </div>
              )
            })}
          </div>

          {/* Action Row */}
          <div style={{ marginTop: '28px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => setIsCompareOpen(true)}
              disabled={!prediction}
              style={{ fontSize: '0.88rem' }}
            >
              <ArrowRightLeft size={16} /> Compare Run Side-by-Side
            </button>

            <button
              type="button"
              className="btn btn-primary"
              style={{ padding: '12px 28px', fontSize: '0.95rem' }}
              onClick={onRunPrediction}
              disabled={isPredicting || Boolean(validationError)}
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
            <div style={{ display: 'flex', gap: '6px' }}>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={onExplainPrediction}
                disabled={isExplaining || !prediction}
                style={{ padding: '6px 12px', fontSize: '0.82rem' }}
              >
                <Sparkles size={14} />
                {isExplaining ? 'Computing SHAP...' : 'Explain'}
              </button>
            </div>
          </div>

          {/* Executive Engineering Insight */}
          <div style={{ padding: '14px 16px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-md)', borderLeft: '4px solid var(--primary)', marginBottom: '20px', fontSize: '0.85rem', lineHeight: 1.5 }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 800, color: 'var(--primary)', textTransform: 'uppercase', display: 'block', marginBottom: '4px' }}>Executive Engineering Directive</span>
            {prediction ? (
              (healthScore !== null && healthScore < 50) || prediction.risk_category === 'Critical'
                ? 'CRITICAL WARNING: Accelerated structural deterioration detected. Immediate physical engineering inspection required. Recommend heavy vehicle traffic restrictions within 14 days.'
                : (healthScore !== null && healthScore < 70) || prediction.risk_category === 'Poor'
                ? 'ATTENTION REQUIRED: Moderate structural distress observed. Schedule detailed engineering assessment within 30 days.'
                : 'OPTIMAL INTEGRITY: Structural health parameters are within normal baseline tolerances. Schedule routine inspection in 180 days.'
            ) : (
              'AWAITING ANALYSIS: No telemetry prediction has been generated yet. Configure sensor parameters on the left and click "Run Telemetry AI Prediction" to execute inference.'
            )}
          </div>



          {/* Natural Language SHAP Explanations */}
          {explanation && (
            <div style={{ padding: '16px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-md)' }}>
              <h4 style={{ fontSize: '0.9rem', margin: '0 0 12px', color: 'var(--ink)' }}>Engineering SHAP Insights</h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '0.85rem' }}>
                {explanation.explanation.feature_importances.slice(0, 4).map((item) => (
                  <div key={item.feature} style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', padding: '6px 0', borderBottom: '1px solid var(--border-line)' }}>
                    <Info size={15} style={{ color: item.direction === 'positive' ? 'var(--danger)' : 'var(--success)', marginTop: '2px', flexShrink: 0 }} />
                    <span style={{ lineHeight: 1.4 }}>
                      {formatShapInsight(item.feature, item.shap_value, item.direction)}
                    </span>
                  </div>
                ))}
              </div>

              {/* Expandable Numeric SHAP Details */}
              <details style={{ marginTop: '12px', fontSize: '0.8rem', color: 'var(--muted)' }}>
                <summary style={{ cursor: 'pointer', fontWeight: 600 }}>View Raw SHAP Values</summary>
                <div style={{ marginTop: '8px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  {explanation.explanation.feature_importances.map((item) => (
                    <div key={item.feature} style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}>
                      <span>{item.feature}</span>
                      <strong>{formatNumber(item.shap_value, 4)}</strong>
                    </div>
                  ))}
                </div>
              </details>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}

export default TelemetryConsole
