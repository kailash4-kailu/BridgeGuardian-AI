/**
 * BridgeGuardian AI — Centralized Domain Types & API Schemas
 * Enterprise TypeScript type definitions for Telemetry, Vision, and Drone Inspection modules.
 */

export type ApiState = 'checking' | 'online' | 'degraded' | 'offline'

export type TabType = 'drone' | 'console' | 'vision' | 'diagnostics'

export interface PredictionHistoryItem {
  id: number
  created_at: string
  health_score: number | null
  failure_probability: number | null
  rul_days: number | null
  risk_category: string | null
  maintenance_priority: string | null
  maintenance_recommendation: string | null
  model_version: string | null
  analysis_type: 'drone_campaign' | 'single_image' | 'structural_health' | string
  campaign_id: number | null
  image_count: number | null
  status: string
  summary_report: string | null
}

export interface PredictionHistoryResponse {
  items: PredictionHistoryItem[]
  total: number
  page: number
  limit: number
}

export type SensorPayload = Record<string, number | string | null>

export interface HealthResponse {
  status: string
  version: string
  model_ready: boolean
  database_ok: boolean
  timestamp: string
}

export interface ModelInfoResponse {
  is_ready: boolean
  model_version: string
  models_available: string[]
  feature_count: number
  training_results: Record<string, unknown> | null
}

export interface PredictionResponse {
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

export interface FeatureImportance {
  feature: string
  shap_value: number
  direction: string
}

export interface ExplainResponse {
  target: string
  explanation: {
    base_value: number
    feature_importances: FeatureImportance[]
    prediction_contribution: number
  }
}

export interface UploadedFile {
  filename: string
  filepath: string
}

export interface DefectMeasurement {
  width_mm: number
  length_mm: number
  area_pct: number
}

export interface DefectDetail {
  defect_id: string
  type: string
  severity: string
  confidence: number
  bbox: [number, number, number, number]
  measurements: DefectMeasurement
  images: string[]
  occurrences: number
  component?: string
}

export interface ImageResult {
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
    defects: DefectDetail[]
  }
  visualizations?: {
    original: string
    bboxes: string
    segmentation: string
    heatmap: string
  }
}

export interface InspectionRecord {
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
  } | null
}
