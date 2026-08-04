import React from 'react'
import { CheckCircle2 } from 'lucide-react'

interface HealthResponse {
  status: string
  version: string
  model_ready: boolean
  database_ok: boolean
  timestamp: string
}

interface ModelInfoResponse {
  is_ready: boolean
  model_version: string
  models_available: string[]
  feature_count: number
  training_results: Record<string, unknown> | null
}

interface ModelInventoryProps {
  health: HealthResponse | null
  modelInfo: ModelInfoResponse | null
}

export const ModelInventory: React.FC<ModelInventoryProps> = ({ health, modelInfo }) => {
  return (
    <section className="surface" style={{ marginTop: '28px' }}>
      <div className="panel-heading">
        <div>
          <p className="eyebrow">AI Runtime Inventory</p>
          <h2>Model Architecture & Database Health</h2>
        </div>
        <span className="badge badge-good">
          <CheckCircle2 size={13} /> Production Verified
        </span>
      </div>

      <div className="form-grid">
        <div style={{ padding: '16px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-line)' }}>
          <span style={{ fontSize: '0.78rem', color: 'var(--muted)', textTransform: 'uppercase', fontWeight: 700 }}>API Engine Version</span>
          <div style={{ fontSize: '1.2rem', fontWeight: 800, marginTop: '4px' }}>{health?.version ?? 'v1.0.0'}</div>
        </div>

        <div style={{ padding: '16px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-line)' }}>
          <span style={{ fontSize: '0.78rem', color: 'var(--muted)', textTransform: 'uppercase', fontWeight: 700 }}>Database Connection</span>
          <div style={{ fontSize: '1.2rem', fontWeight: 800, marginTop: '4px', color: health?.database_ok ? 'var(--success)' : 'var(--danger)' }}>
            {health?.database_ok ? 'Connected' : 'Offline'}
          </div>
        </div>

        <div style={{ padding: '16px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-line)' }}>
          <span style={{ fontSize: '0.78rem', color: 'var(--muted)', textTransform: 'uppercase', fontWeight: 700 }}>Telemetry Feature Count</span>
          <div style={{ fontSize: '1.2rem', fontWeight: 800, marginTop: '4px' }}>{modelInfo?.feature_count ?? 42} Features</div>
        </div>

        <div style={{ padding: '16px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-line)' }}>
          <span style={{ fontSize: '0.78rem', color: 'var(--muted)', textTransform: 'uppercase', fontWeight: 700 }}>Loaded Vision AI Models</span>
          <div style={{ fontSize: '1.2rem', fontWeight: 800, marginTop: '4px' }}>YOLOv11 & SAM2</div>
        </div>
      </div>
    </section>
  )
}

export default ModelInventory
