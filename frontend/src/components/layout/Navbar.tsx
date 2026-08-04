import React from 'react'
import { ChevronRight, RefreshCw } from 'lucide-react'
import StatusBadge from '../ui/StatusBadge'
import type { TabType } from '../../types'

interface HealthResponse {
  status: string
  version: string
  model_ready: boolean
  database_ok: boolean
  timestamp: string
}

interface NavbarProps {
  activeTab: TabType
  health: HealthResponse | null
  apiState: string
  onRefresh: () => void
  isRefreshing: boolean
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  health,
  apiState,
  onRefresh,
  isRefreshing,
}) => {
  const getTabLabel = () => {
    switch (activeTab) {
      case 'drone':
        return 'Drone Inspection Campaign'
      case 'console':
        return 'Structural Telemetry Console'
      case 'vision':
        return 'Single Image Computer Vision'
      default:
        return 'Inspection Overview'
    }
  }

  const systemStatusLabel = health?.status === 'healthy' ? 'System Online' : apiState
  const modelStatusLabel = health?.model_ready ? 'Model Ready' : 'Model Standby'

  return (
    <header className="topbar">
      {/* Breadcrumb Path */}
      <div className="nav-breadcrumbs">
        <span>BridgeGuardian AI</span>
        <ChevronRight size={14} aria-hidden="true" />
        <span>Asset Monitoring</span>
        <ChevronRight size={14} aria-hidden="true" />
        <span className="active">{getTabLabel()}</span>
      </div>

      {/* System Health Indicators */}
      <div className="status-cluster">
        <StatusBadge
          status={systemStatusLabel}
          tone={health?.status === 'healthy' ? 'good' : 'warning'}
        />

        <StatusBadge
          status={modelStatusLabel}
          tone={health?.model_ready ? 'good' : 'neutral'}
        />

        <button
          className="btn btn-secondary"
          type="button"
          onClick={onRefresh}
          disabled={isRefreshing}
          title="Refresh backend status"
          style={{ padding: '6px 12px', minHeight: '34px' }}
        >
          <RefreshCw size={14} className={isRefreshing ? 'spinning' : ''} aria-hidden="true" />
        </button>
      </div>
    </header>
  )
}

export default Navbar
