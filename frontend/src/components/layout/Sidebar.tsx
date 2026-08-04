import React from 'react'
import { Plane, Activity, Camera, History } from 'lucide-react'

export type TabType = 'drone' | 'console' | 'vision' | 'history'

interface SidebarProps {
  activeTab: TabType
  onTabChange: (tab: TabType) => void
}

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, onTabChange }) => {
  return (
    <aside className="sidebar">
      {/* Brand Icon */}
      <div className="brand-mark" title="BridgeGuardian AI Platform">
        <img src="/logo-icon.svg" alt="BridgeGuardian AI Logo" />
      </div>

      {/* Vertical Rail Navigation */}
      <nav className="rail" aria-label="Primary Navigation">
        <button
          type="button"
          className={`rail-button ${activeTab === 'drone' ? 'active' : ''}`}
          onClick={() => onTabChange('drone')}
          title="Drone Inspection Campaign"
        >
          <Plane size={20} aria-hidden="true" />
        </button>

        <button
          type="button"
          className={`rail-button ${activeTab === 'console' ? 'active' : ''}`}
          onClick={() => onTabChange('console')}
          title="Telemetry Sensor Console"
        >
          <Activity size={20} aria-hidden="true" />
        </button>

        <button
          type="button"
          className={`rail-button ${activeTab === 'vision' ? 'active' : ''}`}
          onClick={() => onTabChange('vision')}
          title="Single Image Vision AI"
        >
          <Camera size={20} aria-hidden="true" />
        </button>

        <button
          type="button"
          className={`rail-button ${activeTab === 'history' ? 'active' : ''}`}
          onClick={() => onTabChange('history')}
          title="Audit Trail & History"
        >
          <History size={20} aria-hidden="true" />
        </button>
      </nav>
    </aside>
  )
}

export default Sidebar
