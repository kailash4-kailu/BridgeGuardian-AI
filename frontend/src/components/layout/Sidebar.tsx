import React from 'react'
import { Plane, Activity, Camera } from 'lucide-react'

export type TabType = 'drone' | 'console' | 'vision'

interface SidebarProps {
  activeTab: TabType
  onTabChange: (tab: TabType) => void
}

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, onTabChange }) => {
  return (
    <aside className="sidebar" role="navigation" aria-label="Main Application Rail">
      {/* Brand Mark */}
      <div className="brand-mark" title="BridgeGuardian AI Enterprise Platform">
        <img src="/logo-icon.svg" alt="BridgeGuardian AI Logo" />
      </div>

      {/* Vertical Rail Navigation */}
      <nav className="rail" role="tablist" aria-label="Workflow Tabs">
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'drone'}
          aria-label="Drone Inspection Campaign Workflow"
          tabIndex={0}
          className={`rail-button ${activeTab === 'drone' ? 'active' : ''}`}
          onClick={() => onTabChange('drone')}
          title="Drone Inspection Campaign (Multi-photo analysis)"
        >
          <Plane size={20} aria-hidden="true" />
        </button>

        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'console'}
          aria-label="Structural Telemetry Console Workflow"
          tabIndex={0}
          className={`rail-button ${activeTab === 'console' ? 'active' : ''}`}
          onClick={() => onTabChange('console')}
          title="Telemetry Sensor Console (Predictive AI)"
        >
          <Activity size={20} aria-hidden="true" />
        </button>

        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'vision'}
          aria-label="Single Image Computer Vision AI Workflow"
          tabIndex={0}
          className={`rail-button ${activeTab === 'vision' ? 'active' : ''}`}
          onClick={() => onTabChange('vision')}
          title="Single Image Computer Vision Analysis"
        >
          <Camera size={20} aria-hidden="true" />
        </button>
      </nav>
    </aside>
  )
}

export default Sidebar
