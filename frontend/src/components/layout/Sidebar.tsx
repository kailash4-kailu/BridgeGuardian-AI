import React from 'react'
import { Plane, Activity, Camera } from 'lucide-react'
import type { TabType } from '../../types'

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
      <nav className="rail" role="tablist" aria-label="Workflow Modules">
        <button
          type="button"
          role="tab"
          id="tab-drone"
          aria-selected={activeTab === 'drone'}
          aria-controls="panel-drone"
          aria-label="Drone Inspection Campaign Workflow"
          tabIndex={0}
          className={`rail-button ${activeTab === 'drone' ? 'active' : ''}`}
          onClick={() => onTabChange('drone')}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              onTabChange('drone')
            }
          }}
          title="Drone Inspection Campaign (Multi-photo Flight Analysis)"
        >
          <Plane size={20} aria-hidden="true" />
        </button>

        <button
          type="button"
          role="tab"
          id="tab-console"
          aria-selected={activeTab === 'console'}
          aria-controls="panel-console"
          aria-label="Structural Telemetry Console Workflow"
          tabIndex={0}
          className={`rail-button ${activeTab === 'console' ? 'active' : ''}`}
          onClick={() => onTabChange('console')}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              onTabChange('console')
            }
          }}
          title="Structural Telemetry Console (Sensor Predictive Analytics)"
        >
          <Activity size={20} aria-hidden="true" />
        </button>

        <button
          type="button"
          role="tab"
          id="tab-vision"
          aria-selected={activeTab === 'vision'}
          aria-controls="panel-vision"
          aria-label="Single Image Computer Vision AI Workflow"
          tabIndex={0}
          className={`rail-button ${activeTab === 'vision' ? 'active' : ''}`}
          onClick={() => onTabChange('vision')}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              onTabChange('vision')
            }
          }}
          title="Single Image Computer Vision (Defect Segmentation)"
        >
          <Camera size={20} aria-hidden="true" />
        </button>
      </nav>
    </aside>
  )
}

export default Sidebar
