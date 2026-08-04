import React, { useState } from 'react'
import { Upload, Camera, FileDown, Sparkles, RefreshCw, X, Image as ImageIcon } from 'lucide-react'
import StatusBadge from '../ui/StatusBadge'

interface SingleVisionConsoleProps {
  visionImageId: string | null
  visionImageUrl: string | null
  visionFilename: string | null
  visionPrediction: any | null
  activeOverlay: string
  setActiveOverlay: (overlay: string) => void
  isUploading: boolean
  isAnalyzing: boolean
  isGeneratingReport: boolean
  onImageUpload: (e: React.ChangeEvent<HTMLInputElement>) => void
  onRunVisionPredict: () => void
  onDownloadReport: () => void
  onClearImage: () => void
}

function formatNumber(val: number | null | undefined, digits = 1) {
  if (val === null || val === undefined || Number.isNaN(val)) return '--'
  return val.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  })
}

export const SingleVisionConsole: React.FC<SingleVisionConsoleProps> = ({
  visionImageId,
  visionImageUrl,
  visionFilename,
  visionPrediction,
  activeOverlay,
  setActiveOverlay,
  isUploading,
  isAnalyzing,
  isGeneratingReport,
  onImageUpload,
  onRunVisionPredict,
  onDownloadReport,
  onClearImage,
}) => {
  return (
    <div className="content-grid">
      <section className="surface">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Single Image Inspection</p>
            <h2>Computer Vision Defect Analysis</h2>
          </div>
          {visionImageId && (
            <button type="button" className="btn btn-secondary" onClick={onClearImage}>
              <X size={15} /> Clear Image
            </button>
          )}
        </div>

        {!visionImageId ? (
          <div className="dropzone-container">
            <input
              type="file"
              id="vision-file-input"
              accept="image/jpeg,image/png,image/webp"
              onChange={onImageUpload}
              disabled={isUploading}
              style={{ display: 'none' }}
            />
            <label htmlFor="vision-file-input" style={{ cursor: 'pointer', display: 'block' }}>
              <div className="dropzone-icon-ring">
                {isUploading ? <RefreshCw size={32} className="spinning" /> : <Upload size={32} />}
              </div>
              <h3 style={{ margin: '0 0 8px', fontSize: '1.1rem', fontWeight: 700 }}>
                {isUploading ? 'Uploading Image...' : 'Upload Single Inspection Photo'}
              </h3>
              <p style={{ margin: 0, color: 'var(--muted)', fontSize: '0.9rem' }}>
                Drag & drop a high-res photo or click to browse (JPEG, PNG, WEBP)
              </p>
            </label>
          </div>
        ) : (
          <div>
            <div style={{ position: 'relative', borderRadius: 'var(--radius-lg)', overflow: 'hidden', border: '1px solid var(--border-line)', minHeight: '360px', background: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              {isAnalyzing ? (
                <div style={{ textAlign: 'center', color: '#FFF', padding: '40px' }}>
                  <RefreshCw size={40} className="spinning" style={{ color: 'var(--primary)', marginBottom: '16px' }} />
                  <h3>Running YOLOv11 & SAM2 Vision Engine...</h3>
                </div>
              ) : (
                <img
                  src={visionPrediction ? visionPrediction.visualizations[activeOverlay] : visionImageUrl || ''}
                  alt={visionFilename || 'Bridge Inspection'}
                  style={{ maxWidth: '100%', maxHeight: '520px', objectFit: 'contain' }}
                />
              )}
            </div>

            {!visionPrediction ? (
              <div style={{ marginTop: '20px', display: 'flex', justifyContent: 'center' }}>
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={onRunVisionPredict}
                  disabled={isAnalyzing}
                  style={{ padding: '12px 28px' }}
                >
                  <Sparkles size={18} />
                  {isAnalyzing ? 'Analyzing Image...' : 'Run Vision Defect Analysis'}
                </button>
              </div>
            ) : (
              <div style={{ marginTop: '20px', display: 'flex', gap: '8px', flexWrap: 'wrap', justifyContent: 'center' }}>
                {[
                  { key: 'original', label: 'Original Photo' },
                  { key: 'bboxes', label: 'Bounding Boxes' },
                  { key: 'segmentation', label: 'SAM2 Masks' },
                  { key: 'heatmap', label: 'Heatmap Overlay' },
                ].map((ov) => (
                  <button
                    key={ov.key}
                    type="button"
                    className={`btn ${activeOverlay === ov.key ? 'btn-primary' : 'btn-secondary'}`}
                    onClick={() => setActiveOverlay(ov.key)}
                    style={{ padding: '6px 14px', fontSize: '0.82rem' }}
                  >
                    {ov.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </section>

      {/* Results Side Panel */}
      <section className="surface">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Vision AI Assessment</p>
            <h2>Defect Metrics</h2>
          </div>
        </div>

        {visionPrediction ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div style={{ padding: '16px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-line)' }}>
              <span style={{ fontSize: '0.78rem', fontWeight: 800, color: 'var(--muted)', textTransform: 'uppercase' }}>Structural Health Score</span>
              <div style={{ fontSize: '2.2rem', fontWeight: 800, color: 'var(--ink)', margin: '4px 0' }}>
                {formatNumber(visionPrediction.predictions.health_score, 1)} / 100
              </div>
              <StatusBadge status={visionPrediction.predictions.risk_category} />
            </div>

            <button
              type="button"
              className="btn btn-primary"
              onClick={onDownloadReport}
              disabled={isGeneratingReport}
              style={{ width: '100%' }}
            >
              <FileDown size={18} />
              {isGeneratingReport ? 'Generating PDF...' : 'Download PDF Inspection Report'}
            </button>

            <div style={{ padding: '16px', background: 'var(--surface-alt)', borderRadius: 'var(--radius-md)' }}>
              <h4 style={{ margin: '0 0 12px', fontSize: '0.88rem' }}>Extracted CV Features</h4>
              {[
                { label: 'Crack Density', val: `${visionPrediction.features.crack_density}%` },
                { label: 'Max Crack Width', val: `${visionPrediction.features.crack_width} mm` },
                { label: 'Corrosion Area', val: `${visionPrediction.features.corrosion_percent}%` },
                { label: 'Concrete Spalling', val: `${visionPrediction.features.spalling_percent}%` },
              ].map((row) => (
                <div key={row.label} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--border-line)', fontSize: '0.85rem' }}>
                  <span style={{ color: 'var(--muted)' }}>{row.label}</span>
                  <strong>{row.val}</strong>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div style={{ textAlign: 'center', padding: '40px 20px', color: 'var(--muted)' }}>
            <ImageIcon size={48} style={{ opacity: 0.3, marginBottom: '12px' }} />
            <p style={{ margin: 0, fontSize: '0.9rem' }}>Upload an inspection photo to analyze defect bounding boxes and structural health metrics.</p>
          </div>
        )}
      </section>
    </div>
  )
}

export default SingleVisionConsole
