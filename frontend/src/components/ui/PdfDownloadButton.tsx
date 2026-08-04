import React, { useState } from 'react'
import { FileDown, RefreshCw, CheckCircle2, RotateCcw, AlertTriangle } from 'lucide-react'

interface PdfDownloadButtonProps {
  onDownload: () => Promise<void>
  disabled?: boolean
  disabledTooltip?: string
  label?: string
  className?: string
  style?: React.CSSProperties
  showStatusBanner?: boolean
}

export const PdfDownloadButton: React.FC<PdfDownloadButtonProps> = ({
  onDownload,
  disabled = false,
  disabledTooltip = 'No report available yet. Run an inspection to generate a report.',
  label = 'Download PDF Report',
  className = 'btn btn-primary',
  style,
  showStatusBanner = true,
}) => {
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle')
  const [errorInfo, setErrorInfo] = useState<{ title: string; message: string } | null>(null)
  const [successInfo, setSuccessInfo] = useState<string | null>(null)

  const handleAction = async () => {
    if (disabled || status === 'loading') return

    setStatus('loading')
    setErrorInfo(null)
    setSuccessInfo(null)

    try {
      if (typeof navigator !== 'undefined' && !navigator.onLine) {
        throw new Error('NETWORK_OFFLINE')
      }

      await onDownload()

      setStatus('success')
      setSuccessInfo('PDF inspection report downloaded successfully.')

      setTimeout(() => {
        setStatus('idle')
        setSuccessInfo(null)
      }, 3500)
    } catch (err: any) {
      console.error('[PDF Report Generation Failure]:', err)

      let title = 'Unable to Generate Report'
      let message = 'The report could not be created at this time. Please try again in a few moments.'

      const errStr = (err?.message || err?.toString() || '').toUpperCase()

      if (errStr.includes('OFFLINE') || errStr.includes('FETCH') || err?.name === 'TypeError') {
        title = 'Network Connection Error'
        message = 'Network connection lost. Please check your internet connection and try again.'
      } else if (errStr.includes('503') || errStr.includes('502') || errStr.includes('UNAVAILABLE')) {
        title = 'Reporting Service Unavailable'
        message = 'The reporting service is currently unavailable. Please try again later.'
      } else if (errStr.includes('404') || errStr.includes('NOT_FOUND')) {
        title = 'Report Generation Pending'
        message = 'The inspection completed successfully, but the PDF report is not yet available. Click Retry to generate.'
      }

      setErrorInfo({ title, message })
      setStatus('error')
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', width: style?.width || 'auto' }}>
      {showStatusBanner && successInfo && (
        <div
          role="status"
          style={{
            padding: '8px 12px',
            background: 'var(--success-bg)',
            color: 'var(--success)',
            border: '1px solid var(--success)',
            borderRadius: 'var(--radius-sm)',
            fontSize: '0.8rem',
            fontWeight: 600,
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}
        >
          <CheckCircle2 size={14} /> {successInfo}
        </div>
      )}

      {showStatusBanner && errorInfo && (
        <div
          role="alert"
          style={{
            padding: '10px 14px',
            background: 'var(--danger-bg)',
            color: 'var(--danger)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            borderRadius: 'var(--radius-sm)',
            fontSize: '0.82rem',
            lineHeight: 1.4,
            display: 'flex',
            flexDirection: 'column',
            gap: '6px',
          }}
        >
          <div style={{ fontWeight: 700, display: 'flex', alignItems: 'center', gap: '6px' }}>
            <AlertTriangle size={14} /> {errorInfo.title}
          </div>
          <div>{errorInfo.message}</div>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={handleAction}
            style={{
              alignSelf: 'flex-start',
              padding: '4px 10px',
              fontSize: '0.76rem',
              marginTop: '4px',
              border: '1px solid currentColor',
              color: 'var(--danger)',
              background: 'var(--surface)',
            }}
          >
            <RotateCcw size={12} /> Retry Report Download
          </button>
        </div>
      )}

      <button
        type="button"
        className={className}
        onClick={handleAction}
        disabled={disabled || status === 'loading'}
        title={disabled ? disabledTooltip : label}
        style={{
          ...style,
          opacity: disabled ? 0.6 : 1,
          cursor: disabled || status === 'loading' ? 'not-allowed' : 'pointer',
          background:
            status === 'success'
              ? 'var(--success)'
              : status === 'error'
              ? 'var(--danger)'
              : style?.background,
          borderColor:
            status === 'success'
              ? 'var(--success)'
              : status === 'error'
              ? 'var(--danger)'
              : style?.borderColor,
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '8px',
        }}
      >
        {status === 'loading' ? (
          <>
            <RefreshCw size={16} className="spinning" />
            Generating Report...
          </>
        ) : status === 'success' ? (
          <>
            <CheckCircle2 size={16} />
            Report Ready
          </>
        ) : status === 'error' ? (
          <>
            <RotateCcw size={16} />
            Retry Generation
          </>
        ) : (
          <>
            <FileDown size={16} />
            {label}
          </>
        )}
      </button>
    </div>
  )
}

export default PdfDownloadButton
