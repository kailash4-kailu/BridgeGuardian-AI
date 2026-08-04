/**
 * BridgeGuardian AI — Centralized API Client
 * Encapsulates HTTP communications, error normalization, and typed endpoints.
 */

import { API_BASE } from './api'
import type {
  HealthResponse,
  ModelInfoResponse,
  PredictionResponse,
  ExplainResponse,
  SensorPayload,
  UploadedFile,
  InspectionRecord,
  PredictionHistoryResponse,
} from '../types'

export class ApiError extends Error {
  public status: number
  constructor(message: string, status = 500) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`
  try {
    const res = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
      ...options,
    })

    if (!res.ok) {
      let message = `${res.status} ${res.statusText}`
      try {
        const text = await res.text()
        const parsed = JSON.parse(text)
        message = parsed.detail || parsed.message || message
      } catch {
        // Fallback to default message
      }
      throw new ApiError(message, res.status)
    }

    return (await res.json()) as T
  } catch (err: any) {
    if (err instanceof ApiError) throw err
    throw new ApiError(err.message || 'Network request failed', 0)
  }
}

export const apiClient = {
  getHealth: () => request<HealthResponse>('/health'),

  getModelInfo: () => request<ModelInfoResponse>('/model-info'),

  predictTelemetry: (payload: SensorPayload) =>
    request<PredictionResponse>('/predict', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  explainPrediction: (payload: SensorPayload, target = 'health_score') =>
    request<ExplainResponse>('/explain', {
      method: 'POST',
      body: JSON.stringify({ input_data: payload, target }),
    }),

  uploadSingleVisionImage: async (file: File) => {
    const formData = new FormData()
    formData.append('files', file)
    const res = await fetch(`${API_BASE}/vision/upload-image`, {
      method: 'POST',
      body: formData,
    })
    if (!res.ok) {
      throw new ApiError('Image upload failed', res.status)
    }
    return (await res.json()) as any[]
  },

  predictSingleVision: (imageId: string, pixelToMm = 0.5) =>
    request<any>('/vision/vision-predict', {
      method: 'POST',
      body: JSON.stringify({ image_id: imageId, pixel_to_mm: pixelToMm }),
    }),

  uploadDroneImages: async (files: File[]) => {
    const formData = new FormData()
    files.forEach((file) => formData.append('files', file))
    const res = await fetch(`${API_BASE}/inspection/upload-images`, {
      method: 'POST',
      body: formData,
    })
    if (!res.ok) {
      throw new ApiError('Drone campaign image upload failed', res.status)
    }
    return (await res.json()) as UploadedFile[]
  },

  runDroneInspection: (imagePaths: string[], pixelToMm = 0.5) =>
    request<{ inspection_id: number; status: string }>('/inspection/run-inspection', {
      method: 'POST',
      body: JSON.stringify({ image_paths: imagePaths, pixel_to_mm: pixelToMm }),
    }),

  getDroneInspectionStatus: (id: number) =>
    request<InspectionRecord>(`/inspection/${id}`),

  getHistory: (limit = 50, offset = 0) =>
    request<PredictionHistoryResponse>(`/history?limit=${limit}&offset=${offset}`),
}
