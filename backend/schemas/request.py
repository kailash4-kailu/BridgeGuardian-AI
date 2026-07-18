"""
BridgeGuardian AI — Pydantic Request Schemas
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator


class BridgeSensorInput(BaseModel):
    """Input schema for a single bridge sensor reading."""

    # Structural sensors
    Strain_microstrain: Optional[float] = Field(None, description="Strain in microstrain", example=734.5)
    Deflection_mm: Optional[float] = Field(None, description="Deflection in mm", example=14.99)
    Vibration_ms2: Optional[float] = Field(None, description="Vibration in m/s²", example=1.20)
    Tilt_deg: Optional[float] = Field(None, description="Tilt angle in degrees", example=0.72)
    Displacement_mm: Optional[float] = Field(None, description="Displacement in mm", example=22.36)
    Crack_Propagation_mm: Optional[float] = Field(None, description="Crack propagation in mm", example=0.015)
    Corrosion_Level_percent: Optional[float] = Field(None, description="Corrosion level %", example=0.15)
    Cable_Member_Tension_kN: Optional[float] = Field(None, description="Cable tension in kN", example=447.9)
    Bearing_Joint_Forces_kN: Optional[float] = Field(None, description="Bearing forces in kN", example=260.1)
    Fatigue_Accumulation_au: Optional[float] = Field(None, description="Fatigue accumulation", example=0.30)
    Modal_Frequency_Hz: Optional[float] = Field(None, description="Modal frequency in Hz", example=1.90)

    # Environmental
    Temperature_C: Optional[float] = Field(None, description="Temperature in °C", example=15.0)
    Humidity_percent: Optional[float] = Field(None, description="Humidity %", example=60.3)
    Wind_Speed_ms: Optional[float] = Field(None, description="Wind speed in m/s", example=6.5)
    Wind_Direction_deg: Optional[float] = Field(None, description="Wind direction degrees", example=180.4)
    Precipitation_mmh: Optional[float] = Field(None, description="Precipitation in mm/h", example=0.0)
    Water_Level_m: Optional[float] = Field(None, description="Water level in m", example=2.0)
    Seismic_Activity_ms2: Optional[float] = Field(None, description="Seismic activity in m/s²", example=0.0)
    Solar_Radiation_Wm2: Optional[float] = Field(None, description="Solar radiation W/m²", example=446.5)
    Air_Quality_Index_AQI: Optional[float] = Field(None, description="Air quality index", example=55.0)
    Soil_Settlement_mm: Optional[float] = Field(None, description="Soil settlement in mm", example=0.30)

    # Load & traffic
    Vehicle_Load_tons: Optional[float] = Field(None, description="Vehicle load in tons", example=16.4)
    Traffic_Volume_vph: Optional[float] = Field(None, description="Traffic volume vehicles/h", example=853.2)
    Pedestrian_Load_pph: Optional[float] = Field(None, description="Pedestrian load persons/h", example=96.3)
    Impact_Events_g: Optional[float] = Field(None, description="Impact events in g", example=0.0)
    Dynamic_Load_Distribution_percent: Optional[float] = Field(None, description="Dynamic load distribution %", example=90.1)
    Axle_Counts_pmin: Optional[float] = Field(None, description="Axle counts per minute", example=43.4)

    # Sensor diagnostics
    Anomaly_Detection_Score: Optional[float] = Field(None, description="Anomaly detection score", example=0.0)
    Energy_Dissipation_au: Optional[float] = Field(None, description="Energy dissipation", example=0.156)
    Acoustic_Emissions_levels: Optional[float] = Field(None, description="Acoustic emissions level", example=10.45)
    Visual_Analysis_Defect_Score: Optional[float] = Field(None, description="Visual defect score", example=0.004)
    Electrical_Resistance_ohms: Optional[float] = Field(None, description="Electrical resistance in ohms", example=0.282)
    Localized_Strain_Hotspot: Optional[float] = Field(None, description="Localized strain hotspot flag", example=0.0)

    # Categorical
    Bridge_Mood_Meter: Optional[str] = Field(None, description="Bridge mood category", example="Healthy")
    Vibration_Anomaly_Location: Optional[str] = Field(None, description="Vibration anomaly location", example="Deck")

    # Event flags
    Flood_Event_Flag: Optional[float] = Field(None, description="Flood event flag (0/1)", example=0.0)
    High_Winds_Storms: Optional[float] = Field(None, description="High winds/storms flag", example=0.0)
    Landslide_Ground_Movement: Optional[float] = Field(None, description="Landslide movement flag", example=0.0)
    Abnormal_Traffic_Load_Surges: Optional[float] = Field(None, description="Abnormal traffic flag", example=0.0)

    # Optional forecasts for RUL calculation
    SHI_Predicted_7d_Ahead: Optional[float] = Field(None, description="SHI forecast 7 days ahead", example=0.80)
    SHI_Predicted_30d_Ahead: Optional[float] = Field(None, description="SHI forecast 30 days ahead", example=0.75)

    class Config:
        json_schema_extra = {
            "example": {
                "Strain_microstrain": 734.5,
                "Deflection_mm": 14.99,
                "Vibration_ms2": 1.20,
                "Tilt_deg": 0.72,
                "Displacement_mm": 22.36,
                "Crack_Propagation_mm": 0.015,
                "Corrosion_Level_percent": 0.15,
                "Cable_Member_Tension_kN": 447.9,
                "Bearing_Joint_Forces_kN": 260.1,
                "Fatigue_Accumulation_au": 0.30,
                "Modal_Frequency_Hz": 1.90,
                "Temperature_C": 15.0,
                "Humidity_percent": 60.3,
                "Wind_Speed_ms": 6.5,
                "Traffic_Volume_vph": 853.2,
                "Bridge_Mood_Meter": "Healthy",
            }
        }


class ExplainRequest(BaseModel):
    """Request schema for SHAP explanation endpoint."""
    input_data: BridgeSensorInput
    target: str = Field(default="health_score", description="Model target to explain")

    @field_validator("target")
    @classmethod
    def validate_target(cls, v: str) -> str:
        allowed = {"health_score", "failure_prob", "maintenance_alert"}
        if v not in allowed:
            raise ValueError(f"target must be one of {allowed}")
        return v


class TrainRequest(BaseModel):
    """Request schema to trigger training."""
    config_path: str = Field(default="config/config.yaml")
    force_retrain: bool = Field(default=False)
