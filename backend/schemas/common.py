"""
BridgeGuardian AI — Common Schema Definitions
Provides standardized API envelope responses, pagination structures, and error models.
"""
from __future__ import annotations

from typing import Generic, List, Optional, TypeVar, Any
from pydantic import BaseModel, Field

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """Standardized API JSON envelope wrapper."""
    success: bool = True
    message: str = "Operation completed successfully"
    data: Optional[T] = None
    error: Optional[str] = None


class PageParams(BaseModel):
    """Pagination and sorting query parameters."""
    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")
    sort_by: Optional[str] = Field(default="created_at", description="Field to sort by")
    sort_order: Optional[str] = Field(default="desc", description="Sort direction ('asc' or 'desc')")


class PaginatedResponse(BaseModel, Generic[T]):
    """Standardized paginated list envelope."""
    items: List[T]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool


def create_paginated_response(
    items: List[Any], total: int, page: int, page_size: int
) -> PaginatedResponse:
    """Helper to build a PaginatedResponse envelope."""
    total_pages = (total + page_size - 1) // page_size if page_size > 0 else 0
    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1,
    )
