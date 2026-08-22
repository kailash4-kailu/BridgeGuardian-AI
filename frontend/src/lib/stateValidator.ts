/**
 * BridgeGuardian AI — State Synchronization Validator
 * Enforces single-source-of-truth consistency across Dashboard, Backend, and PDF outputs.
 */

export class StateSynchronizationError extends Error {
  constructor(message: string) {
    super(`[StateSynchronizationError] ${message}`);
    this.name = 'StateSynchronizationError';
  }
}

export interface StateCheckPayload {
  healthScore: number | string | null;
  failureProbability: number | string | null;
  remainingUsefulLife: number | string | null;
  maintenancePriority: string | null;
  acceptedImages: number;
  totalImages: number;
}

export function validateStateConsistency(
  dashboardState: StateCheckPayload,
  resultState: StateCheckPayload
): boolean {
  if (dashboardState.healthScore !== resultState.healthScore) {
    throw new StateSynchronizationError(
      `Health Score mismatch! Dashboard: ${dashboardState.healthScore}, Result: ${resultState.healthScore}`
    );
  }

  if (dashboardState.failureProbability !== resultState.failureProbability) {
    throw new StateSynchronizationError(
      `Failure Probability mismatch! Dashboard: ${dashboardState.failureProbability}, Result: ${resultState.failureProbability}`
    );
  }

  if (dashboardState.remainingUsefulLife !== resultState.remainingUsefulLife) {
    throw new StateSynchronizationError(
      `Remaining Useful Life mismatch! Dashboard: ${dashboardState.remainingUsefulLife}, Result: ${resultState.remainingUsefulLife}`
    );
  }

  if (dashboardState.maintenancePriority !== resultState.maintenancePriority) {
    throw new StateSynchronizationError(
      `Maintenance Priority mismatch! Dashboard: ${dashboardState.maintenancePriority}, Result: ${resultState.maintenancePriority}`
    );
  }

  if (dashboardState.acceptedImages !== resultState.acceptedImages) {
    throw new StateSynchronizationError(
      `Accepted Images count mismatch! Dashboard: ${dashboardState.acceptedImages}, Result: ${resultState.acceptedImages}`
    );
  }

  if (dashboardState.totalImages !== resultState.totalImages) {
    throw new StateSynchronizationError(
      `Total Images count mismatch! Dashboard: ${dashboardState.totalImages}, Result: ${resultState.totalImages}`
    );
  }

  return true;
}
