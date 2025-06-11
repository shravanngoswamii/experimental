import { ref, computed, watch } from 'vue';
import type { Core } from 'cytoscape';

/**
 * Composable for managing grid snapping and visual grid background for Cytoscape.js.
 * It controls the CSS variable for grid size and adds/removes a class for the grid background.
 * @param {Function} getCyInstance - A function that returns the Cytoscape.js core instance.
 */
export function useGridSnapping(getCyInstance: () => Core | null) {
  const isGridEnabledRef = ref<boolean>(false);
  const gridSizeRef = ref<number>(20); // Default grid size in pixels

  // Computed property for the CSS variable value
  const cssGridSize = computed<string>(() => `${gridSizeRef.value}px`);

  /**
   * Updates the CSS class and variable on the Cytoscape container
   * to show or hide the visual grid background.
   */
  const updateGridBackground = () => {
    const cy = getCyInstance();
    if (cy) {
      const container = cy.container();
      if (container) {
        if (isGridEnabledRef.value && gridSizeRef.value > 0) {
          container.classList.add('grid-background');
          container.style.setProperty('--grid-size', cssGridSize.value);
        } else {
          container.classList.remove('grid-background');
          container.style.removeProperty('--grid-size');
        }
      }
    }
  };

  /**
   * Enables the visual grid. Note: Actual snapping logic is handled in GraphCanvas.vue's
   * 'dragfree' event listener, where nodes are programmatically snapped to grid.
   */
  const enableGridSnapping = (): void => {
    isGridEnabledRef.value = true;
    updateGridBackground();
    console.log(`Visual grid enabled with ${gridSizeRef.value}px.`);
  };

  /**
   * Disables the visual grid.
   */
  const disableGridSnapping = (): void => {
    isGridEnabledRef.value = false;
    updateGridBackground();
    console.log('Visual grid disabled.');
  };

  /**
   * Sets a new grid size and updates the visual grid if enabled.
   * @param {number} size - The new grid size in pixels.
   */
  const setGridSize = (size: number): void => {
    gridSizeRef.value = size;
    if (isGridEnabledRef.value) {
      updateGridBackground();
    }
  };

  // Watch for changes in grid state or size to re-apply grid background
  watch([isGridEnabledRef, gridSizeRef], updateGridBackground);

  return {
    isGridEnabledRef, // Reactive state for grid enabled/disabled
    gridSizeRef,      // Reactive state for grid size
    cssGridSize,      // Computed CSS variable string
    enableGridSnapping,
    disableGridSnapping,
    setGridSize,
  };
}
