import { computed, ref } from 'vue';
import { useGraphStore } from '../stores/graphStore';
import type { GraphElement, GraphNode } from '../types';

/**
 * Composable for managing graph elements (nodes and edges) within the current active graph.
 * It provides reactive access to elements and functions to modify them,
 * which in turn updates the Pinia graph store.
 */
export function useGraphElements() {
  const graphStore = useGraphStore();

  // Reactive reference to the currently selected element for the properties panel.
  // This is managed locally within the composable and updated by components (e.g., GraphEditor).
  const selectedElement = ref<GraphElement | null>(null);

  // Computed property to access elements of the currently selected graph from the store.
  // This ensures that any changes to elements are automatically reflected in the UI
  // and persisted via the store's setter.
  const elements = computed<GraphElement[]>({
    get: () => graphStore.currentGraphElements,
    set: (newElements) => {
      if (graphStore.currentGraphId) {
        // When elements are updated, call the store's action to save them.
        graphStore.updateGraphElements(graphStore.currentGraphId, newElements);
      }
    }
  });

  /**
   * Adds a new graph element (node or edge) to the current graph.
   * @param {GraphElement} newElement - The element to add.
   */
  const addElement = (newElement: GraphElement) => {
    // Append the new element to the existing array.
    // The computed setter for `elements` will handle updating the store.
    elements.value = [...elements.value, newElement];
    selectedElement.value = newElement; // Automatically select the newly added element
  };

  /**
   * Updates an existing graph element in the current graph.
   * @param {GraphElement} updatedElement - The element with updated properties.
   */
  const updateElement = (updatedElement: GraphElement) => {
    // Map over the elements and replace the one with a matching ID.
    elements.value = elements.value.map(el =>
      el.id === updatedElement.id ? updatedElement : el
    );
    // If the updated element was currently selected, keep the `selectedElement` in sync.
    if (selectedElement.value?.id === updatedElement.id) {
      selectedElement.value = updatedElement;
    }
  };

  /**
   * Deletes a graph element from the current graph.
   * @param {string} elementId - The ID of the element to delete.
   */
  const deleteElement = (elementId: string) => {
    // Filter out the element with the matching ID.
    elements.value = elements.value.filter(el => el.id !== elementId);
    // If the deleted element was selected, deselect it.
    if (selectedElement.value?.id === elementId) {
      selectedElement.value = null;
    }
  };

  return {
    elements,          // Reactive array of all graph elements in the current graph
    selectedElement,   // Reactive reference to the currently selected element
    addElement,        // Function to add a new element
    updateElement,     // Function to update an existing element
    deleteElement,     // Function to delete an element
  };
}
