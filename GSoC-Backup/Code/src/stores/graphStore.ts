import { defineStore } from 'pinia';
import { ref, computed } from 'vue';
import type { GraphElement } from '../types';

// Interface defining the structure of a graph's content
export interface GraphContent {
  graphId: string;
  elements: GraphElement[]; // Array of nodes and edges
  // Add other graph-specific settings like zoom, pan position etc. if needed
}

export const useGraphStore = defineStore('graph', () => {
  // A Map to store all graph contents, keyed by graphId
  const graphContents = ref<Map<string, GraphContent>>(new Map());
  // The ID of the currently selected graph
  const currentGraphId = ref<string | null>(null);

  /**
   * Computed property that provides reactive, read-only access to the elements of the currently selected graph.
   * State modifications should be done via the `updateGraphElements` action.
   */
  const currentGraphElements = computed<GraphElement[]>(() => {
    if (currentGraphId.value && graphContents.value.has(currentGraphId.value)) {
      return graphContents.value.get(currentGraphId.value)!.elements;
    }
    return []; // Return empty array if no graph is selected or found
  });

  /**
   * Sets the currently active graph. If the graph is not in memory, attempts to load it.
   * @param {string | null} graphId - The ID of the graph to select, or null to deselect.
   */
  const selectGraph = (graphId: string | null) => {
    currentGraphId.value = graphId;
    if (graphId && !graphContents.value.has(graphId)) {
      loadGraph(graphId); // Attempt to load if not already in memory
    }
  };

  /**
   * Creates a new empty graph content object and adds it to the store and persistence.
   * @param {string} graphId - The unique ID for the new graph.
   */
  const createNewGraphContent = (graphId: string) => {
    const newContent: GraphContent = {
      graphId: graphId,
      elements: [], // Start with an empty array of elements
    };
    graphContents.value.set(graphId, newContent);
    saveGraph(graphId, newContent); // Persist the new empty graph
  };

  /**
   * Updates the elements of a specific graph. This is the designated action for all element modifications.
   * @param {string} graphId - The ID of the graph to update.
   * @param {GraphElement[]} newElements - The new array of elements for the graph.
   */
  const updateGraphElements = (graphId: string, newElements: GraphElement[]) => {
    if (graphContents.value.has(graphId)) {
      const content = graphContents.value.get(graphId)!;
      content.elements = newElements;
      saveGraph(graphId, content); // Persist changes
    }
  };

  /**
   * Deletes a graph's content from the store and persistence.
   * @param {string} graphId - The ID of the graph to delete.
   */
  const deleteGraphContent = (graphId: string) => {
    graphContents.value.delete(graphId); // Remove from reactive map
    localStorage.removeItem(`doodlebugs-graph-${graphId}`); // Remove from local storage
    if (currentGraphId.value === graphId) {
      currentGraphId.value = null; // Deselect if the current graph is deleted
    }
  };

  // --- Persistence Methods (Local Storage) ---

  /**
   * Saves a specific graph's content to local storage.
   * @param {string} graphId - The ID of the graph to save.
   * @param {GraphContent} content - The content object of the graph.
   */
  const saveGraph = (graphId: string, content: GraphContent) => {
    localStorage.setItem(`doodlebugs-graph-${graphId}`, JSON.stringify(content));
  };

  /**
   * Loads a specific graph's content from local storage into the store.
   * @param {string} graphId - The ID of the graph to load.
   * @returns {GraphContent | null} The loaded graph content, or null if not found.
   */
  const loadGraph = (graphId: string): GraphContent | null => {
    const storedContent = localStorage.getItem(`doodlebugs-graph-${graphId}`);
    if (storedContent) {
      const content: GraphContent = JSON.parse(storedContent);
      graphContents.value.set(graphId, content); // Add to reactive map
      return content;
    }
    return null;
  };

  return {
    graphContents,        // Reactive map of all graph contents
    currentGraphId,       // Reactive ID of the currently selected graph
    currentGraphElements, // Computed elements of the current graph (read-only)
    selectGraph,
    createNewGraphContent,
    updateGraphElements,
    deleteGraphContent,
    saveGraph,
    loadGraph,
  };
});
