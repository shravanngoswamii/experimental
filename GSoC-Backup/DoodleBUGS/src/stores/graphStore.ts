import { defineStore } from 'pinia';
import { ref, computed } from 'vue';
import type { GraphElement } from '../types';

export interface GraphContent {
  graphId: string;
  elements: GraphElement[];
}

export const useGraphStore = defineStore('graph', () => {
  const graphContents = ref<Map<string, GraphContent>>(new Map());
  const currentGraphId = ref<string | null>(null);

  const currentGraphElements = computed<GraphElement[]>(() => {
    if (currentGraphId.value && graphContents.value.has(currentGraphId.value)) {
      return graphContents.value.get(currentGraphId.value)!.elements;
    }
    return [];
  });

  const selectGraph = (graphId: string | null) => {
    currentGraphId.value = graphId;
    if (graphId && !graphContents.value.has(graphId)) {
      loadGraph(graphId);
    }
  };

  const createNewGraphContent = (graphId: string) => {
    const newContent: GraphContent = {
      graphId: graphId,
      elements: [],
    };
    graphContents.value.set(graphId, newContent);
    saveGraph(graphId, newContent);
  };

  const updateGraphElements = (graphId: string, newElements: GraphElement[]) => {
    if (graphContents.value.has(graphId)) {
      const content = graphContents.value.get(graphId)!;
      content.elements = newElements;
      saveGraph(graphId, content);
    }
  };

  const deleteGraphContent = (graphId: string) => {
    graphContents.value.delete(graphId);
    localStorage.removeItem(`doodlebugs-graph-${graphId}`);
    if (currentGraphId.value === graphId) {
      currentGraphId.value = null;
    }
  };

  const saveGraph = (graphId: string, content: GraphContent) => {
    localStorage.setItem(`doodlebugs-graph-${graphId}`, JSON.stringify(content));
  };

  const loadGraph = (graphId: string): GraphContent | null => {
    const storedContent = localStorage.getItem(`doodlebugs-graph-${graphId}`);
    if (storedContent) {
      const content: GraphContent = JSON.parse(storedContent);
      graphContents.value.set(graphId, content);
      return content;
    }
    return null;
  };

  return {
    graphContents,
    currentGraphId,
    currentGraphElements,
    selectGraph,
    createNewGraphContent,
    updateGraphElements,
    deleteGraphContent,
    saveGraph,
    loadGraph,
  };
});
