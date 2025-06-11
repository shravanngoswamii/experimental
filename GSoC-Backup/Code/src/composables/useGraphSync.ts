import { watch } from 'vue';
import type { Core } from 'cytoscape';
import type { GraphElement, GraphNode, GraphEdge } from '../types';

/**
 * Composable to synchronize a Cytoscape.js instance with a reactive array of GraphElements.
 * It observes changes in the elements array and updates the Cytoscape graph accordingly.
 *
 * @param {() => Core | null} getCyInstance - A function that returns the current Cytoscape.js Core instance.
 * @param {GraphElement[]} elements - The reactive array of graph elements (nodes and edges).
 */
export function useGraphSync(getCyInstance: () => Core | null, elements: GraphElement[]) {

  // Watch the `elements` array deeply for changes
  watch(elements, (newElements) => {
    const cy = getCyInstance();
    if (!cy) {
      console.warn('Cytoscape instance not available for synchronization.');
      return;
    }

    const newElementIds = new Set(newElements.map(el => el.id));
    // FIX: Removed unused 'oldElementIds' variable.

    // Use a batch operation for better performance
    cy.batch(() => {
        // 1. Remove elements that are no longer in the newElements array
        cy.elements().forEach(cyEl => {
          if (!newElementIds.has(cyEl.id())) {
              cyEl.remove();
          }
        });

        // 2. Add or update elements that are in newElements
        newElements.forEach(newEl => {
          const existingCyEl = cy.getElementById(newEl.id);

          if (existingCyEl.empty()) {
              // Element does not exist in Cytoscape, add it
              if (newEl.type === 'node') {
              const nodeData = newEl as GraphNode;
              cy.add({
                  group: 'nodes',
                  data: nodeData,
                  position: nodeData.position
              });
              } else if (newEl.type === 'edge') {
              const edgeData = newEl as GraphEdge;
              cy.add({
                  group: 'edges',
                  data: {
                      id: edgeData.id,
                      name: edgeData.name,
                      source: edgeData.source,
                      target: edgeData.target,
                      relationshipType: edgeData.relationshipType
                  }
              });
              }
          } else {
              // Element exists, update its data
              existingCyEl.data(newEl);
              if (newEl.type === 'node') {
              // Only update position if it has changed
              const newNodePos = (newEl as GraphNode).position;
              const currentCyPos = existingCyEl.position();
              if (newNodePos.x !== currentCyPos.x || newNodePos.y !== currentCyPos.y) {
                  existingCyEl.position(newNodePos);
              }
              }
          }
        });
    });
  }, { deep: true }); // Deep watch is necessary for nested changes within elements
}
