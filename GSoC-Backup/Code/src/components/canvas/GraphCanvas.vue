<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue';
import type { Core, EventObject, NodeSingular, ElementDefinition } from 'cytoscape';
import { useGraphInstance } from '../../composables/useGraphInstance';
import { useGridSnapping } from '../../composables/useGridSnapping';
import type { GraphElement, GraphNode, GraphEdge, NodeType, PaletteItemType } from '../../types';

const props = defineProps<{
  elements: GraphElement[];
  isGridEnabled: boolean;
  gridSize: number;
  currentMode: string;
}>();

const emit = defineEmits<{
  (e: 'canvas-tap', event: EventObject): void;
  (e: 'node-moved', payload: { nodeId: string, position: { x: number; y: number }, parentId: string | undefined }): void;
  (e: 'node-dropped', payload: { nodeType: NodeType; position: { x: number; y: number } }): void;
}>();

const cyContainer = ref<HTMLElement | null>(null);
let cy: Core | null = null;

const { initCytoscape, destroyCytoscape, getCyInstance } = useGraphInstance();
const { enableGridSnapping, disableGridSnapping, setGridSize } = useGridSnapping(getCyInstance);

const validNodeTypes: NodeType[] = ['stochastic', 'deterministic', 'constant', 'observed', 'plate'];

// This helper function now dynamically determines the relationshipType for edges before rendering.
const formatElementsForCytoscape = (elements: GraphElement[]): ElementDefinition[] => {
  return elements.map(el => {
    if (el.type === 'node') {
      return { group: 'nodes', data: { ...el }, position: el.position };
    } else {
      const edge = el as GraphEdge;
      const targetNode = elements.find(n => n.id === edge.target && n.type === 'node') as GraphNode | undefined;
      // The relationship is 'stochastic' if the target is stochastic or observed, otherwise it's 'deterministic'.
      const relType = (targetNode?.nodeType === 'stochastic' || targetNode?.nodeType === 'observed') ? 'stochastic' : 'deterministic';
      return { 
        group: 'edges', 
        data: { 
          ...edge,
          relationshipType: relType 
        } 
      };
    }
  });
};

onMounted(() => {
  if (cyContainer.value) {
    cy = initCytoscape(cyContainer.value, []);

    setGridSize(props.gridSize);
    if (props.isGridEnabled) {
      enableGridSnapping();
    } else {
      disableGridSnapping();
    }

    cy.on('tap', (evt: EventObject) => {
      emit('canvas-tap', evt);
    });

    cy.on('dragfree', 'node', (evt: EventObject) => {
      const node = evt.target as NodeSingular;
      if (node.data('nodeType') === 'plate') return;

      const snappedPos = {
        x: Math.round(node.position('x') / props.gridSize) * props.gridSize,
        y: Math.round(node.position('y') / props.gridSize) * props.gridSize,
      };
      node.position(snappedPos);

      let newParentId: string | undefined = undefined;
      const plates = cy?.nodes('[nodeType="plate"]');
      if (plates) {
          for (const plate of plates) {
            if (plate.id() === node.id()) continue;

            const bb = plate.boundingBox();
            if (snappedPos.x > bb.x1 && snappedPos.x < bb.x2 && snappedPos.y > bb.y1 && snappedPos.y < bb.y2) {
              newParentId = plate.id();
              break;
            }
          }
      }

      emit('node-moved', {
        nodeId: node.id(),
        position: snappedPos,
        parentId: newParentId
      });
    });

    cy.on('tap', 'node, edge', (evt: EventObject) => {
      cy?.elements().removeClass('cy-selected');
      evt.target.addClass('cy-selected');
    });
    cy.on('tap', (evt: EventObject) => {
      if (evt.target === cy) {
        cy?.elements().removeClass('cy-selected');
      }
    });

    cyContainer.value.addEventListener('dragover', (event) => {
      event.preventDefault();
      if (event.dataTransfer) {
        event.dataTransfer.dropEffect = 'copy';
      }
    });

    cyContainer.value.addEventListener('drop', (event) => {
      event.preventDefault();

      if (event.dataTransfer) {
        const droppedItemType = event.dataTransfer.getData('text/plain') as PaletteItemType;
        if (validNodeTypes.includes(droppedItemType as NodeType)) {
          const bbox = cyContainer.value?.getBoundingClientRect();
          if (bbox && cy) {
            const clientX = event.clientX;
            const clientY = event.clientY;
            const renderedPos = { x: clientX - bbox.left, y: clientY - bbox.top };
            const modelPos = cy.panzoom().renderedPositionToModelPosition(renderedPos);
            emit('node-dropped', { nodeType: droppedItemType as NodeType, position: modelPos });
          }
        }
      }
    });
  }
});

onUnmounted(() => {
  if (cy) {
    destroyCytoscape(cy);
  }
});

watch(() => props.isGridEnabled, (newValue) => {
  if (newValue) {
    enableGridSnapping();
  } else {
    disableGridSnapping();
  }
});

watch(() => props.gridSize, (newValue) => {
  setGridSize(newValue);
  if (props.isGridEnabled) {
    enableGridSnapping();
  }
});

watch(() => props.elements, (newElements) => {
  // FIX 1: Add a guard clause to prevent running if cy is not initialized.
  if (!cy) return;

  const formattedElements = formatElementsForCytoscape(newElements);

  cy.batch(() => {
    const newElementIds = new Set(newElements.map(el => el.id));

    cy!.elements().forEach(cyEl => {
      if (!newElementIds.has(cyEl.id())) {
        cyEl.remove();
      }
    });

    formattedElements.forEach(formattedEl => {
      // FIX 2: Ensure the element has an ID before proceeding.
      if (!formattedEl.data.id) return;

      const existingCyEl = cy!.getElementById(formattedEl.data.id);
      
      if (existingCyEl.empty()) {
        cy!.add(formattedEl);
      } else {
        existingCyEl.data(formattedEl.data);
        if (formattedEl.group === 'nodes') {
          const newNode = formattedEl as ElementDefinition & { position: {x: number, y: number} };
          const currentCyPos = existingCyEl.position();
          if (newNode.position.x !== currentCyPos.x || newNode.position.y !== currentCyPos.y) {
            existingCyEl.position(newNode.position);
          }
          // FIX 3: Safely get the parent ID. .parent() returns a collection.
          const parentCollection = existingCyEl.parent();
          const currentParentId = parentCollection.length > 0 ? parentCollection.first().id() : undefined;
          
          if (newNode.data.parent !== currentParentId) {
            existingCyEl.move({ parent: newNode.data.parent ?? null });
          }
        }
      }
    });
  });
}, { deep: true, immediate: true });
</script>

<template>
  <div
    ref="cyContainer"
    class="cytoscape-container"
    :class="{
      'grid-background': isGridEnabled && gridSize > 0,
      'mode-add-node': currentMode === 'add-node',
      'mode-add-edge': currentMode === 'add-edge',
      'mode-select': currentMode === 'select'
    }"
    :style="{ '--grid-size': `${gridSize}px` }"
  ></div>
</template>

<style scoped>
.cytoscape-container {
  flex-grow: 1;
  background-color: var(--color-background-soft);
  position: relative;
  overflow: hidden;
  cursor: grab;
}

.cytoscape-container.mode-add-node {
  cursor: crosshair;
}

.cytoscape-container.mode-add-edge {
  cursor: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="%23333" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-arrow-right"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>') 12 12, crosshair;
}
</style>
