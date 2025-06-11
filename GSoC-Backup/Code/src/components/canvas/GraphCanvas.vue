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
  (e: 'node-dragged', nodeData: GraphNode): void;
  (e: 'node-dropped', payload: { nodeType: NodeType; position: { x: number; y: number } }): void;
}>();

const cyContainer = ref<HTMLElement | null>(null);
let cy: Core | null = null;

const { initCytoscape, destroyCytoscape, getCyInstance } = useGraphInstance();
const { enableGridSnapping, disableGridSnapping, setGridSize } = useGridSnapping(getCyInstance);

const validNodeTypes: NodeType[] = ['stochastic', 'deterministic', 'constant', 'observed', 'plate'];

/**
 * Helper function to format our GraphElement[] into Cytoscape's ElementDefinition[].
 * Cytoscape requires a 'data' object wrapper.
 */
const formatElementsForCytoscape = (elements: GraphElement[]): ElementDefinition[] => {
  return elements.map(el => {
    if (el.type === 'node') {
      return { group: 'nodes', data: { ...el }, position: el.position };
    } else { // 'edge'
      return { group: 'edges', data: { ...el } };
    }
  });
};

onMounted(() => {
  if (cyContainer.value) {
    cy = initCytoscape(cyContainer.value, formatElementsForCytoscape(props.elements));

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
      const snappedPos = {
        x: Math.round(node.position('x') / props.gridSize) * props.gridSize,
        y: Math.round(node.position('y') / props.gridSize) * props.gridSize,
      };
      node.position(snappedPos);
      emit('node-dragged', { ...node.data(), position: snappedPos });
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
  if (!cy) return; // Add guard clause

  const newElementIds = new Set(newElements.map(el => el.id));
  cy.batch(() => {
    cy!.elements().forEach(cyEl => {
      if (!newElementIds.has(cyEl.id())) {
        cyEl.remove();
      }
    });

    newElements.forEach(newEl => {
      const existingCyEl = cy!.getElementById(newEl.id);
      if (existingCyEl.empty()) {
        const formattedEl = formatElementsForCytoscape([newEl])[0];
        cy!.add(formattedEl);
      } else {
        existingCyEl.data(newEl);
        if (newEl.type === 'node') {
          const newNodePos = (newEl as GraphNode).position;
          const currentCyPos = existingCyEl.position();
          if (newNodePos.x !== currentCyPos.x || newNodePos.y !== currentCyPos.y) {
            existingCyEl.position(newNodePos);
          }
        }
      }
    });
  });
}, { deep: true });
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
