<script setup lang="ts">
import type { NodeType, PaletteItemType } from '../../types';

const emit = defineEmits<{
  (e: 'select-palette-item', itemType: PaletteItemType): void;
}>();

// Define the available node types for the palette
const nodeItems: { label: string; type: NodeType; icon: string; styleClass: string; description: string }[] = [
  { label: 'Stochastic', type: 'stochastic', icon: '~', styleClass: 'stochastic', description: 'Random variable with a distribution' },
  { label: 'Deterministic', type: 'deterministic', icon: '<-', styleClass: 'deterministic', description: 'Logical function of parents' },
  { label: 'Constant', type: 'constant', icon: 'C', styleClass: 'constant', description: 'A fixed value or parameter' },
  { label: 'Observed', type: 'observed', icon: 'O', styleClass: 'observed', description: 'A data node with a fixed value' },
  { label: 'Plate', type: 'plate', icon: '[]', styleClass: 'plate', description: 'Represents a loop structure' },
];

// Define connection types for the palette
const connectionItems: { label: string; type: 'add-stochastic-edge' | 'add-deterministic-edge'; styleClass: string; description: string }[] = [
  { label: 'Stochastic (~)', type: 'add-stochastic-edge', styleClass: 'stochastic', description: 'Stochastic dependency' },
  { label: 'Deterministic (<--)', type: 'add-deterministic-edge', styleClass: 'deterministic', description: 'Logical dependency' },
];

/**
 * Handles the drag start event for a palette item.
 * @param event The DragEvent object.
 * @param itemType The type of item being dragged.
 */
const onDragStart = (event: DragEvent, itemType: PaletteItemType) => {
  if (event.dataTransfer) {
    event.dataTransfer.setData('text/plain', itemType);
    event.dataTransfer.effectAllowed = 'copy';
  }
};

/**
 * Handles the click event on a palette item.
 * @param itemType The type of item that was clicked.
 */
const onClickPaletteItem = (itemType: PaletteItemType) => {
  emit('select-palette-item', itemType);
};
</script>

<template>
  <div class="node-palette">
    <div class="palette-section">
      <h5 class="section-title">Nodes</h5>
      <div class="palette-grid">
        <div
          v-for="node in nodeItems"
          :key="node.type"
          class="palette-card"
          :class="node.styleClass"
          draggable="true"
          @dragstart="onDragStart($event, node.type)"
          @click="onClickPaletteItem(node.type)"
          :title="node.description"
        >
          <div class="card-icon" :class="`icon-${node.type}`">{{ node.icon }}</div>
          <span class="card-label">{{ node.label }}</span>
        </div>
      </div>
    </div>

    <div class="palette-section">
      <h5 class="section-title">Connections</h5>
      <div class="palette-grid">
        <div
          v-for="connection in connectionItems"
          :key="connection.type"
          class="palette-card"
          :class="connection.styleClass"
          draggable="true"
          @dragstart="onDragStart($event, connection.type)"
          @click="onClickPaletteItem(connection.type)"
          :title="connection.description"
        >
          <div class="card-icon connection-icon" :class="`icon-${connection.type}`"></div>
          <span class="card-label">{{ connection.label }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.node-palette {
  padding: 12px;
  background-color: var(--color-background-soft);
  height: 100%;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.palette-section {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.section-title {
  font-size: 0.9em;
  font-weight: 600;
  color: var(--color-heading);
  padding-bottom: 8px;
  border-bottom: 1px solid var(--color-border-light);
  margin: 0;
}

.palette-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px;
}

.palette-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 15px 10px;
  border-radius: 8px;
  border: 1px solid var(--color-border);
  background-color: #fff;
  cursor: grab;
  text-align: center;
  transition: all 0.2s ease-in-out;
  user-select: none;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

.palette-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  border-color: var(--color-primary);
}

.palette-card:active {
  cursor: grabbing;
  transform: translateY(-1px);
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

.card-label {
  font-size: 0.8em;
  font-weight: 500;
  color: var(--color-text);
  margin-top: 8px;
}

.card-icon {
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.5em;
  font-weight: bold;
  border-radius: 50%;
  color: #fff;
}

/* Node Specific Styles */
.icon-stochastic { background-color: #dc3545; }
.icon-deterministic { background-color: #28a745; border-radius: 8px; font-size: 1.2em; }
.icon-constant { background-color: #6c757d; border-radius: 4px; }
.icon-observed {
  background-color: #fff;
  border: 2px dashed #007bff;
  color: #007bff;
}
.icon-plate {
  background-color: #fff;
  border: 2px dashed #495057;
  color: #495057;
  border-radius: 8px;
  font-size: 1.2em;
}

/* Connection Specific Styles */
.connection-icon {
  width: 100%;
  height: 20px;
  background-color: transparent !important;
  position: relative;
  border-radius: 0;
}

.connection-icon::before { /* The line */
  content: '';
  position: absolute;
  left: 10%;
  right: 10%;
  top: 50%;
  height: 2px;
  transform: translateY(-50%);
}

.connection-icon::after { /* The arrowhead */
  content: '';
  position: absolute;
  right: 10%;
  top: 50%;
  transform: translateY(-50%);
  width: 0;
  height: 0;
  border-style: solid;
}

.icon-add-stochastic-edge::before {
  border-top: 2px dashed #dc3545;
}
.icon-add-stochastic-edge::after {
  border-width: 6px 0 6px 10px;
  border-color: transparent transparent transparent #dc3545;
}

.icon-add-deterministic-edge::before {
  background-color: #28a745;
}
.icon-add-deterministic-edge::after {
  border-width: 6px 0 6px 10px;
  border-color: transparent transparent transparent #28a745;
}

</style>
