<script setup lang="ts">
import BaseButton from '../ui/BaseButton.vue';
import type { NodeType } from '../../types';

// The 'props' constant is removed as it's not used in the script block.
// The props are still available directly in the template.
defineProps<{
  currentMode: string; // 'select', 'add-node', 'add-edge'
  currentNodeType: NodeType; // The currently selected node type for adding
  isConnecting: boolean; // Whether an edge connection is in progress
  sourceNodeName: string | undefined; // Name of the source node if connecting
}>();

const emit = defineEmits<{
  (e: 'update:currentMode', mode: string): void;
  (e: 'update:currentNodeType', type: NodeType): void;
}>();

// Available node types for BUGS models
const availableNodeTypes: { label: string; value: NodeType }[] = [
  { label: 'Stochastic Node', value: 'stochastic' },
  { label: 'Deterministic Node', value: 'deterministic' },
  { label: 'Constant Node', value: 'constant' },
  { label: 'Observed Node', value: 'observed' },
  { label: 'Plate (Loop)', value: 'plate' },
];

/**
 * Sets the current mode of the graph editor.
 * @param mode The mode to set ('select', 'add-node', 'add-edge').
 */
const setMode = (mode: string) => {
  emit('update:currentMode', mode);
};

/**
 * Updates the currently selected node type for adding new nodes.
 * @param event The change event from the select element.
 */
const updateNodeType = (event: Event) => {
  const target = event.target as HTMLSelectElement;
  emit('update:currentNodeType', target.value as NodeType);
};
</script>

<template>
  <div class="canvas-toolbar">
    <BaseButton
      :class="{ active: currentMode === 'select' }"
      @click="setMode('select')"
    >
      Select
    </BaseButton>
    <BaseButton
      :class="{ active: currentMode === 'add-node' }"
      @click="setMode('add-node')"
    >
      Add Node
    </BaseButton>
    <BaseButton
      :class="{ active: currentMode === 'add-edge' }"
      @click="setMode('add-edge')"
    >
      Add Edge
    </BaseButton>

    <div v-if="currentMode === 'add-node'" class="node-type-selector">
      <label for="node-type">Node Type:</label>
      <select id="node-type" :value="currentNodeType" @change="updateNodeType">
        <option v-for="type in availableNodeTypes" :key="type.value" :value="type.value">
          {{ type.label }}
        </option>
      </select>
    </div>

    <span v-if="isConnecting" class="connecting-message">
      Connecting from: <strong>{{ sourceNodeName }}</strong> (Click target node)
    </span>
  </div>
</template>

<style scoped>
.canvas-toolbar {
  display: flex;
  gap: 10px;
  padding: 10px;
  background-color: var(--color-background-soft);
  border-bottom: 1px solid var(--color-border-light);
  align-items: center;
  flex-wrap: wrap; /* Allow items to wrap on smaller screens */
  flex-shrink: 0; /* Prevent toolbar from shrinking */
}

.canvas-toolbar .base-button {
  /* Override base button styles for toolbar specific look if needed */
  padding: 8px 15px;
  border: 1px solid var(--color-border-dark);
  background-color: #fff;
  cursor: pointer;
  border-radius: 4px;
  transition: background-color 0.2s ease, border-color 0.2s ease;
}

.canvas-toolbar .base-button.active {
  background-color: var(--color-primary);
  color: white;
  border-color: var(--color-primary);
}

.canvas-toolbar .base-button:hover:not(.active) {
  background-color: var(--color-border-light);
}

.node-type-selector {
  display: flex;
  align-items: center;
  gap: 5px;
  margin-left: 10px; /* Spacing from buttons */
}

.node-type-selector label {
  font-size: 0.9em;
  color: #555;
}

.node-type-selector select {
  padding: 6px 8px;
  border: 1px solid var(--color-border-dark);
  border-radius: 4px;
  background-color: white;
  font-size: 0.9em;
  cursor: pointer;
}

.connecting-message {
  margin-left: auto; /* Push to the right */
  font-style: italic;
  color: #666;
  font-size: 0.9em;
  white-space: nowrap; /* Prevent message from wrapping */
}
</style>
