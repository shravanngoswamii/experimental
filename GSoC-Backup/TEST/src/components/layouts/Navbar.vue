<script setup lang="ts">
import { computed } from 'vue';
import type { NodeType } from '../../types';
import BaseButton from '../ui/BaseButton.vue';
import BaseInput from '../ui/BaseInput.vue';
import DropdownMenu from '../common/DropdownMenu.vue';

const props = defineProps<{
  projectName: string | null;
  activeGraphName: string | null;
  isGridEnabled: boolean;
  gridSize: number;
  currentMode: string;
  currentNodeType: NodeType;
  isLeftSidebarOpen: boolean;
  isRightSidebarOpen: boolean;
}>();

const emit = defineEmits<{
  (e: 'update:isGridEnabled', value: boolean): void;
  (e: 'update:gridSize', value: number): void;
  (e: 'update:currentMode', mode: string): void;
  (e: 'update:currentNodeType', type: NodeType): void;
  (e: 'new-project'): void;
  (e: 'new-graph'): void;
  (e: 'save-current-graph'): void;
  (e: 'toggle-left-sidebar'): void;
  (e: 'toggle-right-sidebar'): void;
  (e: 'open-about-modal'): void;
}>();

const displayTitle = computed(() => {
  if (props.projectName && props.activeGraphName) {
    return `${props.projectName} — ${props.activeGraphName}`;
  }
  if (props.projectName) {
    return props.projectName;
  }
  return 'No Project Selected';
});

const basicNodeTypes: { label: string; value: NodeType }[] = [
  { label: 'Stochastic Node', value: 'stochastic' },
  { label: 'Deterministic Node', value: 'deterministic' },
  { label: 'Constant Node', value: 'constant' },
  { label: 'Observed Node', value: 'observed' },
  { label: 'Plate (Loop)', value: 'plate' },
];

const setAddNodeType = (type: NodeType) => {
  emit('update:currentNodeType', type);
  emit('update:currentMode', 'add-node');
};

const handleGridSizeInput = (event: Event) => {
  const target = event.target as HTMLInputElement;
  emit('update:gridSize', Number(target.value));
};
</script>

<template>
  <nav class="navbar">
    <!-- Left Section -->
    <div class="navbar-left">
      <div class="navbar-brand">DoodleBUGS</div>
      <div class="navbar-menu">
        <!-- Project Menu -->
        <DropdownMenu>
          <template #trigger>
            <BaseButton type="secondary" size="small">Project</BaseButton>
          </template>
          <template #content>
            <a href="#" @click.prevent="emit('new-project')">New Project...</a>
            <a href="#" @click.prevent="emit('new-graph')">New Graph...</a>
            <a href="#" @click.prevent="emit('save-current-graph')">Save Current Graph</a>
          </template>
        </DropdownMenu>

        <!-- Add Menu -->
        <DropdownMenu>
          <template #trigger>
            <BaseButton type="secondary" size="small">Add</BaseButton>
          </template>
          <template #content>
            <div class="dropdown-section-title">Nodes</div>
            <a v-for="nodeType in basicNodeTypes" :key="nodeType.value" href="#"
              @click.prevent="setAddNodeType(nodeType.value)">
              {{ nodeType.label }}
            </a>
            <div class="dropdown-divider"></div>
            <a href="#" @click.prevent="emit('update:currentMode', 'add-edge')">Add Edge</a>
          </template>
        </DropdownMenu>

        <!-- View Menu -->
        <DropdownMenu>
          <template #trigger>
            <BaseButton type="secondary" size="small">View</BaseButton>
          </template>
          <template #content>
            <label class="dropdown-checkbox" @click.stop>
              <input type="checkbox" :checked="isGridEnabled"
                @change="emit('update:isGridEnabled', ($event.target as HTMLInputElement).checked)" />
              Show Grid
            </label>
            <div class="dropdown-input-group" @click.stop>
              <label for="grid-size-nav">Grid Size:</label>
              <BaseInput id="grid-size-nav" type="number" :model-value="gridSize" @input="handleGridSizeInput" min="10"
                max="100" step="5" class="w-20" />
              <span>px</span>
            </div>
          </template>
        </DropdownMenu>

        <!-- Help Menu -->
        <DropdownMenu>
          <template #trigger>
            <BaseButton type="secondary" size="small">Help</BaseButton>
          </template>
          <template #content>
            <a href="#" @click.prevent="emit('open-about-modal')">About DoodleBUGS</a>
            <span class="dropdown-item-placeholder">Documentation (Coming Soon)</span>
            <a href="https://github.com/TuringLang/JuliaBUGS.jl/issues" target="_blank" rel="noopener noreferrer">Report
              an Issue</a>
          </template>
        </DropdownMenu>
      </div>
    </div>

    <!-- Right Section -->
    <div class="navbar-right">
      <div class="pane-toggles">
        <button @click="emit('toggle-left-sidebar')" :class="{ active: isLeftSidebarOpen }" title="Toggle Left Sidebar">
          <svg viewBox="0 0 64 64" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path
              d="M49.984,56l-35.989,0c-3.309,0 -5.995,-2.686 -5.995,-5.995l0,-36.011c0,-3.308 2.686,-5.995 5.995,-5.995l35.989,0c3.309,0 5.995,2.687 5.995,5.995l0,36.011c0,3.309 -2.686,5.995 -5.995,5.995Zm-25.984,-4.001l0,-39.999l-9.012,0c-1.65,0 -2.989,1.339 -2.989,2.989l0,34.021c0,1.65 1.339,2.989 2.989,2.989l9.012,0Zm24.991,-39.999l-20.991,0l0,39.999l20.991,0c1.65,0 2.989,-1.339 2.989,-2.989l0,-34.021c0,-1.65 -1.339,-2.989 -2.989,-2.989Z">
            </path>
          </svg>
        </button>
        <button @click="emit('toggle-right-sidebar')" :class="{ active: isRightSidebarOpen }"
          title="Toggle Right Sidebar">
          <svg viewBox="0 0 64 64" fill="currentColor" xmlns="http://www.w3.org/2000/svg"
            transform="matrix(-1, 0, 0, 1, 0, 0)">
            <path
              d="M49.984,56l-35.989,0c-3.309,0 -5.995,-2.686 -5.995,-5.995l0,-36.011c0,-3.308 2.686,-5.995 5.995,-5.995l35.989,0c3.309,0 5.995,2.687 5.995,5.995l0,36.011c0,3.309 -2.686,5.995 -5.995,5.995Zm-25.984,-4.001l0,-39.999l-9.012,0c-1.65,0 -2.989,1.339 -2.989,2.989l0,34.021c0,1.65 1.339,2.989 2.989,2.989l9.012,0Zm24.991,-39.999l-20.991,0l0,39.999l20.991,0c1.65,0 2.989,-1.339 2.989,-2.989l0,-34.021c0,-1.65 -1.339,-2.989 -2.989,-2.989Z">
            </path>
          </svg>
        </button>
      </div>
      <span class="project-name">{{ displayTitle }}</span>
    </div>
  </nav>
</template>

<style scoped>
/* Your original Navbar.vue styles are preserved here */
.navbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background-color: var(--color-background-dark);
  color: var(--color-text-light);
  padding: 0 20px;
  height: var(--navbar-height);
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
  z-index: 50;
  flex-shrink: 0;
}

.navbar-left,
.navbar-right {
  display: flex;
  align-items: center;
  gap: 20px;
}

.navbar-brand {
  font-size: 1.3em;
  font-weight: 600;
  color: white;
}

.project-name {
  font-size: 1em;
  color: var(--color-text-light);
  opacity: 0.8;
  white-space: nowrap;
}

.navbar-menu {
  display: flex;
  gap: 5px;
}

.dropdown-content a {
  padding: 10px 15px;
  color: var(--color-text);
  text-decoration: none;
  display: block;
  font-size: 0.9em;
}

.dropdown-content a:hover {
  background-color: var(--color-primary);
  color: white;
}

.dropdown-divider {
  height: 1px;
  background-color: var(--color-border-light);
  margin: 8px 0;
}

.dropdown-section-title {
  padding: 5px 15px;
  font-size: 0.8em;
  color: var(--color-secondary);
  text-transform: uppercase;
  font-weight: 600;
}

.dropdown-checkbox,
.dropdown-input-group {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 15px;
  font-size: 0.9em;
  color: var(--color-text);
  cursor: pointer;
}

.dropdown-checkbox input[type="checkbox"] {
  cursor: pointer;
}

.dropdown-checkbox:hover {
  background-color: var(--color-background-mute);
}

.dropdown-input-group .base-input {
  width: 50px;
}

.dropdown-item-placeholder {
  padding: 10px 15px;
  color: var(--color-secondary);
  font-size: 0.9em;
  cursor: default;
  opacity: 0.7;
}

.pane-toggles {
  display: flex;
  align-items: center;
  gap: 5px;
  border: 1px solid #555;
  border-radius: 5px;
  padding: 2px;
}

.pane-toggles button {
  background-color: transparent;
  border: 1px solid transparent;
  color: var(--color-text-light);
  opacity: 0.7;
  padding: 4px;
  border-radius: 3px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s ease;
}

.pane-toggles button:hover {
  opacity: 1;
  background-color: rgba(255, 255, 255, 0.1);
}

.pane-toggles button.active {
  opacity: 1;
  background-color: var(--color-primary);
  color: white;
}

.pane-toggles button svg {
  width: 18px;
  height: 18px;
}
</style>
