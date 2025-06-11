<script setup lang="ts">
import { ref, watch, computed } from 'vue';
import type { StyleValue } from 'vue';
import GraphEditor from '../canvas/GraphEditor.vue';
import ProjectManager from '../left-sidebar/ProjectManager.vue';
import NodePalette from '../left-sidebar/NodePalette.vue';
import DataInputPanel from '../panels/DataInputPanel.vue';
import NodePropertiesPanel from '../right-sidebar/NodePropertiesPanel.vue';
import CodePreviewPanel from '../panels/CodePreviewPanel.vue';
import TheNavbar from './TheNavbar.vue'; // UPDATED
import BaseModal from '../common/BaseModal.vue';
import BaseInput from '../ui/BaseInput.vue';
import BaseButton from '../ui/BaseButton.vue';
import AboutModal from './AboutModal.vue';

import { useGraphElements } from '../../composables/useGraphElements';
import { useProjectStore } from '../../stores/projectStore';
import { useGraphStore } from '../../stores/graphStore';
import type { GraphElement, NodeType, PaletteItemType } from '../../types';

const projectStore = useProjectStore();
const graphStore = useGraphStore();
const { selectedElement, updateElement, deleteElement } = useGraphElements();

const activeLeftTab = ref<'project' | 'palette' | 'data' | null>('project');
const activeRightTab = ref<'properties' | 'code'>('properties');
const isLeftSidebarOpen = ref(true);
const isRightSidebarOpen = ref(true);
const currentMode = ref<string>('select');
const currentNodeType = ref<NodeType>('stochastic');
const isGridEnabled = ref(true);
const gridSize = ref(20);

const showNewProjectModal = ref(false);
const newProjectName = ref('');
const showNewGraphModal = ref(false);
const newGraphName = ref('');
const showAboutModal = ref(false);

const currentProjectName = computed(() => projectStore.currentProject?.name || null);
const activeGraphName = computed(() => {
  if (projectStore.currentProject && graphStore.currentGraphId) {
    const graphMeta = projectStore.currentProject.graphs.find(g => g.id === graphStore.currentGraphId);
    return graphMeta?.name || null;
  }
  return null;
});

const handleLeftTabClick = (tabName: 'project' | 'palette' | 'data') => {
  if (activeLeftTab.value === tabName && isLeftSidebarOpen.value) {
    isLeftSidebarOpen.value = false;
  } else {
    isLeftSidebarOpen.value = true;
    activeLeftTab.value = tabName;
  }
};

const toggleLeftSidebar = () => {
  isLeftSidebarOpen.value = !isLeftSidebarOpen.value;
};

const toggleRightSidebar = () => {
  isRightSidebarOpen.value = !isRightSidebarOpen.value;
};

const leftSidebarContentStyle = computed((): StyleValue => ({
  width: isLeftSidebarOpen.value ? 'var(--sidebar-content-width-left)' : '0',
  opacity: isLeftSidebarOpen.value ? '1' : '0',
  pointerEvents: isLeftSidebarOpen.value ? 'auto' : 'none',
}));
const leftSidebarClass = computed(() => ({
  'left-sidebar': true,
  'sidebar-collapsed-content': !isLeftSidebarOpen.value,
}));
const rightSidebarStyle = computed((): StyleValue => ({
  width: isRightSidebarOpen.value ? 'var(--sidebar-width-right)' : '0',
  opacity: isRightSidebarOpen.value ? '1' : '0',
  pointerEvents: isRightSidebarOpen.value ? 'auto' : 'none',
  borderLeft: isRightSidebarOpen.value ? '1px solid var(--color-border)' : 'none',
}));

const handleElementSelected = (element: GraphElement | null) => {
  selectedElement.value = element;
};

const handleUpdateElement = (updatedEl: GraphElement) => {
  updateElement(updatedEl);
};

const handleDeleteElement = (elementId: string) => {
  deleteElement(elementId);
};

const handlePaletteSelection = (itemType: PaletteItemType) => {
  if (itemType === 'add-stochastic-edge' || itemType === 'add-deterministic-edge') {
    currentMode.value = 'add-edge';
  } else {
    currentMode.value = 'add-node';
    currentNodeType.value = itemType as NodeType;
  }
  isLeftSidebarOpen.value = false;
};

const createNewProject = () => {
  if (newProjectName.value.trim()) {
    projectStore.createProject(newProjectName.value.trim());
    showNewProjectModal.value = false;
    newProjectName.value = '';
    activeLeftTab.value = 'project';
    isLeftSidebarOpen.value = true;
  }
};

const createNewGraph = () => {
  if (projectStore.currentProject && newGraphName.value.trim()) {
    projectStore.addGraphToProject(projectStore.currentProject.id, newGraphName.value.trim());
    showNewGraphModal.value = false;
    newGraphName.value = '';
    activeLeftTab.value = 'project';
    isLeftSidebarOpen.value = true;
  }
};

const saveCurrentGraph = () => {
  if (graphStore.currentGraphId) {
    graphStore.saveGraph(graphStore.currentGraphId, graphStore.graphContents.get(graphStore.currentGraphId)!);
    console.log(`Graph "${graphStore.currentGraphId}" saved.`);
  } else {
    console.warn("No graph currently selected to save.");
  }
};

watch(selectedElement, (newVal) => {
  console.log('Selected element changed in MainLayout:', newVal);
}, { deep: true });
</script>

<template>
  <div class="main-layout">
    <!-- UPDATED to use TheNavbar -->
    <TheNavbar :project-name="currentProjectName" :active-graph-name="activeGraphName" :is-grid-enabled="isGridEnabled"
      @update:is-grid-enabled="isGridEnabled = $event" :grid-size="gridSize" @update:grid-size="gridSize = $event"
      :current-mode="currentMode" @update:current-mode="currentMode = $event" :current-node-type="currentNodeType"
      @update:current-node-type="currentNodeType = $event" :is-left-sidebar-open="isLeftSidebarOpen"
      :is-right-sidebar-open="isRightSidebarOpen" @toggle-left-sidebar="toggleLeftSidebar"
      @toggle-right-sidebar="toggleRightSidebar" @new-project="showNewProjectModal = true"
      @new-graph="showNewGraphModal = true" @save-current-graph="saveCurrentGraph"
      @open-about-modal="showAboutModal = true" />

    <div class="content-area">
      <aside :class="leftSidebarClass">
        <div class="vertical-tabs-container">
          <button :class="{ active: activeLeftTab === 'project' }" @click="handleLeftTabClick('project')"
            title="Project Manager">
            <i class="fas fa-folder"></i> <span v-show="isLeftSidebarOpen">Project</span>
          </button>
          <button :class="{ active: activeLeftTab === 'palette' }" @click="handleLeftTabClick('palette')"
            title="Node Palette">
            <i class="fas fa-shapes"></i> <span v-show="isLeftSidebarOpen">Palette</span>
          </button>
          <button :class="{ active: activeLeftTab === 'data' }" @click="handleLeftTabClick('data')" title="Data Input">
            <i class="fas fa-database"></i> <span v-show="isLeftSidebarOpen">Data</span>
          </button>
        </div>
        <div class="left-sidebar-content" :style="leftSidebarContentStyle">
          <div v-show="activeLeftTab === 'project'">
            <ProjectManager @new-project="showNewProjectModal = true" @new-graph="showNewGraphModal = true" />
          </div>
          <div v-show="activeLeftTab === 'palette'">
            <NodePalette @select-palette-item="handlePaletteSelection" />
          </div>
          <div v-show="activeLeftTab === 'data'">
            <DataInputPanel />
          </div>
        </div>
      </aside>
      <main class="graph-editor-wrapper">
        <GraphEditor :is-grid-enabled="isGridEnabled" :grid-size="gridSize" :current-mode="currentMode"
          :elements="graphStore.currentGraphElements" :current-node-type="currentNodeType"
          @update:current-mode="currentMode = $event" @update:current-node-type="currentNodeType = $event"
          @element-selected="handleElementSelected" />
      </main>
      <aside class="right-sidebar" :style="rightSidebarStyle">
        <div class="tabs-header">
          <button :class="{ active: activeRightTab === 'properties' }"
            @click="activeRightTab = 'properties'">Properties</button>
          <button :class="{ active: activeRightTab === 'code' }" @click="activeRightTab = 'code'">Code</button>
        </div>
        <div class="tabs-content">
          <div v-show="activeRightTab === 'properties'">
            <NodePropertiesPanel :selected-element="selectedElement" @update-element="handleUpdateElement"
              @delete-element="handleDeleteElement" />
          </div>
          <div v-show="activeRightTab === 'code'">
            <CodePreviewPanel />
          </div>
        </div>
      </aside>
    </div>

    <BaseModal :is-open="showNewProjectModal" @close="showNewProjectModal = false">
      <template #header>
        <h3>Create New Project</h3>
      </template>
      <template #body>
        <label for="new-project-name" style="display: block; margin-bottom: 8px; font-weight: 500;">Project
          Name:</label>
        <BaseInput id="new-project-name" v-model="newProjectName" placeholder="Enter project name"
          @keyup.enter="createNewProject" />
      </template>
      <template #footer>
        <BaseButton @click="showNewProjectModal = false" type="secondary">Cancel</BaseButton>
        <BaseButton @click="createNewProject" type="primary">Create</BaseButton>
      </template>
    </BaseModal>
    <BaseModal :is-open="showNewGraphModal" @close="showNewGraphModal = false">
      <template #header>
        <h3>Create New Graph</h3>
      </template>
      <template #body>
        <label for="new-graph-name" style="display: block; margin-bottom: 8px; font-weight: 500;">Graph Name:</label>
        <BaseInput id="new-graph-name" v-model="newGraphName" placeholder="Enter graph name"
          @keyup.enter="createNewGraph" />
      </template>
      <template #footer>
        <BaseButton @click="showNewGraphModal = false" type="secondary">Cancel</BaseButton>
        <BaseButton @click="createNewGraph" type="primary">Create</BaseButton>
      </template>
    </BaseModal>
    <AboutModal :is-open="showAboutModal" @close="showAboutModal = false" />
  </div>
</template>

<style scoped>
/* Your original MainLayout.vue styles are preserved here */
.main-layout {
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
}

.content-area {
  display: flex;
  flex-grow: 1;
  overflow: hidden;
}

.left-sidebar {
  display: flex;
  background-color: var(--color-background-soft);
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
  z-index: 10;
  flex-shrink: 0;
  border-right: 1px solid var(--color-border);
  transition: width 0.3s ease-in-out;
}

.left-sidebar:not(.sidebar-collapsed-content) {
  width: calc(var(--vertical-tab-width) + var(--sidebar-content-width-left));
}

.left-sidebar.sidebar-collapsed-content {
  width: var(--vertical-tab-width);
}

.vertical-tabs-container {
  display: flex;
  flex-direction: column;
  width: var(--vertical-tab-width);
  border-right: 1px solid var(--color-border-light);
  background-color: var(--color-background-dark);
  padding-top: 10px;
  flex-shrink: 0;
}

.vertical-tabs-container button {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  width: 100%;
  padding: 10px 0;
  border: none;
  background-color: transparent;
  color: var(--color-text-light);
  font-size: 0.75em;
  font-weight: 500;
  transition: all 0.2s ease;
  gap: 5px;
  cursor: pointer;
  white-space: nowrap;
}

.vertical-tabs-container button i {
  font-size: 1.3em;
  color: var(--color-secondary);
  transition: color 0.2s ease;
}

.vertical-tabs-container button:hover {
  background-color: var(--color-primary-hover);
  color: white;
}

.vertical-tabs-container button:hover i {
  color: white;
}

.vertical-tabs-container button.active {
  background-color: var(--color-primary);
  color: white;
  border-left: 2px solid white;
}

.vertical-tabs-container button.active i {
  color: white;
}

.left-sidebar-content {
  flex-grow: 1;
  overflow-y: auto;
  padding: 15px;
  -webkit-overflow-scrolling: touch;
  transition: width 0.3s ease-in-out, opacity 0.3s ease-in-out;
  box-sizing: border-box;
}

.right-sidebar {
  display: flex;
  flex-direction: column;
  background-color: var(--color-background-soft);
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
  z-index: 10;
  flex-shrink: 0;
  transition: width 0.3s ease-in-out, opacity 0.3s ease-in-out;
}

.tabs-header {
  display: flex;
  border-bottom: 1px solid var(--color-border-light);
  flex-shrink: 0;
}

.tabs-header button {
  flex: 1;
  padding: 10px 15px;
  border: none;
  background-color: transparent;
  border-bottom: 2px solid transparent;
  font-weight: 500;
  color: var(--color-text);
  transition: all 0.2s ease;
  white-space: nowrap;
}

.tabs-header button:hover {
  background-color: var(--color-background-mute);
}

.tabs-header button.active {
  color: var(--color-primary);
  border-bottom-color: var(--color-primary);
  background-color: var(--color-background-soft);
}

.tabs-content {
  flex-grow: 1;
  overflow-y: auto;
  padding: 15px;
  -webkit-overflow-scrolling: touch;
}

.graph-editor-wrapper {
  flex-grow: 1;
  display: flex;
  flex-direction: column;
  position: relative;
  background-color: var(--color-background-mute);
  min-width: 0;
}
</style>
