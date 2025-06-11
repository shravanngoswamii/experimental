import { defineStore } from 'pinia';
import { ref, computed, watch } from 'vue';
import { useGraphStore } from './graphStore'; // Import the graph store

// Interface for graph metadata (stored within a project)
export interface GraphMeta {
  id: string;
  name: string;
  createdAt: number;
  lastModified: number;
}

// Interface for a project
export interface Project {
  id: string;
  name: string;
  createdAt: number;
  lastModified: number;
  graphs: GraphMeta[]; // List of graph metadata belonging to this project
}

export const useProjectStore = defineStore('project', () => {
  const projects = ref<Project[]>([]); // Reactive array of all projects
  const currentProjectId = ref<string | null>(null); // ID of the currently selected project

  const graphStore = useGraphStore(); // Get instance of the graph store

  /**
   * Creates a new project and adds it to the list.
   * @param {string} name - The name of the new project.
   */
  const createProject = (name: string) => {
    const newProject: Project = {
      id: `project_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`, // Unique ID
      name: name,
      createdAt: Date.now(),
      lastModified: Date.now(),
      graphs: [], // New projects start with no graphs
    };
    projects.value.push(newProject); // Add to reactive array
    saveProjects(); // Persist changes
    selectProject(newProject.id); // Automatically select the new project
  };

  /**
   * Sets the currently active project.
   * Automatically selects the first graph in the project if available.
   * @param {string | null} projectId - The ID of the project to select, or null to deselect.
   */
  const selectProject = (projectId: string | null) => {
    currentProjectId.value = projectId;
    if (projectId) {
      const project = projects.value.find(p => p.id === projectId);
      if (project && project.graphs.length > 0) {
        // Select the first graph in the project, or last opened (future enhancement)
        graphStore.selectGraph(project.graphs[0].id);
      } else {
        graphStore.selectGraph(null); // No graphs in this project, deselect current graph
      }
    } else {
      graphStore.selectGraph(null); // No project selected, deselect current graph
    }
  };

  /**
   * Deletes a project and all its associated graphs.
   * @param {string} projectId - The ID of the project to delete.
   */
  const deleteProject = (projectId: string) => {
    const projectToDelete = projects.value.find(p => p.id === projectId);
    if (projectToDelete) {
      // Delete all graphs associated with this project from the graph store
      projectToDelete.graphs.forEach(graphMeta => {
        graphStore.deleteGraphContent(graphMeta.id);
      });
      // Remove the project from the reactive array
      projects.value = projects.value.filter(p => p.id !== projectId);
      // If the deleted project was current, deselect it
      if (currentProjectId.value === projectId) {
        currentProjectId.value = null;
        graphStore.selectGraph(null); // Also deselect any active graph
      }
      saveProjects(); // Persist changes
    }
  };

  /**
   * Computed property for the currently selected project object.
   */
  const currentProject = computed(() => {
    return projects.value.find(p => p.id === currentProjectId.value) || null;
  });

  /**
   * Adds a new graph (metadata) to a specified project.
   * Also creates an empty content entry for this new graph in the graph store.
   * @param {string} projectId - The ID of the project to add the graph to.
   * @param {string} graphName - The name of the new graph.
   */
  const addGraphToProject = (projectId: string, graphName: string) => {
    const project = projects.value.find(p => p.id === projectId);
    if (project) {
      const newGraphMeta: GraphMeta = {
        id: `graph_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`, // Unique ID
        name: graphName,
        createdAt: Date.now(),
        lastModified: Date.now(),
      };
      project.graphs.push(newGraphMeta); // Add metadata to project
      project.lastModified = Date.now(); // Update project modification timestamp
      saveProjects(); // Persist changes
      graphStore.createNewGraphContent(newGraphMeta.id); // Create empty content for new graph
      graphStore.selectGraph(newGraphMeta.id); // Select the new graph
    }
  };

  /**
   * Deletes a graph from a specified project and its content from the graph store.
   * @param {string} projectId - The ID of the project containing the graph.
   * @param {string} graphId - The ID of the graph to delete.
   */
  const deleteGraphFromProject = (projectId: string, graphId: string) => {
    const project = projects.value.find(p => p.id === projectId);
    if (project) {
      project.graphs = project.graphs.filter(g => g.id !== graphId); // Remove metadata
      project.lastModified = Date.now(); // Update project modification timestamp
      saveProjects(); // Persist changes
      graphStore.deleteGraphContent(graphId); // Delete graph content as well
    }
  };

  /**
   * Returns the list of graph metadata for a given project.
   * @param {string} projectId - The ID of the project.
   * @returns {GraphMeta[]} An array of graph metadata.
   */
  const getGraphsForProject = (projectId: string): GraphMeta[] => {
    return projects.value.find(p => p.id === projectId)?.graphs || [];
  };

  // --- Persistence Methods (Local Storage) ---

  /**
   * Saves the entire projects array to local storage.
   */
  const saveProjects = () => {
    localStorage.setItem('doodlebugs-projects', JSON.stringify(projects.value));
  };

  /**
   * Loads projects from local storage into the store.
   */
  const loadProjects = () => {
    const storedProjects = localStorage.getItem('doodlebugs-projects');
    if (storedProjects) {
      projects.value = JSON.parse(storedProjects);
    }
  };

  // Watch for changes in current project and update graph store accordingly
  watch(currentProjectId, (newProjectId) => {
    if (newProjectId) {
      const project = projects.value.find(p => p.id === newProjectId);
      if (project && project.graphs.length > 0) {
        // If a project is selected and it has graphs, ensure one is selected in graphStore
        if (!graphStore.currentGraphId || !project.graphs.some(g => g.id === graphStore.currentGraphId)) {
          graphStore.selectGraph(project.graphs[0].id); // Select the first graph by default
        }
      } else {
        graphStore.selectGraph(null); // If no graphs in project, deselect graph
      }
    } else {
      graphStore.selectGraph(null); // If no project selected, deselect graph
    }
  }, { immediate: true }); // Run immediately on component mount

  return {
    projects,
    currentProjectId,
    currentProject,
    createProject,
    selectProject,
    deleteProject,
    addGraphToProject,
    deleteGraphFromProject,
    getGraphsForProject,
    loadProjects,
  };
});
