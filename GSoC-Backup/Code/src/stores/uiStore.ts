import { defineStore } from 'pinia';
import { ref } from 'vue';

export type RightSidebarTab = 'properties' | 'code' | 'json';

export const useUiStore = defineStore('ui', () => {
  // Load initial state from localStorage or set defaults
  const activeRightTab = ref<RightSidebarTab>(
    (localStorage.getItem('doodlebugs-activeRightTab') as RightSidebarTab) || 'properties'
  );
  const isRightTabPinned = ref<boolean>(
    localStorage.getItem('doodlebugs-isRightTabPinned') === 'true'
  );

  /**
   * Sets the active tab for the right sidebar and persists it to localStorage.
   * @param tab The tab to activate.
   */
  const setActiveRightTab = (tab: RightSidebarTab) => {
    activeRightTab.value = tab;
    localStorage.setItem('doodlebugs-activeRightTab', tab);
  };

  /**
   * Toggles the pinned state of the right sidebar tab and persists it.
   */
  const toggleRightTabPinned = () => {
    isRightTabPinned.value = !isRightTabPinned.value;
    localStorage.setItem('doodlebugs-isRightTabPinned', isRightTabPinned.value.toString());
  };

  return {
    activeRightTab,
    isRightTabPinned,
    setActiveRightTab,
    toggleRightTabPinned,
  };
});
