<script setup lang="ts">
import { computed } from 'vue';

// Interface for select options
interface SelectOption {
  value: string;
  label: string;
}

const props = defineProps<{
  modelValue: string; // The currently selected value (v-model binding)
  options: SelectOption[]; // Array of options to display
  disabled?: boolean; // Whether the select is disabled
}>();

const emit = defineEmits(['update:modelValue', 'change']);

/**
 * Handles the change event of the select element.
 * Emits the new value for v-model and the original change event.
 * @param event The change event.
 */
const handleChange = (event: Event) => {
  const target = event.target as HTMLSelectElement;
  emit('update:modelValue', target.value); // Emit for v-model
  emit('change', event); // Emit original change event
};

// Compute classes for styling, including disabled state
const selectClass = computed(() => ({
  'base-select': true,
  'select-disabled': props.disabled
}));
</script>

<template>
  <select
    :value="modelValue"
    @change="handleChange"
    :disabled="disabled"
    :class="selectClass"
  >
    <option v-for="option in options" :key="option.value" :value="option.value">
      {{ option.label }}
    </option>
  </select>
</template>

<style scoped>
.base-select {
  padding: 8px 12px;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  background-color: var(--color-background-soft);
  color: var(--color-text);
  box-sizing: border-box;
  font-size: 0.9em;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
  /* Remove default browser styling for a custom look */
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  /* Custom arrow icon using SVG data URI */
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' fill='%236c757d'%3E%3Cpath fill-rule='evenodd' d='M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z' clip-rule='evenodd'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 8px center;
  background-size: 1.2em;
  cursor: pointer;
}

.base-select:focus {
  border-color: var(--color-primary);
  outline: none;
  box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
}

.base-select:disabled {
  background-color: var(--color-background-mute);
  cursor: not-allowed;
  opacity: 0.8;
}
</style>
