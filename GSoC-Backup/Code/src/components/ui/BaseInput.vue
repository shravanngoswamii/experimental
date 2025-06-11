<script setup lang="ts">
import { computed } from 'vue';

const props = defineProps<{
  modelValue: string | number; // Value bound via v-model
  type?: string; // Input type (e.g., 'text', 'number', 'email')
  placeholder?: string; // Placeholder text
  disabled?: boolean; // Whether the input is disabled
  readonly?: boolean; // Whether the input is read-only
}>();

const emit = defineEmits(['update:modelValue', 'change', 'input', 'keyup.enter']);

// Compute the input type, defaulting to 'text'
const inputType = computed(() => props.type || 'text');

/**
 * Handles the input event to update the v-model binding.
 * @param event The input event.
 */
const handleInput = (event: Event) => {
  const target = event.target as HTMLInputElement;
  emit('update:modelValue', target.value); // Emit for v-model
  emit('input', event); // Emit original input event
};

/**
 * Handles the change event.
 * @param event The change event.
 */
const handleChange = (event: Event) => {
  emit('change', event); // Emit original change event
};

/**
 * Handles the keyup event, specifically for the Enter key.
 * @param event The keyboard event.
 */
const handleKeyUpEnter = (event: KeyboardEvent) => {
  if (event.key === 'Enter') {
    emit('keyup.enter', event); // Emit custom keyup.enter event
  }
};
</script>

<template>
  <input
    :type="inputType"
    :value="modelValue"
    :placeholder="placeholder"
    :disabled="disabled"
    :readonly="readonly"
    @input="handleInput"
    @change="handleChange"
    @keyup="handleKeyUpEnter"
    class="base-input"
  />
</template>

<style scoped>
.base-input {
  padding: 8px 12px;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  background-color: var(--color-background-soft);
  color: var(--color-text);
  box-sizing: border-box; /* Include padding and border in the element's total width and height */
  font-size: 0.9em;
  transition: border-color 0.2s ease, box-shadow 0.2s ease; /* Smooth transitions */
}

.base-input:focus {
  border-color: var(--color-primary);
  outline: none;
  box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25); /* Focus ring */
}

/* Styles for disabled and read-only states */
.base-input:disabled,
.base-input:readonly {
  background-color: var(--color-background-mute);
  cursor: not-allowed;
  opacity: 0.8;
}

/* Specific styles for number inputs to remove spin buttons */
.base-input[type="number"] {
  -moz-appearance: textfield; /* Firefox specific */
  appearance: textfield; /* Standard property */
}
.base-input[type="number"]::-webkit-outer-spin-button,
.base-input[type="number"]::-webkit-inner-spin-button {
  -webkit-appearance: none;
  margin: 0;
}
</style>
