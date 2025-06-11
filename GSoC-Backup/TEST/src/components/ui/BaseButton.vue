<script setup lang="ts">
import { computed } from 'vue';

const props = defineProps<{
  type?: 'primary' | 'secondary' | 'danger'; // Button style type
  size?: 'small' | 'medium' | 'large'; // Button size
  disabled?: boolean; // Whether the button is disabled
}>();

// Dynamically compute CSS classes based on props
const buttonClass = computed(() => {
  return {
    'base-button': true, // Always apply base styles
    [`button-${props.type || 'default'}`]: true, // Apply type-specific styles
    [`button-${props.size || 'medium'}`]: true, // Apply size-specific styles
    'button-disabled': props.disabled // Apply disabled style if prop is true
  };
});
</script>

<template>
  <button :class="buttonClass" :disabled="disabled">
    <slot></slot> </button>
</template>

<style scoped>
/* Base styles for all buttons */
.base-button {
  padding: 8px 15px;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  background-color: var(--color-background-mute);
  color: var(--color-text);
  cursor: pointer;
  transition: all 0.2s ease; /* Smooth transitions for hover/focus states */
  font-weight: 500;
  white-space: nowrap; /* Prevent text from wrapping inside the button */
}

/* Hover state for non-disabled buttons */
.base-button:hover:not(.button-disabled) {
  background-color: var(--color-border-light);
}

/* Focus state for accessibility */
.base-button:focus {
  outline: none;
  box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25); /* Focus ring */
}

/* Primary button styles */
.button-primary {
  background-color: var(--color-primary);
  color: white;
  border-color: var(--color-primary);
}

.button-primary:hover:not(.button-disabled) {
  background-color: var(--color-primary-hover);
  border-color: var(--color-primary-hover);
}

/* Secondary button styles */
.button-secondary {
  background-color: var(--color-secondary);
  color: white;
  border-color: var(--color-secondary);
}

.button-secondary:hover:not(.button-disabled) {
  background-color: var(--color-secondary-hover);
  border-color: var(--color-secondary-hover);
}

/* Danger button styles (for destructive actions) */
.button-danger {
  background-color: var(--color-danger);
  color: white;
  border-color: var(--color-danger);
}

.button-danger:hover:not(.button-disabled) {
  background-color: #c82333; /* Darker red */
  border-color: #bd2130;
}

/* Sizes */
.button-small {
  padding: 6px 10px;
  font-size: 0.8em;
}

.button-medium {
  padding: 8px 15px;
  font-size: 0.9em;
}

.button-large {
  padding: 12px 20px;
  font-size: 1.1em;
}

/* Disabled state */
.button-disabled {
  opacity: 0.6;
  cursor: not-allowed;
  background-color: var(--color-background-mute) !important;
  color: var(--color-secondary) !important;
  border-color: var(--color-border) !important;
}
</style>
