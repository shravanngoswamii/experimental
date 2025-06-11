<script setup lang="ts">
import BaseButton from '../ui/BaseButton.vue'; // Updated import path
// This component will display the generated BUGS code.
// For now, it's a placeholder. Later it will:
// - Take the graph elements and generate BUGS code
// - Display the code in a syntax-highlighted editor
// - Possibly allow copying the code.
</script>

<template>
  <div class="code-preview-panel">
    <h4>Generated BUGS Code</h4>
    <div class="code-output">
      <p>BUGS model code will be generated and displayed here.</p>
      <pre class="code-block">
        <code>
model {
  # Example: Define a normal distribution
  # mu ~ dnorm(0, 1.0E-6)
  # tau ~ dgamma(0.01, 0.01)

  # Example: A simple deterministic relationship
  # theta <- exp(mu) / (1 + exp(mu))

  # Example: A loop for indexed variables (Plate)
  # for (i in 1:N) {
  #   y[i] ~ dbin(theta[i], n[i])
  # }

  # Add your nodes and connections visually to generate code!
}
        </code>
      </pre>
      <BaseButton class="copy-button" type="secondary">Copy Code</BaseButton>
    </div>
  </div>
</template>

<style scoped>
.code-preview-panel {
  padding: 15px;
  height: 100%;
  display: flex;
  flex-direction: column;
}

h4 {
  margin: 0 0 10px 0;
  color: var(--color-heading);
  text-align: center;
  border-bottom: 1px solid var(--color-border-light);
  padding-bottom: 10px;
}

.code-output {
  flex-grow: 1;
  display: flex;
  flex-direction: column;
  background-color: #282c34; /* Dark background for code */
  color: #abb2bf; /* Light text for code */
  border-radius: 8px;
  overflow: hidden;
  position: relative;
}

.code-output p {
  padding: 15px;
  margin: 0;
  font-family: 'Fira Code', 'Cascadia Code', monospace;
  font-size: 0.9em;
  color: #61afef; /* A softer blue for info text */
}

.code-block {
  flex-grow: 1;
  padding: 15px;
  margin: 0;
  overflow: auto; /* Enable scrolling for code */
  font-family: 'Fira Code', 'Cascadia Code', monospace;
  font-size: 0.85em;
  line-height: 1.4;
  white-space: pre-wrap; /* Preserve whitespace and wrap */
  word-break: break-all; /* Break long words */
}

.copy-button {
  position: sticky; /* Sticky to the bottom of the code-output */
  bottom: 0;
  align-self: flex-end; /* Align to the right */
  margin: 10px;
  padding: 8px 15px;
  background-color: var(--color-secondary);
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  z-index: 10;
  opacity: 0.9;
}

.copy-button:hover {
  opacity: 1;
  background-color: var(--color-secondary-hover);
}
</style>
