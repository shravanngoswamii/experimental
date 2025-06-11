<script setup lang="ts">
import MainLayout from './components/layouts/MainLayout.vue'; // Import the main layout component
</script>

<template>
  <!--
    The App.vue component now primarily serves as the entry point for the Vue application.
    It imports and renders the MainLayout component, which defines the overall
    structure of the DoodleBUGS editor (sidebars, canvas).
    All global styling has been moved to src/assets/styles/global.css.
  -->
  <MainLayout />
</template>

<!--
  No <style> block here. All global styles are now in src/assets/styles/global.css.
  Component-specific styles will be defined within their respective .vue files
  using <style scoped>.
-->
