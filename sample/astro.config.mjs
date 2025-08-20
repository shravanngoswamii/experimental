// @ts-check
import { defineConfig } from 'astro/config';

/**
 * Astro configuration file.
 * @see https://astro.build/config
 */
export default defineConfig({
  // The final domain where your site will be deployed.
  // This is crucial for generating correct canonical URLs and sitemaps.
  site: 'https://shravangoswami.com',

  // The subdirectory your site lives in.
  // This must match the repository name and any subfolders in your deployment.
  base: '/experimental/washer/',
});