import { createApp } from 'vue';
import { createPinia } from 'pinia';
import App from './App.vue';
import './assets/styles/global.css'; // Import the consolidated global CSS file

const app = createApp(App);

app.use(createPinia()); // Use Pinia for state management

app.mount('#app'); // Mount the Vue application to the #app element
