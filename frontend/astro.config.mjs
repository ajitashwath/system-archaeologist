// @ts-check
import { defineConfig } from 'astro/config';

// https://astro.build/config
export default defineConfig({
  output: 'static',
  server: {
    port: 4321,
  },
  vite: {
    define: {
      'import.meta.env.PUBLIC_API_BASE': JSON.stringify(
        process.env.PUBLIC_API_BASE || 'http://localhost:8000'
      ),
    },
  },
});
