import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: '/simple_agent_tracer/',
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8781',
      '/ws': { target: 'ws://localhost:8781', ws: true },
    },
  },
});
