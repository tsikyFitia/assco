import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // 🔥 ← ceci est crucial pour Docker
    port: 5173,
    watch: {
      usePolling: true // 🔁 pour que le hot reload fonctionne dans Docker
    }
  }
})
