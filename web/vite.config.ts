import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";

export default defineConfig({
  plugins: [
    react(),
    // vite-plugin-pwa chosen over a hand-rolled service worker because it
    // integrates with Vite's asset pipeline, handles precache manifest
    // versioning automatically (each build SHA produces a unique sw.js), and
    // exposes Workbox's runtime-caching API without boilerplate.
    VitePWA({
      registerType: "autoUpdate",
      manifest: {
        name: "Fantasy Coach",
        short_name: "Fantasy Coach",
        description:
          "NRL match predictions — win probabilities and feature explanations for every round.",
        theme_color: "#1a1a1a",
        background_color: "#fafafa",
        display: "standalone",
        start_url: "/",
        icons: [
          {
            src: "/icons/icon-192.svg",
            sizes: "192x192",
            type: "image/svg+xml",
          },
          {
            src: "/icons/icon-512.svg",
            sizes: "512x512",
            type: "image/svg+xml",
            purpose: "any maskable",
          },
        ],
      },
      workbox: {
        // Activate the new SW on first reload after deploy. Without these the
        // SW installs but stays "waiting" until every tab of the origin closes,
        // which surfaces as users seeing the previous build for hours/days
        // after a deploy.
        skipWaiting: true,
        clientsClaim: true,
        // Precache all built JS, CSS, HTML, and image assets.
        globPatterns: ["**/*.{js,css,html,svg,png,ico,woff2}"],
        runtimeCaching: [
          {
            // Network-first + stale-while-revalidate for the predictions API.
            // Matches both same-origin /predictions (dev proxy) and the
            // cross-origin Cloud Run URL (production).
            urlPattern: ({ url }) => url.pathname.startsWith("/predictions"),
            handler: "NetworkFirst",
            options: {
              cacheName: "predictions-v1",
              networkTimeoutSeconds: 10,
              cacheableResponse: { statuses: [0, 200] },
            },
          },
        ],
      },
    }),
  ],
  server: {
    port: 5173,
  },
});
