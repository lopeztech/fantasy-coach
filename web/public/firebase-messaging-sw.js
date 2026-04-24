/**
 * Firebase Messaging Service Worker (#172).
 *
 * Handles background push messages when the app tab is not in focus.
 * Must live at /firebase-messaging-sw.js (Firebase SDK hard-codes this path).
 *
 * This file is intentionally NOT processed by Vite — it runs directly in
 * the service worker scope where ES module imports are unavailable in most
 * browsers. We use the compat Firebase SDK (importScripts) to stay on the
 * safe side.
 *
 * Firebase config is duplicated here because service workers cannot access
 * Vite env vars. Values are non-secret (client-side identifiers only).
 * See docs/spa.md "Auth flow" for why this is safe.
 */

importScripts("https://www.gstatic.com/firebasejs/10.12.0/firebase-app-compat.js");
importScripts("https://www.gstatic.com/firebasejs/10.12.0/firebase-messaging-compat.js");

// ---------------------------------------------------------------------------
// Firebase config injection: build-time substitution via firebase.json or
// just falls back gracefully if values are not injected.
// ---------------------------------------------------------------------------

// VITE_FIREBASE_* vars are NOT available here. Platform-infra manages the
// hosting rewrite that serves this file; config values are injected at deploy
// time via firebase.json's `headers` or a hosting function. In development,
// this SW won't be active (FCM requires HTTPS + valid config).
//
// If you need to test locally, manually replace the placeholder values below
// with real config from the Firebase console, then revert before committing.

const firebaseConfig = {
  apiKey: self.__FIREBASE_API_KEY__ || "",
  authDomain: self.__FIREBASE_AUTH_DOMAIN__ || "",
  projectId: self.__FIREBASE_PROJECT_ID__ || "",
  appId: self.__FIREBASE_APP_ID__ || "",
};

if (firebaseConfig.apiKey) {
  firebase.initializeApp(firebaseConfig);
  const messaging = firebase.messaging();

  messaging.onBackgroundMessage((payload) => {
    const title = payload.notification?.title ?? "Fantasy Coach";
    const body = payload.notification?.body ?? "New update available";
    const data = payload.data ?? {};

    const options = {
      body,
      icon: "/icons/icon-192.svg",
      badge: "/icons/icon-192.svg",
      data,
      actions: data.action_url
        ? [{ action: "open", title: "View" }]
        : [],
    };

    self.registration.showNotification(title, options);
  });

  self.addEventListener("notificationclick", (event) => {
    event.notification.close();
    const url = event.notification.data?.action_url || "/";
    event.waitUntil(
      clients.matchAll({ type: "window", includeUncontrolled: true }).then((clientList) => {
        for (const client of clientList) {
          if (client.url.includes(url) && "focus" in client) {
            return client.focus();
          }
        }
        if (clients.openWindow) {
          return clients.openWindow(url);
        }
      }),
    );
  });
}
