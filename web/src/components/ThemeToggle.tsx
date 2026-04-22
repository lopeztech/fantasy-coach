import { useEffect, useState } from "react";

type Theme = "auto" | "light" | "dark";

const STORAGE_KEY = "fc-theme";

function applyTheme(theme: Theme) {
  const root = document.documentElement;
  if (theme === "dark") {
    root.setAttribute("data-theme", "dark");
  } else if (theme === "light") {
    root.setAttribute("data-theme", "light");
  } else {
    root.removeAttribute("data-theme");
  }
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    // localStorage unavailable (private browsing, permissions policy)
  }
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY) as Theme | null;
      if (stored === "dark" || stored === "light" || stored === "auto") return stored;
    } catch {
      // ignore
    }
    return "auto";
  });

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  return (
    <div className="theme-toggle" role="group" aria-label="Colour theme">
      {(["auto", "light", "dark"] as const).map((t) => (
        <button
          key={t}
          type="button"
          className={`theme-btn${theme === t ? " active" : ""}`}
          aria-pressed={theme === t}
          onClick={() => setTheme(t)}
        >
          {t.charAt(0).toUpperCase() + t.slice(1)}
        </button>
      ))}
    </div>
  );
}
