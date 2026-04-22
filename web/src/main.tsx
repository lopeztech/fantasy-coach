import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";

import App from "./App";
import Home from "./routes/Home";
import MatchDetail from "./routes/MatchDetail";
import Round from "./routes/Round";
import Scoreboard from "./routes/Scoreboard";
import "./styles.css";

const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
    children: [
      { index: true, element: <Home /> },
      { path: "round/:season/:round", element: <Round /> },
      { path: "round/:season/:round/:matchId", element: <MatchDetail /> },
      { path: "scoreboard", element: <Scoreboard /> },
    ],
  },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
