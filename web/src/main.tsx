import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";

import Accuracy from "./routes/Accuracy";
import App from "./App";
import Home from "./routes/Home";
import MatchDetail from "./routes/MatchDetail";
import Round from "./routes/Round";
import Scoreboard from "./routes/Scoreboard";
import TeamProfileRoute from "./routes/TeamProfile";
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
      { path: "accuracy", element: <Accuracy /> },
      { path: "team/:teamId", element: <TeamProfileRoute /> },
    ],
  },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
