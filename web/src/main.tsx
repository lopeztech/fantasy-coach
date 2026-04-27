import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";

import Accuracy from "./routes/Accuracy";
import App from "./App";
import GroupDetail from "./routes/GroupDetail";
import Groups from "./routes/Groups";
import Home from "./routes/Home";
import Ladder from "./routes/Ladder";
import Leaderboard from "./routes/Leaderboard";
import MatchDetail from "./routes/MatchDetail";
import Round from "./routes/Round";
import Scoreboard from "./routes/Scoreboard";
import Team from "./routes/Team";
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
      { path: "ladder", element: <Ladder /> },
      { path: "team/:teamId", element: <Team /> },
      { path: "leaderboard", element: <Leaderboard /> },
      { path: "groups", element: <Groups /> },
      { path: "groups/:gid", element: <GroupDetail /> },
    ],
  },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
