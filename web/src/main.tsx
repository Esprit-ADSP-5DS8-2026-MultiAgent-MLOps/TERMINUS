import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router";
import { Header } from "./components/Header";
import { Agents } from "./pages/Agents";
import { Decisions } from "./pages/Decisions";
import { Lignes } from "./pages/Lignes";
import { LatestPipeline, Pipeline } from "./pages/Pipeline";
import { Registre } from "./pages/Registre";
import { Sante } from "./pages/Sante";
import "./styles.css";

function App() {
  return (
    <BrowserRouter>
      <Header />
      <Routes>
        <Route path="/" element={<Lignes />} />
        <Route path="/pipeline" element={<LatestPipeline />} />
        <Route path="/lignes/:id" element={<Pipeline />} />
        <Route path="/decisions" element={<Decisions />} />
        <Route path="/agents" element={<Agents />} />
        <Route path="/registre" element={<Registre />} />
        <Route path="/sante" element={<Sante />} />
        <Route path="*" element={<main className="wrap page"><p className="empty">Voie inconnue.</p></main>} />
      </Routes>
    </BrowserRouter>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
