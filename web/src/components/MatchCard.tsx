import { Link } from "react-router-dom";

import { TeamFormSparkline } from "./TeamFormSparkline";
import { TipEntry } from "./TipEntry";
import type { TipChoice } from "../tips";
import type { AlternativeModels, Prediction, TeamFormHistory } from "../types";

function consensusStatus(
  primaryWinner: "home" | "away",
  alternatives: AlternativeModels | null | undefined,
): "unanimous" | "split" | null {
  if (!alternatives) return null;
  const others = [alternatives.logistic, alternatives.bookmaker].filter(Boolean);
  if (others.length === 0) return null;
  const allAgree = others.every((p) => p!.predictedWinner === primaryWinner);
  return allAgree ? "unanimous" : "split";
}

function formatKickoff(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    weekday: "short",
    day: "numeric",
    month: "short",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function MatchCard({
  prediction,
  season,
  round,
  preview,
  tip,
  savingTip,
  onTip,
  homeForm,
  awayForm,
}: {
  prediction: Prediction;
  season: number;
  round: number;
  preview?: string;
  tip?: TipChoice | null;
  savingTip?: boolean;
  onTip?: (choice: TipChoice) => void;
  homeForm?: TeamFormHistory | null;
  awayForm?: TeamFormHistory | null;
}) {
  const p = prediction;
  const homePct = Math.round(p.homeWinProbability * 100);
  const awayPct = 100 - homePct;
  const winnerName = p.predictedWinner === "home" ? p.home.name : p.away.name;
  const winnerPct = p.predictedWinner === "home" ? homePct : awayPct;
  const isKickedOff = new Date(p.kickoff) <= new Date();

  return (
    <article className="match-card">
      {/*
       * Stretched link covers the full card so clicking anywhere navigates to match detail.
       * Team-name links have position: relative; z-index: 2 so they sit above this overlay.
       */}
      <Link
        to={`/round/${season}/${round}/${p.matchId}`}
        className="match-card-link"
        aria-label={`${p.home.name} vs ${p.away.name} — view match detail`}
      />

      <header className="match-card-head">
        <div className="teams">
          <Link
            to={`/team/${p.home.id}?season=${season}`}
            className="team home team-profile-link"
            aria-label={`View ${p.home.name} team profile`}
          >
            {p.home.name}
          </Link>
          <span className="muted"> vs </span>
          <Link
            to={`/team/${p.away.id}?season=${season}`}
            className="team away team-profile-link"
            aria-label={`View ${p.away.name} team profile`}
          >
            {p.away.name}
          </Link>
        </div>
        <time className="kickoff muted" dateTime={p.kickoff}>
          {formatKickoff(p.kickoff)}
        </time>
      </header>

      {(homeForm || awayForm) && (
        <div className="form-sparklines" aria-hidden="false">
          <TeamFormSparkline
            matches={homeForm?.matches ?? []}
            teamName={p.home.name}
          />
          <TeamFormSparkline
            matches={awayForm?.matches ?? []}
            teamName={p.away.name}
          />
        </div>
      )}

      <div className="prob-bar" role="img" aria-label={`Home win probability ${homePct}%`}>
        <span
          className="prob-bar-home"
          style={{ width: `${homePct}%` }}
          aria-hidden="true"
        />
      </div>
      <div className="prob-labels">
        <span>
          <strong>{p.home.name}</strong> {homePct}%
        </span>
        <span>
          {awayPct}% <strong>{p.away.name}</strong>
        </span>
      </div>

      <p className="pick">
        Pick: <strong>{winnerName}</strong> ({winnerPct}%)
      </p>

      {(() => {
        const status = consensusStatus(p.predictedWinner, p.alternatives);
        if (!status) return null;
        return (
          <span
            className={`consensus-badge consensus-badge--${status}`}
            aria-label={
              status === "unanimous"
                ? "All sources agree on this pick"
                : "Sources disagree on this pick"
            }
          >
            {status === "unanimous" ? "Consensus" : "Split pick"}
          </span>
        );
      })()}

      {preview ? <p className="preview">{preview}</p> : null}

      {onTip != null && (
        <div
          className="tip-entry-wrap"
          onClick={(e) => e.stopPropagation()}
          onKeyDown={(e) => e.stopPropagation()}
          role="none"
        >
          <TipEntry
            homeTeam={p.home.name}
            awayTeam={p.away.name}
            value={tip ?? null}
            locked={isKickedOff}
            saving={savingTip ?? false}
            onTip={onTip}
          />
        </div>
      )}
    </article>
  );
}
