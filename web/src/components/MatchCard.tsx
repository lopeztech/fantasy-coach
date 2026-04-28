import { Link, useNavigate } from "react-router-dom";

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

/**
 * Compact NRL-draw-style kickoff time, e.g. "6:00pm". Used in the centred
 * column of the match card. Falls back to formatKickoff() for non-times.
 */
function formatKickoffTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d
    .toLocaleString(undefined, { hour: "numeric", minute: "2-digit", hour12: true })
    .replace(/\s/g, "")
    .toLowerCase();
}

function formatOdds(odds: number | null | undefined): string {
  if (odds == null || !Number.isFinite(odds)) return "—";
  return `$${odds.toFixed(2)}`;
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
  const navigate = useNavigate();
  const p = prediction;
  const homePct = Math.round(p.homeWinProbability * 100);
  const awayPct = 100 - homePct;
  const winnerName = p.predictedWinner === "home" ? p.home.name : p.away.name;
  const winnerPct = p.predictedWinner === "home" ? homePct : awayPct;
  const isKickedOff = new Date(p.kickoff) <= new Date();

  const venueLine = [p.venue, p.venueCity].filter(Boolean).join(", ");

  return (
    <Link
      to={`/round/${season}/${round}/${p.matchId}`}
      className="match-card-link"
      aria-label={`${p.home.name} vs ${p.away.name} — view why`}
    >
      <article className="match-card">
        <header className="match-card-head">
          <div
            className="match-team match-team--home"
            role="link"
            tabIndex={0}
            onClick={(e) => { e.preventDefault(); navigate(`/team/${p.home.id}`); }}
            onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); navigate(`/team/${p.home.id}`); } }}
          >
            <span className="match-team-name">{p.home.name}</span>
          </div>
          <div className="match-time-cell">
            <time className="match-kickoff-time" dateTime={p.kickoff}>
              {formatKickoffTime(p.kickoff)}
            </time>
            <span className="match-kickoff-date muted">{formatKickoff(p.kickoff)}</span>
          </div>
          <div
            className="match-team match-team--away"
            role="link"
            tabIndex={0}
            onClick={(e) => { e.preventDefault(); navigate(`/team/${p.away.id}`); }}
            onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); navigate(`/team/${p.away.id}`); } }}
          >
            <span className="match-team-name">{p.away.name}</span>
          </div>
        </header>

        {(p.home.odds != null || p.away.odds != null) && (
          <div className="match-odds-row" aria-label="Bookmaker odds">
            <span className="match-odds-pill">{formatOdds(p.home.odds)}</span>
            <span className="match-odds-source">odds</span>
            <span className="match-odds-pill">{formatOdds(p.away.odds)}</span>
          </div>
        )}

        {venueLine && (
          <p className="match-venue muted">{venueLine}</p>
        )}

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
            onClick={(e) => e.preventDefault()}
            onKeyDown={(e) => e.preventDefault()}
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
    </Link>
  );
}
