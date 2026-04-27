import { useEffect, useRef, useState } from "react";

const STEPS = [
  {
    title: "This week's round",
    body: "The model's best pick is highlighted for every match. Win probability shows how confident the model is.",
    cta: "Next",
  },
  {
    title: "Dig into the why",
    body: "Tap any match to see which factors drove the prediction — form, Elo, rest days, weather, and more.",
    cta: "Next",
  },
  {
    title: "Tip against the model",
    body: "Pick your own winner for every match and track your accuracy against the model over the season.",
    cta: "Next",
  },
  {
    title: "Set your favourite team",
    body: "Pick a team to see their next fixture, the model's prediction, and personalise this screen.",
    cta: "Done",
  },
] as const;

type Props = {
  onComplete: () => void;
};

export function OnboardingTour({ onComplete }: Props) {
  const [step, setStep] = useState(0);
  const cardRef = useRef<HTMLDivElement>(null);
  const current = STEPS[step];

  useEffect(() => {
    cardRef.current?.focus();
  }, [step]);

  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") onComplete();
    }
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [onComplete]);

  function handleTabKey(e: React.KeyboardEvent<HTMLDivElement>) {
    if (e.key !== "Tab") return;
    const focusable = cardRef.current?.querySelectorAll<HTMLElement>(
      "button, [href], input, [tabindex]:not([tabindex=\"-1\"])",
    );
    if (!focusable || focusable.length === 0) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  }

  function handleNext() {
    if (step >= STEPS.length - 1) {
      onComplete();
    } else {
      setStep(step + 1);
    }
  }

  return (
    <div
      className="onboarding-backdrop"
      role="dialog"
      aria-modal="true"
      aria-labelledby="onboarding-title"
      aria-describedby="onboarding-body"
    >
      <div
        ref={cardRef}
        className="onboarding-card"
        tabIndex={-1}
        onKeyDown={handleTabKey}
      >
        <div className="onboarding-progress" aria-hidden="true">
          {STEPS.map((_, i) => (
            <span
              key={i}
              className={`onboarding-dot${i === step ? " active" : i < step ? " done" : ""}`}
            />
          ))}
        </div>
        <p className="onboarding-step-label muted">
          Step {step + 1} of {STEPS.length}
        </p>
        <div aria-live="assertive" aria-atomic="true">
          <h2 id="onboarding-title" className="onboarding-title">
            {current.title}
          </h2>
          <p id="onboarding-body" className="onboarding-body">
            {current.body}
          </p>
        </div>
        <div className="onboarding-actions">
          <button type="button" className="onboarding-skip" onClick={onComplete}>
            Skip tour
          </button>
          <button type="button" className="onboarding-next" onClick={handleNext}>
            {current.cta}
          </button>
        </div>
      </div>
    </div>
  );
}
