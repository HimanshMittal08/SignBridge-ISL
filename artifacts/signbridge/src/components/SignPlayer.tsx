/**
 * SignPlayer.tsx
 *
 * Reusable ISL Sign Sequence Player.
 *
 * Architecture notes:
 * - Each sign is rendered through <SignFrame>, which is the single injection
 *   point for real GIF / video / avatar assets. When `gloss.asset` is null
 *   a consistent placeholder is rendered instead — swap it for a real asset
 *   later without touching the player logic.
 * - The player exposes Previous / Next / Play / Pause / Replay controls and
 *   shows "Sign N of M" progress.
 * - Auto-play advances one sign every `intervalMs` milliseconds (default 2 s).
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { ChevronLeft, ChevronRight, Pause, Play, RotateCcw } from 'lucide-react';
import type { IslGloss } from '@/lib/islMapper';

// ─── SignFrame ────────────────────────────────────────────────────────────────

interface SignFrameProps {
  gloss: IslGloss;
  /** Whether this sign is the actively-displayed one (for a11y / styling). */
  active: boolean;
}

/**
 * Renders a single ISL sign.
 *
 * When `gloss.asset` is a URL a <video> or <img> is shown.
 * Otherwise a styled placeholder is displayed so the layout is stable and the
 * component can be swapped in later with zero structural changes.
 */
function SignFrame({ gloss, active }: SignFrameProps) {
  const [hasError, setHasError] = useState(false);
  const hasAsset = Boolean(gloss.asset) && !hasError;

  useEffect(() => {
    setHasError(false);
  }, [gloss.asset]);

  return (
    <div
      className={`sign-frame ${active ? 'sign-frame--active' : ''}`}
      aria-label={`ISL sign for ${gloss.label}`}
      role="img"
    >
      {hasAsset ? (
        gloss.asset?.endsWith('.mp4') || gloss.asset?.endsWith('.webm') ? (
          <video
            key={gloss.asset}
            src={gloss.asset}
            autoPlay
            loop
            muted
            playsInline
            onError={() => setHasError(true)}
            className="sign-asset-video"
            aria-label={`Sign demonstration for ${gloss.label}`}
          />
        ) : (
          <img
            src={gloss.asset ?? ''}
            alt={`Sign demonstration for ${gloss.label}`}
            onError={() => setHasError(true)}
            className="sign-asset-img"
          />
        )
      ) : (
        /* ── Placeholder (shown when asset pending or load error) ── */
        <div className="sign-placeholder" aria-hidden="true">
          <div className="sign-placeholder__orbit" />
          <div className="sign-placeholder__gesture" />
          <span className="sign-placeholder__label">{gloss.label[0]}</span>
          <span className="sign-placeholder__hint">{hasError ? 'load error' : 'asset pending'}</span>
        </div>
      )}
    </div>
  );
}

// ─── SignPlayer ───────────────────────────────────────────────────────────────

interface SignPlayerProps {
  glosses: IslGloss[];
  /** Auto-advance interval in ms (default: 2000). */
  intervalMs?: number;
  /** Called when the player is closed / dismissed. */
  onClose?: () => void;
}

export function SignPlayer({ glosses, intervalMs = 2000, onClose }: SignPlayerProps) {
  const [current, setCurrent] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const total = glosses.length;

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const goTo = useCallback(
    (index: number) => {
      setCurrent(Math.max(0, Math.min(index, total - 1)));
    },
    [total],
  );

  const prev = useCallback(() => {
    setPlaying(false);
    clearTimer();
    goTo(current - 1);
  }, [clearTimer, current, goTo]);

  const next = useCallback(() => {
    setPlaying(false);
    clearTimer();
    goTo(current + 1);
  }, [clearTimer, current, goTo]);

  const replay = useCallback(() => {
    clearTimer();
    setCurrent(0);
    setPlaying(true);
  }, [clearTimer]);

  const togglePlay = useCallback(() => {
    setPlaying(prev => !prev);
  }, []);

  // Auto-advance logic
  useEffect(() => {
    clearTimer();
    if (!playing) return;

    timerRef.current = setInterval(() => {
      setCurrent(prev => {
        if (prev >= total - 1) {
          // Reached the end: stop auto-play
          setPlaying(false);
          clearInterval(timerRef.current!);
          timerRef.current = null;
          return prev;
        }
        return prev + 1;
      });
    }, intervalMs);

    return clearTimer;
  }, [clearTimer, intervalMs, playing, total]);

  // Reset when a new gloss list comes in
  useEffect(() => {
    setCurrent(0);
    setPlaying(false);
    clearTimer();
  }, [glosses, clearTimer]);

  if (total === 0) return null;

  const gloss = glosses[current];

  return (
    <div className="sign-player" role="region" aria-label="ISL sign sequence player">
      {/* Header */}
      <div className="sign-player__header">
        <div className="sign-player__title">
          <span className="eyebrow">ISL concept sequence</span>
        </div>
        {onClose && (
          <button
            className="icon-button"
            onClick={onClose}
            aria-label="Close sign player"
            style={{ width: 28, height: 28 }}
          >
            ✕
          </button>
        )}
      </div>

      {/* Gloss strip (breadcrumb-style) */}
      <div className="sign-player__strip" role="list" aria-label="Sign sequence">
        {glosses.map((g, i) => (
          <div key={`${g.label}-${i}`} style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            {i > 0 && <span className="sign-chip-arrow" aria-hidden="true">→</span>}
            <button
              className={`sign-chip ${i === current ? 'sign-chip--active' : ''}`}
              onClick={() => { setPlaying(false); clearTimer(); goTo(i); }}
              aria-current={i === current ? 'true' : undefined}
              aria-label={`Go to sign ${i + 1}: ${g.label}`}
              role="listitem"
            >
              {g.label}
            </button>
          </div>
        ))}
      </div>

      {/* Main sign display */}
      <div className="sign-player__stage">
        <SignFrame gloss={gloss} active={true} />

        <div className="sign-player__info">
          <h3 className="sign-player__gloss-name">{gloss.label}</h3>
          <p className="sign-player__meaning">{gloss.meaning}</p>
          <span className="sign-player__category">{gloss.category}</span>
        </div>
      </div>

      {/* Progress */}
      <div className="sign-player__progress" aria-live="polite" aria-atomic="true">
        Sign {current + 1} of {total}
      </div>

      {/* Controls */}
      <div className="sign-player__controls" role="toolbar" aria-label="Playback controls">
        <button
          className="button soft small"
          onClick={prev}
          disabled={current === 0}
          aria-label="Previous sign"
          data-testid="button-sign-prev"
        >
          <ChevronLeft size={14} /> Prev
        </button>

        <button
          className="button primary small"
          onClick={togglePlay}
          aria-label={playing ? 'Pause sign sequence' : 'Play sign sequence'}
          data-testid="button-sign-play-pause"
        >
          {playing ? <Pause size={14} /> : <Play size={14} />}
          {playing ? 'Pause' : 'Play'}
        </button>

        <button
          className="button soft small"
          onClick={replay}
          aria-label="Replay from beginning"
          data-testid="button-sign-replay"
        >
          <RotateCcw size={14} /> Replay
        </button>

        <button
          className="button soft small"
          onClick={next}
          disabled={current === total - 1}
          aria-label="Next sign"
          data-testid="button-sign-next"
        >
          Next <ChevronRight size={14} />
        </button>
      </div>

      {/* Asset disclaimer */}
      <div className="sign-player__disclaimer">
        <span className="eyebrow">Demo vocabulary · {total} concept{total !== 1 ? 's' : ''}</span>
        <span style={{ color: 'hsl(var(--muted-foreground))', fontSize: 9 }}>
          Placeholder frames shown — real ISL assets to be connected
        </span>
      </div>
    </div>
  );
}

// ─── TextToISLPanel ───────────────────────────────────────────────────────────

import { mapTextToISL } from '@/lib/islMapper';
import { Send } from 'lucide-react';

interface TextToISLPanelProps {
  /** Optional initial text to pre-populate the input (e.g. from conversation). */
  initialText?: string;
}

/**
 * Self-contained panel: text input + mapper + SignPlayer.
 * Drop this inside any layout without additional props.
 */
export function TextToISLPanel({ initialText = '' }: TextToISLPanelProps) {
  const [input, setInput] = useState(initialText);
  const [result, setResult] = useState<ReturnType<typeof mapTextToISL> | null>(null);
  const [submitted, setSubmitted] = useState('');

  const handleSubmit = () => {
    const text = input.trim();
    if (!text) return;
    setResult(mapTextToISL(text));
    setSubmitted(text);
  };

  return (
    <div className="text-to-isl-panel">
      <div className="text-to-isl-input-row">
        <input
          type="text"
          className="search-input"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') handleSubmit(); }}
          placeholder="Type English text to see ISL signs…"
          aria-label="Enter English text to convert to ISL sign sequence"
          data-testid="input-text-to-isl"
          style={{ borderRadius: 9 }}
        />
        <button
          className="button primary small"
          onClick={handleSubmit}
          disabled={!input.trim()}
          aria-label="Convert to ISL sign sequence"
          data-testid="button-text-to-isl-submit"
        >
          <Send size={13} /> Show signs
        </button>
      </div>

      {result && (
        <div style={{ marginTop: 16 }}>
          {result.matched ? (
            <SignPlayer
              glosses={result.glosses}
              onClose={() => { setResult(null); setSubmitted(''); }}
            />
          ) : (
            <div className="text-to-isl-no-match">
              <span className="eyebrow">Outside demo vocabulary</span>
              <p>
                No supported ISL concepts were found in <em>"{submitted}"</em>.
                The current demo vocabulary covers: HELLO, HELP, WATER, PLEASE, YES, NO, GO, EAT.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
