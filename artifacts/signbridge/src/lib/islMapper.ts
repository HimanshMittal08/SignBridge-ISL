/**
 * islMapper.ts
 *
 * Deterministic English text → ISL gloss sequence mapper.
 *
 * This is a concept-level mapper for the demo vocabulary only.
 * It is NOT a full English-to-ISL translator — ISL has its own grammar,
 * spatial structure, and morphology that a token mapper cannot capture.
 *
 * Real assets (GIF/video/avatar) can be added later by populating
 * the `asset` field in DEMO_GLOSSES without changing the player architecture.
 */

/** A single ISL gloss entry that the player can render. */
export interface IslGloss {
  /** The normalised gloss label shown in the UI, e.g. "HELLO". */
  label: string;
  /**
   * Optional asset URL to display (GIF, WebM, poster image …).
   * When null the player renders a verified placeholder instead.
   */
  asset: string | null;
  /** Human-readable description of the sign's meaning. */
  meaning: string;
  /** Broad semantic category for display / filtering. */
  category: string;
}

/**
 * The complete set of ISL glosses supported in the demo vocabulary.
 * Extend this record to grow the vocabulary — the player will pick it up
 * automatically.
 *
 * asset is null until verified ISL reference video / GIF assets are available.
 */
export const DEMO_GLOSSES: Record<string, IslGloss> = {
  HELLO: {
    label: 'HELLO',
    asset: '/signs/hello.mp4',
    meaning: 'A greeting or an opening to a conversation.',
    category: 'Conversation',
  },
  HELP: {
    label: 'HELP',
    asset: '/signs/help.mp4',
    meaning: 'A request for assistance or support.',
    category: 'Needs',
  },
  WATER: {
    label: 'WATER',
    asset: '/signs/water.mp4',
    meaning: 'Water, or a request for something to drink.',
    category: 'Everyday',
  },
  PLEASE: {
    label: 'PLEASE',
    asset: '/signs/please.mp4',
    meaning: 'A polite request or softening of a message.',
    category: 'Conversation',
  },
  YES: {
    label: 'YES',
    asset: '/signs/yes.mp4',
    meaning: 'Agreement, confirmation, or an affirmative answer.',
    category: 'Conversation',
  },
  NO: {
    label: 'NO',
    asset: '/signs/no.mp4',
    meaning: 'Disagreement, refusal, or a negative answer.',
    category: 'Conversation',
  },
  GO: {
    label: 'GO',
    asset: '/signs/go.mp4',
    meaning: 'Movement toward another place or destination.',
    category: 'Everyday',
  },
  EAT: {
    label: 'EAT',
    asset: '/signs/eat.mp4',
    meaning: 'Food, a meal, or something to eat.',
    category: 'Everyday',
  },
};

/**
 * Word-level aliases that map English tokens → ISL gloss keys.
 *
 * Sorted longest-first within each group so multi-word phrases are matched
 * before their constituent words.
 */
const TOKEN_MAP: Array<{ tokens: string[]; gloss: keyof typeof DEMO_GLOSSES }> = [
  // Multi-word first
  { tokens: ['thank you', 'thanks'], gloss: 'HELLO' }, // not in current demo vocab; kept as future hook
  // Single-word aliases
  { tokens: ['hello', 'hi', 'hey', 'greetings'], gloss: 'HELLO' },
  { tokens: ['help', 'assist', 'assistance', 'support'], gloss: 'HELP' },
  { tokens: ['water', 'drink'], gloss: 'WATER' },
  { tokens: ['please', 'kindly'], gloss: 'PLEASE' },
  { tokens: ['yes', 'yeah', 'yep', 'ok', 'okay', 'sure', 'alright'], gloss: 'YES' },
  { tokens: ['no', 'nope', 'never', 'not'], gloss: 'NO' },
  { tokens: ['go', 'going', 'went', 'leave', 'move'], gloss: 'GO' },
  { tokens: ['eat', 'food', 'meal', 'hungry', 'hunger', 'snack'], gloss: 'EAT' },
];

export interface MapResult {
  /** Ordered ISL gloss sequence (may be empty). */
  glosses: IslGloss[];
  /** Whether at least one concept was mapped. */
  matched: boolean;
  /**
   * Glosses that appeared in the input text but fall outside the current
   * demo vocabulary (always empty for this deterministic mapper — kept for
   * future extension).
   */
  unsupported: string[];
}

/**
 * Map English free text to an ordered ISL gloss sequence.
 *
 * The mapper preserves the left-to-right order of the first match position
 * so the resulting gloss sequence roughly mirrors the English word order
 * (not ISL grammar — a full ISL translation engine would be needed for that).
 */
export function mapTextToISL(text: string): MapResult {
  const normalized = text.toLowerCase().trim();

  if (!normalized) {
    return { glosses: [], matched: false, unsupported: [] };
  }

  const hits: Array<{ position: number; gloss: keyof typeof DEMO_GLOSSES }> = [];

  for (const entry of TOKEN_MAP) {
    for (const token of entry.tokens) {
      // Use word-boundary-like matching: check if the token appears as a
      // standalone word (not inside another word).
      const regex = new RegExp(`(?<![a-z])${token.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?![a-z])`, 'i');
      const match = regex.exec(normalized);
      if (match) {
        // Avoid duplicate gloss entries.
        if (!hits.some(h => h.gloss === entry.gloss)) {
          hits.push({ position: match.index, gloss: entry.gloss });
        }
        break; // matched this entry, no need to try other aliases
      }
    }
  }

  // Sort by position so sequence follows input word order.
  hits.sort((a, b) => a.position - b.position);

  const glosses = hits.map(h => DEMO_GLOSSES[h.gloss]);
  return { glosses, matched: glosses.length > 0, unsupported: [] };
}
