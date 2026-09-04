/**
 * islMapper.ts
 *
 * Deterministic English text → ISL gloss sequence mapper for all 60 ISL classes.
 */

export interface IslGloss {
  label: string;
  asset: string | null;
  meaning: string;
  category: string;
}

export const DEMO_GLOSSES: Record<string, IslGloss> = {
  BOY: { label: 'BOY', asset: '/signs/boy.mp4', meaning: 'A male child or young man.', category: 'People' },
  BROTHER: { label: 'BROTHER', asset: '/signs/brother.mp4', meaning: 'A male sibling.', category: 'Family' },
  DOCTOR: { label: 'DOCTOR', asset: '/signs/doctor.mp4', meaning: 'A medical professional or physician.', category: 'Emergency' },
  DRINK: { label: 'DRINK', asset: '/signs/drink.mp4', meaning: 'To swallow liquid or beverage.', category: 'Everyday' },
  EAT: { label: 'EAT', asset: '/signs/eat.mp4', meaning: 'Food, a meal, or something to eat.', category: 'Everyday' },
  FAMILY: { label: 'FAMILY', asset: '/signs/family.mp4', meaning: 'A group of related individuals.', category: 'Family' },
  FATHER: { label: 'FATHER', asset: '/signs/father.mp4', meaning: 'A male parent.', category: 'Family' },
  FOOD: { label: 'FOOD', asset: '/signs/food.mp4', meaning: 'Nourishment or edible items.', category: 'Everyday' },
  FRIEND: { label: 'FRIEND', asset: '/signs/friend.mp4', meaning: 'A person with whom one has a bond of mutual affection.', category: 'People' },
  GIRL: { label: 'GIRL', asset: '/signs/girl.mp4', meaning: 'A female child or young woman.', category: 'People' },
  GO: { label: 'GO', asset: '/signs/go.mp4', meaning: 'Movement toward another place or destination.', category: 'Everyday' },
  GOOD_AFTERNOON: { label: 'GOOD_AFTERNOON', asset: '/signs/good_afternoon.mp4', meaning: 'An afternoon greeting.', category: 'Conversation' },
  GOOD_EVENING: { label: 'GOOD_EVENING', asset: '/signs/good_evening.mp4', meaning: 'An evening greeting.', category: 'Conversation' },
  GOOD_MORNING: { label: 'GOOD_MORNING', asset: '/signs/good_morning.mp4', meaning: 'A morning greeting.', category: 'Conversation' },
  GOOD_NIGHT: { label: 'GOOD_NIGHT', asset: '/signs/good_night.mp4', meaning: 'A nighttime farewell.', category: 'Conversation' },
  HE: { label: 'HE', asset: '/signs/he.mp4', meaning: 'Referring to a male person.', category: 'People' },
  HELLO: { label: 'HELLO', asset: '/signs/hello.mp4', meaning: 'A greeting or an opening to a conversation.', category: 'Conversation' },
  HELP: { label: 'HELP', asset: '/signs/help.mp4', meaning: 'A request for assistance or support.', category: 'Emergency' },
  HOSPITAL: { label: 'HOSPITAL', asset: '/signs/hospital.mp4', meaning: 'A medical care facility.', category: 'Emergency' },
  HOUSE: { label: 'HOUSE', asset: '/signs/house.mp4', meaning: 'A residential building or home.', category: 'Places' },
  HOW_ARE_YOU: { label: 'HOW_ARE_YOU', asset: '/signs/how_are_you.mp4', meaning: 'Inquiring about someone\'s well-being.', category: 'Conversation' },
  I: { label: 'I', asset: '/signs/i.mp4', meaning: 'First-person singular pronoun.', category: 'People' },
  MOTHER: { label: 'MOTHER', asset: '/signs/mother.mp4', meaning: 'A female parent.', category: 'Family' },
  NO: { label: 'NO', asset: '/signs/no.mp4', meaning: 'Disagreement, refusal, or a negative answer.', category: 'Conversation' },
  OKAY: { label: 'OKAY', asset: '/signs/okay.mp4', meaning: 'Expressing approval or agreement.', category: 'Conversation' },
  PLEASE: { label: 'PLEASE', asset: '/signs/please.mp4', meaning: 'A polite request or softening of a message.', category: 'Conversation' },
  SCHOOL: { label: 'SCHOOL', asset: '/signs/school.mp4', meaning: 'An educational institution.', category: 'Places' },
  SHE: { label: 'SHE', asset: '/signs/she.mp4', meaning: 'Referring to a female person.', category: 'People' },
  SISTER: { label: 'SISTER', asset: '/signs/sister.mp4', meaning: 'A female sibling.', category: 'Family' },
  STUDENT: { label: 'STUDENT', asset: '/signs/student.mp4', meaning: 'A learner or person attending school.', category: 'People' },
  TEA: { label: 'TEA', asset: '/signs/tea.mp4', meaning: 'A hot aromatic beverage.', category: 'Everyday' },
  TEACHER: { label: 'TEACHER', asset: '/signs/teacher.mp4', meaning: 'An educator or instructor.', category: 'People' },
  THANK_YOU: { label: 'THANK_YOU', asset: '/signs/thank_you.mp4', meaning: 'An expression of gratitude.', category: 'Conversation' },
  TODAY: { label: 'TODAY', asset: '/signs/today.mp4', meaning: 'On or during the present day.', category: 'Everyday' },
  WATER: { label: 'WATER', asset: '/signs/water.mp4', meaning: 'Water, or a request for something to drink.', category: 'Everyday' },
  WE: { label: 'WE', asset: '/signs/we.mp4', meaning: 'First-person plural pronoun.', category: 'People' },
  WHAT: { label: 'WHAT', asset: '/signs/what.mp4', meaning: 'Question token inquiring about a thing.', category: 'Conversation' },
  WHERE: { label: 'WHERE', asset: '/signs/where.mp4', meaning: 'Question token inquiring about a location.', category: 'Conversation' },
  YES: { label: 'YES', asset: '/signs/yes.mp4', meaning: 'Agreement, confirmation, or an affirmative answer.', category: 'Conversation' },
  YOU: { label: 'YOU', asset: '/signs/you.mp4', meaning: 'Second-person pronoun.', category: 'People' }
};

const TOKEN_MAP: Array<{ tokens: string[]; gloss: keyof typeof DEMO_GLOSSES }> = [
  // Multi-word phrase aliases (ordered longest-first)
  { tokens: ['good morning'], gloss: 'GOOD_MORNING' },
  { tokens: ['good afternoon'], gloss: 'GOOD_AFTERNOON' },
  { tokens: ['good evening'], gloss: 'GOOD_EVENING' },
  { tokens: ['good night'], gloss: 'GOOD_NIGHT' },
  { tokens: ['how are you', 'how do you do', 'how is it going'], gloss: 'HOW_ARE_YOU' },
  { tokens: ['thank you', 'thanks', 'thankyou'], gloss: 'THANK_YOU' },

  // Single word aliases
  { tokens: ['boy', 'lad'], gloss: 'BOY' },
  { tokens: ['brother', 'bro'], gloss: 'BROTHER' },
  { tokens: ['doctor', 'physician', 'doc'], gloss: 'DOCTOR' },
  { tokens: ['drink', 'drinking'], gloss: 'DRINK' },
  { tokens: ['eat', 'eating', 'hungry', 'meal', 'snack'], gloss: 'EAT' },
  { tokens: ['family', 'relatives'], gloss: 'FAMILY' },
  { tokens: ['father', 'dad', 'papa'], gloss: 'FATHER' },
  { tokens: ['food', 'lunch', 'dinner', 'breakfast'], gloss: 'FOOD' },
  { tokens: ['friend', 'pal', 'buddy'], gloss: 'FRIEND' },
  { tokens: ['girl'], gloss: 'GIRL' },
  { tokens: ['go', 'going', 'leave', 'walk'], gloss: 'GO' },
  { tokens: ['he', 'him', 'his'], gloss: 'HE' },
  { tokens: ['hello', 'hi', 'hey', 'greetings'], gloss: 'HELLO' },
  { tokens: ['help', 'assist', 'assistance'], gloss: 'HELP' },
  { tokens: ['hospital', 'clinic'], gloss: 'HOSPITAL' },
  { tokens: ['house', 'home'], gloss: 'HOUSE' },
  { tokens: ['i', 'me', 'my', 'myself'], gloss: 'I' },
  { tokens: ['mother', 'mom', 'mum', 'mama'], gloss: 'MOTHER' },
  { tokens: ['no', 'nope', 'never', 'not'], gloss: 'NO' },
  { tokens: ['ok', 'okay', 'alright', 'fine'], gloss: 'OKAY' },
  { tokens: ['please', 'kindly'], gloss: 'PLEASE' },
  { tokens: ['school', 'class'], gloss: 'SCHOOL' },
  { tokens: ['she', 'her', 'hers'], gloss: 'SHE' },
  { tokens: ['sister', 'sis'], gloss: 'SISTER' },
  { tokens: ['student', 'pupil'], gloss: 'STUDENT' },
  { tokens: ['tea', 'chai'], gloss: 'TEA' },
  { tokens: ['teacher', 'tutor', 'professor'], gloss: 'TEACHER' },
  { tokens: ['today'], gloss: 'TODAY' },
  { tokens: ['water'], gloss: 'WATER' },
  { tokens: ['we', 'us', 'our'], gloss: 'WE' },
  { tokens: ['what'], gloss: 'WHAT' },
  { tokens: ['where'], gloss: 'WHERE' },
  { tokens: ['yes', 'yeah', 'yep', 'sure'], gloss: 'YES' },
  { tokens: ['you', 'your', 'yours'], gloss: 'YOU' }
];

export interface MapResult {
  glosses: IslGloss[];
  matched: boolean;
  unsupported: string[];
}

export function mapTextToISL(text: string): MapResult {
  const normalized = text.toLowerCase().trim();

  if (!normalized) {
    return { glosses: [], matched: false, unsupported: [] };
  }

  const hits: Array<{ position: number; gloss: keyof typeof DEMO_GLOSSES }> = [];

  for (const entry of TOKEN_MAP) {
    for (const token of entry.tokens) {
      const regex = new RegExp(`(?<![a-z])${token.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?![a-z])`, 'i');
      const match = regex.exec(normalized);
      if (match) {
        if (!hits.some(h => h.gloss === entry.gloss)) {
          hits.push({ position: match.index, gloss: entry.gloss });
        }
        break;
      }
    }
  }

  hits.sort((a, b) => a.position - b.position);

  const glosses = hits.map(h => DEMO_GLOSSES[h.gloss]);
  return { glosses, matched: glosses.length > 0, unsupported: [] };
}
