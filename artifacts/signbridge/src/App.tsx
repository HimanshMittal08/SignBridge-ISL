import { type ReactNode, useEffect, useMemo, useRef, useState } from 'react';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ArrowRight, AudioLines, Bell, BookOpen, Camera, ChevronRight, CircleHelp, Ear, Hand, Languages, Mic, Pause, Play, Search, Send, Settings2, Sparkles, Terminal, Volume2, X } from 'lucide-react';
import { Link, Route, Switch, Router as WouterRouter, useLocation } from 'wouter';
import { ErrorBoundary } from '@/components/error-boundary';
import { Toaster } from '@/components/ui/toaster';
import { TooltipProvider } from '@/components/ui/tooltip';
import NotFound from '@/pages/not-found';
import { SignPlayer } from '@/components/SignPlayer';
import { mapTextToISL } from '@/lib/islMapper';

type Direction = 'signer' | 'hearing';
type Message = {
  id: number;
  direction: Direction;
  exactTranscript: string;
  rawSigns?: string[];
  concepts: string[];
  interpretation: string;
  alternatives?: string[];
  timestamp: string;
  status: string;
  emergency?: boolean;
  textToIsl?: boolean;
};
type SignConcept = {
  id: string;
  label: string;
  meaning: string;
  category: string;
  priority: 'high' | 'normal';
  confidence: number;
  demoAvailable: boolean;
};
type CameraStatus = 'disabled' | 'requesting' | 'streaming' | 'unsupported' | 'denied' | 'unavailable' | 'no-device' | 'error';
type HandLandmark = { x: number; y: number; z: number };
type DetectedHand = { handedness: string; landmarks: HandLandmark[] };
type ModelStatus = 'idle' | 'loading' | 'ready' | 'error';
const HAND_CONNECTIONS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];
const MEDIAPIPE_WASM_ROOT = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm';
const HAND_LANDMARKER_MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task';

const queryClient = new QueryClient();

const concepts: SignConcept[] = [
  { id: 'bank', label: 'BANK', meaning: 'A financial institution or banking service.', category: 'Places', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'boy', label: 'BOY', meaning: 'A male child or young man.', category: 'People', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'brother', label: 'BROTHER', meaning: 'A male sibling.', category: 'Family', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'bus', label: 'BUS', meaning: 'A large public road vehicle.', category: 'Transport', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'car', label: 'CAR', meaning: 'An automobile or personal road vehicle.', category: 'Transport', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'city', label: 'CITY', meaning: 'A large human settlement or town.', category: 'Places', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'cold', label: 'COLD', meaning: 'Low temperature or feeling chilly.', category: 'Feelings', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'doctor', label: 'DOCTOR', meaning: 'A medical professional or physician.', category: 'Emergency', priority: 'high', confidence: .95, demoAvailable: true },
  { id: 'drink', label: 'DRINK', meaning: 'To swallow liquid or beverage.', category: 'Everyday', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'eat', label: 'EAT', meaning: 'Food, a meal, or something to eat.', category: 'Everyday', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'family', label: 'FAMILY', meaning: 'A group of related individuals.', category: 'Family', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'father', label: 'FATHER', meaning: 'A male parent.', category: 'Family', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'food', label: 'FOOD', meaning: 'Nourishment or edible items.', category: 'Everyday', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'friend', label: 'FRIEND', meaning: 'A person with whom one has a bond of mutual affection.', category: 'People', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'girl', label: 'GIRL', meaning: 'A female child or young woman.', category: 'People', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'go', label: 'GO', meaning: 'Movement toward another place or destination.', category: 'Everyday', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'good_afternoon', label: 'GOOD_AFTERNOON', meaning: 'An afternoon greeting.', category: 'Conversation', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'good_evening', label: 'GOOD_EVENING', meaning: 'An evening greeting.', category: 'Conversation', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'good_morning', label: 'GOOD_MORNING', meaning: 'A morning greeting.', category: 'Conversation', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'good_night', label: 'GOOD_NIGHT', meaning: 'A nighttime farewell.', category: 'Conversation', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'happy', label: 'HAPPY', meaning: 'Feeling or showing pleasure or contentment.', category: 'Feelings', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'he', label: 'HE', meaning: 'Referring to a male person.', category: 'People', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'hello', label: 'HELLO', meaning: 'A greeting or an opening to a conversation.', category: 'Conversation', priority: 'normal', confidence: .98, demoAvailable: true },
  { id: 'help', label: 'HELP', meaning: 'A request for assistance or support.', category: 'Emergency', priority: 'high', confidence: .95, demoAvailable: true },
  { id: 'hospital', label: 'HOSPITAL', meaning: 'A medical care facility.', category: 'Emergency', priority: 'high', confidence: .95, demoAvailable: true },
  { id: 'house', label: 'HOUSE', meaning: 'A residential building or home.', category: 'Places', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'how_are_you', label: 'HOW_ARE_YOU', meaning: 'Inquiring about someone\'s well-being.', category: 'Conversation', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'i', label: 'I', meaning: 'First-person singular pronoun.', category: 'People', priority: 'normal', confidence: .97, demoAvailable: true },
  { id: 'india', label: 'INDIA', meaning: 'The country India.', category: 'Places', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'library', label: 'LIBRARY', meaning: 'A building containing books and resources.', category: 'Places', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'location', label: 'LOCATION', meaning: 'A particular place or position.', category: 'Everyday', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'market', label: 'MARKET', meaning: 'A place for buying and selling goods.', category: 'Places', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'mother', label: 'MOTHER', meaning: 'A female parent.', category: 'Family', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'no', label: 'NO', meaning: 'Disagreement, refusal, or a negative answer.', category: 'Conversation', priority: 'normal', confidence: .99, demoAvailable: true },
  { id: 'office', label: 'OFFICE', meaning: 'A room or building used for business work.', category: 'Places', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'okay', label: 'OKAY', meaning: 'Expressing approval or agreement.', category: 'Conversation', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'park', label: 'PARK', meaning: 'A public green area for recreation.', category: 'Places', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'please', label: 'PLEASE', meaning: 'A polite request or softening of a message.', category: 'Conversation', priority: 'normal', confidence: .92, demoAvailable: true },
  { id: 'police', label: 'POLICE', meaning: 'Law enforcement officer or force.', category: 'Emergency', priority: 'high', confidence: .95, demoAvailable: true },
  { id: 'restaurant', label: 'RESTAURANT', meaning: 'A business where meals are served.', category: 'Places', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'school', label: 'SCHOOL', meaning: 'An educational institution.', category: 'Places', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'she', label: 'SHE', meaning: 'Referring to a female person.', category: 'People', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'sick', label: 'SICK', meaning: 'Feeling ill or unwell.', category: 'Emergency', priority: 'high', confidence: .95, demoAvailable: true },
  { id: 'sister', label: 'SISTER', meaning: 'A female sibling.', category: 'Family', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'sit', label: 'SIT', meaning: 'To rest one\'s body on a seat.', category: 'Everyday', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'store_or_shop', label: 'STORE_OR_SHOP', meaning: 'A retail establishment or shop.', category: 'Places', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'student', label: 'STUDENT', meaning: 'A learner or person attending school.', category: 'People', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'tea', label: 'TEA', meaning: 'A hot aromatic beverage.', category: 'Everyday', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'teacher', label: 'TEACHER', meaning: 'An educator or instructor.', category: 'People', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'thank_you', label: 'THANK_YOU', meaning: 'An expression of gratitude.', category: 'Conversation', priority: 'normal', confidence: .97, demoAvailable: true },
  { id: 'time', label: 'TIME', meaning: 'Clock time or temporal measurement.', category: 'Everyday', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'today', label: 'TODAY', meaning: 'On or during the present day.', category: 'Everyday', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'train', label: 'TRAIN', meaning: 'A railway vehicle or transport.', category: 'Transport', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'train_station', label: 'TRAIN_STATION', meaning: 'A railway station terminal.', category: 'Transport', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'water', label: 'WATER', meaning: 'Water, or a request for something to drink.', category: 'Everyday', priority: 'normal', confidence: .94, demoAvailable: true },
  { id: 'we', label: 'WE', meaning: 'First-person plural pronoun.', category: 'People', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'what', label: 'WHAT', meaning: 'Question token inquiring about a thing.', category: 'Conversation', priority: 'normal', confidence: .95, demoAvailable: true },
  { id: 'where', label: 'WHERE', meaning: 'Question token inquiring about a location.', category: 'Conversation', priority: 'normal', confidence: .94, demoAvailable: true },
  { id: 'yes', label: 'YES', meaning: 'Agreement, confirmation, or an affirmative answer.', category: 'Conversation', priority: 'normal', confidence: .99, demoAvailable: true },
  { id: 'you', label: 'YOU', meaning: 'Second-person pronoun.', category: 'People', priority: 'normal', confidence: .97, demoAvailable: true }
];

const conceptAliases: Record<string, string[]> = {
  bank: ['bank'], boy: ['boy'], brother: ['brother'], bus: ['bus'], car: ['car'], city: ['city'], cold: ['cold'],
  doctor: ['doctor'], drink: ['drink'], eat: ['eat'], family: ['family'], father: ['father'], food: ['food'],
  friend: ['friend'], girl: ['girl'], go: ['go'], good_afternoon: ['good afternoon'], good_evening: ['good evening'],
  good_morning: ['good morning'], good_night: ['good night'], happy: ['happy'], he: ['he'], hello: ['hello', 'hi', 'hey'],
  help: ['help', 'assist', 'assistance'], hospital: ['hospital'], house: ['house', 'home'], how_are_you: ['how are you'],
  i: ['i', 'me', 'my'], india: ['india'], library: ['library'], location: ['location', 'place'], market: ['market'],
  mother: ['mother', 'mom'], no: ['no', 'not'], office: ['office', 'work'], okay: ['okay', 'ok'], park: ['park'],
  please: ['please'], police: ['police', 'cop'], restaurant: ['restaurant'], school: ['school'], she: ['she', 'her'],
  sick: ['sick', 'ill'], sister: ['sister'], sit: ['sit'], store_or_shop: ['store', 'shop'], student: ['student'],
  tea: ['tea', 'chai'], teacher: ['teacher'], thank_you: ['thank you', 'thanks'], time: ['time'], today: ['today'],
  train: ['train'], train_station: ['train station'], water: ['water', 'drink'], we: ['we', 'us'], what: ['what'],
  where: ['where'], yes: ['yes', 'yeah'], you: ['you', 'your']
};

function extractConcepts(text: string) {
  const normalized = text.toLowerCase();
  return concepts
    .map(concept => {
      const matches = (conceptAliases[concept.id] ?? [concept.label.toLowerCase()])
        .map(alias => ({ alias, index: normalized.indexOf(alias) }))
        .filter(match => match.index >= 0)
        .sort((a, b) => a.index - b.index);
      return { label: concept.label, index: matches[0]?.index ?? -1 };
    })
    .filter(item => item.index >= 0)
    .sort((a, b) => a.index - b.index)
    .map(item => item.label);
}

const initialMessages: Message[] = [
  {
    id: 1, direction: 'signer', exactTranscript: 'HELLO. I NEED WATER, PLEASE.', rawSigns: ['HELLO', 'WATER', 'PLEASE'], concepts: ['HELLO', 'WATER'], interpretation: 'Hello — could I have some water, please?', alternatives: ['I would like water.', 'Please bring water.'], timestamp: '10:41:08', status: 'Interpreted',
  },
  {
    id: 2, direction: 'hearing', exactTranscript: 'Of course. I will bring you water now.', concepts: ['YES', 'WATER'], interpretation: 'Of course. I will bring you water now.', timestamp: '10:41:22', status: 'Spoken',
  },
];

function extractFrameFeatures(detectedHands: DetectedHand[]): number[] {
  const features = new Array(126).fill(0);
  if (!detectedHands || detectedHands.length === 0) return features;

  for (const hand of detectedHands) {
    const slot = hand.handedness.toLowerCase() === 'left' ? 0 : 1;
    const landmarks = hand.landmarks;
    if (!landmarks || landmarks.length < 21) continue;

    const wrist = landmarks[0];
    const relX9 = landmarks[9].x - wrist.x;
    const relY9 = landmarks[9].y - wrist.y;
    const dist = Math.sqrt(relX9 * relX9 + relY9 * relY9);
    const scale = Math.max(dist, 1e-6);

    const offset = slot * 63;
    for (let i = 0; i < 21; i++) {
      const pt = landmarks[i];
      features[offset + i * 3 + 0] = (pt.x - wrist.x) / scale;
      features[offset + i * 3 + 1] = (pt.y - wrist.y) / scale;
      features[offset + i * 3 + 2] = (pt.z - wrist.z) / scale;
    }
  }

  return features;
}


function Logo() {
  return <span className="brand-mark" aria-hidden="true"><Languages size={19} strokeWidth={2.5} /></span>;
}

function Shell({ children }: { children: ReactNode }) {
  const [location] = useLocation();
  const nav = [
    { href: '/', label: 'Overview', icon: Sparkles },
    { href: '/conversation', label: 'Conversation', icon: AudioLines },
    { href: '/vocabulary', label: 'ISL vocabulary', icon: BookOpen },
  ];
  return (
    <div className="app-shell">
      <header className="mobile-header">
        <Link href="/" className="brand" data-testid="link-mobile-logo"><Logo /><span className="brand-word">Sign<span>Bridge</span></span></Link>
        <nav className="mobile-nav" aria-label="Mobile navigation">
          {nav.map(({ href, label, icon: Icon }) => <Link key={href} href={href} aria-label={label} className={location === href ? 'active' : ''} data-testid={`link-mobile-${label.toLowerCase().replaceAll(' ', '-')}`}><Icon size={17} /></Link>)}
        </nav>
      </header>
      <aside className="sidebar">
        <Link href="/" className="brand" data-testid="link-sidebar-logo"><Logo /><span className="brand-word">Sign<span>Bridge</span></span></Link>
        <div className="sidebar-rule" />
        <div className="nav-label">Workspace</div>
        <nav className="nav-list" aria-label="Primary navigation">
          {nav.map(({ href, label, icon: Icon }) => <Link key={href} href={href} className={`nav-link ${location === href ? 'active' : ''}`} data-testid={`link-nav-${label.toLowerCase().replaceAll(' ', '-')}`}><Icon size={17} /><span>{label}</span>{location === href && <ChevronRight size={14} style={{ marginLeft: 'auto' }} />}</Link>)}
        </nav>
        <div className="sidebar-foot">
          <strong>A clearer kind of conversation.</strong>
          Raw signs, exact words, and meaning stay visible — so both people can choose what feels right.
        </div>
      </aside>
      <main className="main">{children}</main>
    </div>
  );
}

function Topbar({ detail }: { detail: string }) {
  return <div className="topbar"><div className="topbar-kicker">{detail}</div><div className="topbar-actions"><div className="presence"><span className="presence-dot" /> Local workspace</div><button className="icon-button" aria-label="Help and guidance" title="Help and guidance" onClick={() => window.alert('SignBridge keeps raw signs, exact words, and interpretation visible together.')} data-testid="button-help"><CircleHelp size={18} /></button><button className="icon-button" aria-label="Notifications" title="Notifications" onClick={() => window.alert('No new notifications in this local workspace.')} data-testid="button-notifications"><Bell size={18} /></button></div></div>;
}

function HomePage() {
  const [, setLocation] = useLocation();
  return <Shell><div className="page">
    <Topbar detail="A communication workspace for two" />
    <section className="hero-grid">
      <div>
        <div className="eyebrow">Indian Sign Language / spoken language</div>
        <h1 className="hero-title">Meaning,<br /><em>made visible.</em></h1>
        <p className="hero-copy">SignBridge keeps the whole conversation in view — the signs detected, the words heard, and the interpretation between them. Built for Deaf and hearing people to meet in the middle.</p>
        <div className="hero-actions">
          <button className="button primary" onClick={() => setLocation('/conversation')} data-testid="button-start-conversation">Start a conversation <ArrowRight size={15} /></button>
          <button className="button ghost" onClick={() => setLocation('/vocabulary')} data-testid="button-explore-vocabulary">Explore vocabulary <BookOpen size={14} /></button>
        </div>
      </div>
      <div className="hero-aside">
        <div className="signal-card">
          <div className="signal-header"><span className="signal-live"><span className="presence-dot" /> Demo mode</span><span style={{ color: 'hsl(195 12% 72%)', fontSize: 10 }}>01 / 03</span></div>
          <div className="signal-visual"><div className="gesture-line" /><div className="hand-orbit" /></div>
          <div className="signal-quote">“I can see what you meant.”</div>
          <div className="signal-foot"><span>Raw signs + interpretation</span><span>ISL ↔ English</span></div>
        </div>
      </div>
    </section>
    <section className="home-strip" aria-label="SignBridge principles">
      <div className="strip-item"><div className="strip-number">01</div><div className="strip-copy">A shared view of every signal, not a black box.</div></div>
      <div className="strip-item"><div className="strip-number">02</div><div className="strip-copy">Confidence shown as context, never as a promise.</div></div>
      <div className="strip-item"><div className="strip-number">03</div><div className="strip-copy">One calm workspace for both sides of the exchange.</div></div>
    </section>
    <div className="section-heading"><div><div className="eyebrow">The SignBridge approach</div><h2 className="section-title">Nothing important<br />gets translated away.</h2></div><p className="section-note">A conversation can hold ambiguity. We give people the pieces they need to resolve it together.</p></div>
    <section className="principles">
      <article className="principle"><div className="principle-icon"><Hand size={23} /></div><h3>See the signal</h3><p>Detected signs and ISL concepts remain available alongside the natural-language interpretation.</p></article>
      <article className="principle"><div className="principle-icon"><Ear size={23} /></div><h3>Hear the exact words</h3><p>Spoken transcripts are preserved word for word, with replay when supported.</p></article>
      <article className="principle"><div className="principle-icon"><Settings2 size={23} /></div><h3>Stay in control</h3><p>Try a different interpretation, type a reply, or pause the devices at any time.</p></article>
    </section>
    <section className="quote-band"><div><blockquote>“Clarity is something we build together.”</blockquote><cite>SignBridge field note / 01</cite></div><button className="button primary" onClick={() => setLocation('/conversation')} data-testid="button-try-demo">Try the demo <ArrowRight size={15} /></button></section>
  </div></Shell>;
}

function MessageBubble({ message, alternative, onAlternative, onSpeak }: { message: Message; alternative?: string; onAlternative: (value: string) => void; onSpeak: (text: string) => void }) {
  const isSigner = message.direction === 'signer';
  const islMapping = useMemo(() => {
    if (!isSigner && message.textToIsl && message.exactTranscript) {
      return mapTextToISL(message.exactTranscript);
    }
    return null;
  }, [isSigner, message.textToIsl, message.exactTranscript]);

  return <article className={`message ${isSigner ? 'signer' : 'hearing'}`} data-testid={`message-${message.id}`}>
    <div className="message-avatar" aria-hidden="true">{isSigner ? <Hand size={16} /> : <Ear size={16} />}</div>
    <div className="message-body">
      <div className="message-meta">{isSigner ? 'Signer · ISL' : 'Hearing participant · English'} · {message.timestamp}</div>
       <div className={`bubble ${message.emergency ? 'emergency-bubble' : ''}`}>
         {message.emergency && <div className="emergency-note" role="alert"><Bell size={13} /> Emergency context detected</div>}
        {isSigner ? <><span className="bubble-label">Interpretation</span><div className="bubble-text" data-testid={`text-interpretation-${message.id}`}>{alternative || message.interpretation}</div><div className="raw-line"><b>Exact signs:</b> {message.exactTranscript}</div><div className="concept-line">{message.concepts.map(concept => <span className="concept-tag" key={concept}>{concept}</span>)}</div>
          {message.alternatives && <div className="alternative-list"><span>Try:</span>{message.alternatives.map(item => <button key={item} className={`alternative-button ${alternative === item ? 'chosen' : ''}`} onClick={() => onAlternative(item)} data-testid={`button-alternative-${message.id}-${item.toLowerCase().replaceAll(' ', '-')}`}>{item}</button>)}</div>}
        </> : <><span className="bubble-label">Exact spoken transcript</span><div className="bubble-text" data-testid={`text-transcript-${message.id}`}>{message.exactTranscript}</div>
          {islMapping && (
            <div className="message-isl-player-wrap" style={{ marginTop: 14 }}>
              {islMapping.matched ? (
                <SignPlayer glosses={islMapping.glosses} />
              ) : (
                <div className="text-to-isl-no-match" style={{ marginTop: 10 }}>
                  <span className="eyebrow">Outside demo vocabulary</span>
                  <p>
                    No supported ISL concepts were found in <em>"{message.exactTranscript}"</em>.<br />
                    The current demo vocabulary covers all 60 ISL classes.
                  </p>
                </div>
              )}
            </div>
          )}
          {!islMapping && message.concepts.length > 0 && <div className="concept-line">{message.concepts.map(concept => <span className="concept-tag" key={concept}>{concept}</span>)}</div>}
        </>}
      </div>
      <div style={{ display: 'flex', justifyContent: isSigner ? 'flex-start' : 'flex-end', marginTop: 7 }}><button className="icon-button" style={{ width: 28, height: 28 }} onClick={() => onSpeak(alternative || message.exactTranscript)} aria-label={`Replay ${isSigner ? 'interpretation' : 'spoken transcript'}`} data-testid={`button-replay-${message.id}`}><Volume2 size={14} /></button></div>
    </div>
  </article>;
}

function ConversationPage() {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [mode, setMode] = useState<'demo' | 'live'>('demo');
  const [camera, setCamera] = useState(false);
  const [cameraStatus, setCameraStatus] = useState<CameraStatus>('disabled');
  const [modelStatus, setModelStatus] = useState<ModelStatus>('idle');
  const [handDetected, setHandDetected] = useState(false);
  const [hands, setHands] = useState<DetectedHand[]>([]);
  const [inferenceFps, setInferenceFps] = useState(0);
  const [inferenceLatency, setInferenceLatency] = useState(0);
  const [microphone, setMicrophone] = useState(false);
  const [debug, setDebug] = useState(false);
  const [draft, setDraft] = useState('');
  const [selectedConcepts, setSelectedConcepts] = useState<string[]>(['HELP']);
  const [alternatives, setAlternatives] = useState<Record<number, string>>({});
  const [scenario, setScenario] = useState('Everyday hello');
  const [speaking, setSpeaking] = useState(false);
  const [backendStatus, setBackendStatus] = useState<'offline' | 'connected' | 'checking'>('checking');
  const [livePrediction, setLivePrediction] = useState<{ label: string; confidence: number } | null>(null);
  const [inferenceHint, setInferenceHint] = useState<string>('Waiting for camera');

  const streamRef = useRef<MediaStream | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const cameraRequestRef = useRef(0);
  const messageListRef = useRef<HTMLDivElement | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const isUserNearBottomRef = useRef(true);

  const sequenceBufferRef = useRef<number[][]>([]);
  const isPredictingRef = useRef(false);
  const lastPredictTimeRef = useRef(0);
  const historyRef = useRef<{ label: string; confidence: number }[]>([]);
  const lastCommittedRef = useRef<{ label: string; timestamp: number } | null>(null);
  const noHandFrameCountRef = useRef<number>(0);
  const hadNeutralStateRef = useRef<boolean>(true);

  const activeSentenceTokensRef = useRef<string[]>([]);
  const lastAcceptedTokenRef = useRef<string | null>(null);
  const [recognizingSessionTokens, setRecognizingSessionTokens] = useState<string[]>([]);

  const scrollToBottom = (smooth = true) => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({
        behavior: smooth ? 'smooth' : 'auto',
        block: 'end',
      });
    }
    if (messageListRef.current) {
      messageListRef.current.scrollTo({
        top: messageListRef.current.scrollHeight,
        behavior: smooth ? 'smooth' : 'auto',
      });
    }
  };

  // Auto-scroll conversation container after newly committed message has rendered
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | null = null;
    if (isUserNearBottomRef.current) {
      timer = setTimeout(() => {
        scrollToBottom(true);
      }, 50);
    }
    return () => {
      if (timer) clearTimeout(timer);
    };
  }, [messages]);

  const handleScroll = () => {
    const el = messageListRef.current;
    if (!el) return;
    const threshold = 100;
    const isAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight <= threshold;
    isUserNearBottomRef.current = isAtBottom;
  };

  useEffect(() => {
    if (mode !== 'live') return;
    let active = true;
    const checkHealth = async () => {
      try {
        const res = await fetch('http://localhost:8000/health');
        if (res.ok) {
          const data = await res.json();
          if (active && data.status === 'ok' && data.model_loaded) {
            setBackendStatus('connected');
            return;
          }
        }
      } catch {
        // API offline
      }
      if (active) setBackendStatus('offline');
    };
    void checkHealth();
    const interval = setInterval(checkHealth, 4000);
    return () => { active = false; clearInterval(interval); };
  }, [mode]);

  const stopCamera = () => {
    cameraRequestRef.current += 1;
    streamRef.current?.getTracks().forEach(track => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCamera(false);
    setCameraStatus('disabled');
    sequenceBufferRef.current = [];
    historyRef.current = [];
    activeSentenceTokensRef.current = [];
    lastAcceptedTokenRef.current = null;
    lastCommittedRef.current = null;
    setRecognizingSessionTokens([]);
    setLivePrediction(null);
    setInferenceHint('Camera off');
    hadNeutralStateRef.current = true;
  };

  const enableCamera = async () => {
    if (streamRef.current || cameraStatus === 'requesting') return;
    if (!navigator.mediaDevices?.getUserMedia || !navigator.mediaDevices.enumerateDevices) {
      setCameraStatus('unsupported');
      return;
    }

    const requestId = cameraRequestRef.current + 1;
    cameraRequestRef.current = requestId;
    setCamera(true);
    setCameraStatus('requesting');

    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      if (!devices.some(device => device.kind === 'videoinput')) {
        if (cameraRequestRef.current === requestId) {
          setCamera(false);
          setCameraStatus('no-device');
        }
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      if (cameraRequestRef.current !== requestId) {
        stream.getTracks().forEach(track => track.stop());
        return;
      }

      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setCameraStatus('streaming');
    } catch (error) {
      if (cameraRequestRef.current !== requestId) return;
      setCamera(false);
      if (error instanceof DOMException && error.name === 'NotAllowedError') setCameraStatus('denied');
      else if (error instanceof DOMException && error.name === 'NotFoundError') setCameraStatus('no-device');
      else if (error instanceof DOMException && ['NotReadableError', 'AbortError'].includes(error.name)) setCameraStatus('unavailable');
      else setCameraStatus('error');
    }
  };

  useEffect(() => {
    return () => {
      cameraRequestRef.current += 1;
      streamRef.current?.getTracks().forEach(track => track.stop());
      streamRef.current = null;
      if (animationFrameRef.current !== null) cancelAnimationFrame(animationFrameRef.current);
      handLandmarkerRef.current?.close();
      handLandmarkerRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (cameraStatus === 'streaming' && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
    }
  }, [cameraStatus]);

  useEffect(() => {
    if (mode !== 'live' || cameraStatus !== 'streaming') return;
    let cancelled = false;
    const load = async () => {
      if (handLandmarkerRef.current) {
        setModelStatus('ready');
        return;
      }
      setModelStatus('loading');
      try {
        const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_ROOT);
        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: HAND_LANDMARKER_MODEL },
          runningMode: 'VIDEO',
          numHands: 2,
        });
        if (cancelled) {
          landmarker.close();
          return;
        }
        handLandmarkerRef.current = landmarker;
        setModelStatus('ready');
      } catch {
        if (!cancelled) setModelStatus('error');
      }
    };
    void load();
    return () => { cancelled = true; };
  }, [cameraStatus, mode]);

  useEffect(() => {
    if (mode !== 'live' || cameraStatus !== 'streaming' || modelStatus !== 'ready') return;
    let cancelled = false;
    let frames = 0;
    let lastFpsAt = performance.now();
    const draw = (detectedHands: DetectedHand[]) => {
      const video = videoRef.current;
      const canvas = overlayRef.current;
      if (!video || !canvas || !video.videoWidth || !video.videoHeight) return;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext('2d');
      if (!context) return;
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.strokeStyle = '#56e0d1';
      context.fillStyle = '#fff3bd';
      context.lineWidth = Math.max(2, canvas.width / 360);
      detectedHands.forEach(hand => {
        HAND_CONNECTIONS.forEach(([start, end]) => {
          const a = hand.landmarks[start]; const b = hand.landmarks[end];
          context.beginPath(); context.moveTo(a.x * canvas.width, a.y * canvas.height); context.lineTo(b.x * canvas.width, b.y * canvas.height); context.stroke();
        });
        hand.landmarks.forEach(point => { context.beginPath(); context.arc(point.x * canvas.width, point.y * canvas.height, Math.max(2, canvas.width / 180), 0, Math.PI * 2); context.fill(); });
      });
    };
    const run = () => {
      if (cancelled) return;
      const video = videoRef.current;
      const landmarker = handLandmarkerRef.current;
      if (video && landmarker && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        try {
          const startedAt = performance.now();
          const result = landmarker.detectForVideo(video, startedAt);
          const detectedHands = result.landmarks.map((landmarks, index) => ({ handedness: result.handedness[index]?.[0]?.categoryName ?? 'Unknown', landmarks: landmarks.map(({ x, y, z }) => ({ x, y, z })) }));
          setHands(detectedHands);
          setHandDetected(detectedHands.length > 0);
          setInferenceLatency(Math.round((performance.now() - startedAt) * 10) / 10);
          draw(detectedHands);
          frames += 1;
          const now = performance.now();
          if (now - lastFpsAt >= 1000) { setInferenceFps(Math.round((frames * 1000) / (now - lastFpsAt))); frames = 0; lastFpsAt = now; }

          const handsPresent = detectedHands.length > 0;
          if (!handsPresent) {
            noHandFrameCountRef.current += 1;
            if (noHandFrameCountRef.current >= 3) {
              hadNeutralStateRef.current = true;
            }
            sequenceBufferRef.current = [];
            historyRef.current = [];
            setLivePrediction(null);

            // Check if active sentence session should finalize when hands are absent for >= 6 consecutive frames (~200ms)
            if (noHandFrameCountRef.current >= 6 && activeSentenceTokensRef.current.length > 0) {
              const tokens = [...activeSentenceTokensRef.current];
              console.log(`[SignBridge ML] FINALIZE: ${tokens.join(' ')}`);

              const exactTranscript = tokens.join('. ') + '.';
              const rawSigns = [...tokens];
              const concepts = [...tokens];
              const sentenceText = tokens.join(' ');
              const interpretation = sentenceText;

              isUserNearBottomRef.current = true;
              setMessages(prev => [...prev, {
                id: Date.now(),
                direction: 'signer',
                exactTranscript,
                rawSigns,
                concepts,
                interpretation,
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
                status: 'Recognized (Live ML)',
              }]);

              // Trigger TTS for finalized sentence
              if ('speechSynthesis' in window) {
                window.speechSynthesis.cancel();
                const utterance = new SpeechSynthesisUtterance(interpretation);
                window.speechSynthesis.speak(utterance);
              }

              // Reset sentence session cleanly
              activeSentenceTokensRef.current = [];
              lastAcceptedTokenRef.current = null;
              lastCommittedRef.current = null;
              setRecognizingSessionTokens([]);
            }

            if (activeSentenceTokensRef.current.length > 0) {
              setInferenceHint(`Hands left frame — finishing sentence: ${activeSentenceTokensRef.current.join(' → ')}`);
            } else {
              setInferenceHint('No hand detected — waiting for sign');
            }
          } else {
            noHandFrameCountRef.current = 0;
            const frameFeat = extractFrameFeatures(detectedHands);
            sequenceBufferRef.current.push(frameFeat);
            if (sequenceBufferRef.current.length > 36) sequenceBufferRef.current.shift();

            if (sequenceBufferRef.current.length < 36) {
              if (activeSentenceTokensRef.current.length > 0) {
                setInferenceHint(`Recognizing: ${activeSentenceTokensRef.current.join(' → ')} (Buffering next sign ${sequenceBufferRef.current.length}/36)`);
              } else {
                setInferenceHint(`Buffering sign movement (${sequenceBufferRef.current.length}/36 frames)`);
              }
            }
          }

          // Query live FastAPI inference service when buffer has 36 continuous hand-present frames
          if (
            handsPresent &&
            backendStatus === 'connected' &&
            sequenceBufferRef.current.length === 36 &&
            !isPredictingRef.current &&
            now - lastPredictTimeRef.current >= 180
          ) {
            isPredictingRef.current = true;
            lastPredictTimeRef.current = now;
            const seqPayload = [...sequenceBufferRef.current];

            fetch('http://localhost:8000/predict', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ landmarks: seqPayload }),
            })
              .then(r => r.ok ? r.json() : null)
              .then(data => {
                if (!data) return;
                setLivePrediction({ label: data.label, confidence: data.confidence });
                historyRef.current.push({ label: data.label, confidence: data.confidence });
                if (historyRef.current.length > 5) historyRef.current.shift();

                // Conservative Gate: confidence >= 0.55 AND at least 4 matching predictions in last 5 with confidence >= 0.55
                const strongMatches = historyRef.current.filter(
                  p => p.label === data.label && p.confidence >= 0.55
                );

                if (data.confidence >= 0.55 && strongMatches.length >= 4) {
                  const lastToken = lastAcceptedTokenRef.current;
                  const isNewToken = lastToken !== data.label;

                  if (isNewToken) {
                    console.log(`[SignBridge ML] TOKEN: ${data.label}`);
                    activeSentenceTokensRef.current.push(data.label);
                    lastAcceptedTokenRef.current = data.label;
                    lastCommittedRef.current = { label: data.label, timestamp: Date.now() };

                    // Reset sequence buffer & smoothing history so next gesture can be detected cleanly without requiring hands to leave frame
                    sequenceBufferRef.current = [];
                    historyRef.current = [];

                    setRecognizingSessionTokens([...activeSentenceTokensRef.current]);
                    setInferenceHint(`Recognizing: ${activeSentenceTokensRef.current.join(' → ')}`);
                  } else {
                    setInferenceHint(`Recognizing: ${activeSentenceTokensRef.current.join(' → ')} (Holding ${data.label})`);
                  }
                } else {
                  if (activeSentenceTokensRef.current.length > 0) {
                    setInferenceHint(`Recognizing: ${activeSentenceTokensRef.current.join(' → ')}`);
                  } else {
                    setInferenceHint('Waiting for clear sign');
                  }
                }
              })
              .catch(() => {
                setBackendStatus('offline');
                setLivePrediction(null);
                setInferenceHint('Inference offline');
              })
              .finally(() => {
                isPredictingRef.current = false;
              });
          }
        } catch {
          setModelStatus('error');
          return;
        }
      }
      animationFrameRef.current = requestAnimationFrame(run);
    };
    animationFrameRef.current = requestAnimationFrame(run);
    return () => {
      cancelled = true;
      if (animationFrameRef.current !== null) cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
      const canvas = overlayRef.current;
      canvas?.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
    };
  }, [backendStatus, cameraStatus, mode, modelStatus]);

  const speak = (text: string) => {
    if (!('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.onstart = () => setSpeaking(true);
    utterance.onend = () => setSpeaking(false);
    window.speechSynthesis.speak(utterance);
  };
  const sendText = () => {
    const text = draft.trim();
    if (!text) return;
    const extractedConcepts = extractConcepts(text);
    const emergency = extractedConcepts.includes('HELP') && (extractedConcepts.includes('DOCTOR') || extractedConcepts.includes('HOSPITAL'));
    isUserNearBottomRef.current = true;
    setMessages(prev => [...prev, {
      id: Date.now(),
      direction: 'hearing',
      exactTranscript: text,
      concepts: extractedConcepts,
      interpretation: emergency ? 'I need a doctor.' : text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
      status: emergency ? 'Emergency context' : 'Spoken',
      emergency,
      textToIsl: true,
    }]);
    setDraft('');
  };
  const runScenario = (name: string) => {
    setScenario(name);
    isUserNearBottomRef.current = true;
    if (name === 'Everyday hello') setMessages(initialMessages);
    if (name === 'Need assistance') setMessages([...initialMessages.slice(0, 1), { id: 3, direction: 'signer', exactTranscript: 'HELP. DOCTOR?', rawSigns: ['HELP', 'DOCTOR'], concepts: ['HELP', 'DOCTOR'], interpretation: 'I need help. Is there a doctor nearby?', alternatives: ['Please help me find a doctor.', 'I need medical help.'], timestamp: '10:42:01', status: 'Interpreted', emergency: true }]);
    if (name === 'Clarify a phrase') setMessages([{ id: 4, direction: 'signer', exactTranscript: 'LATER. HOME.', rawSigns: ['LATER', 'HOME'], concepts: ['LATER', 'HOME'], interpretation: 'I will go home later.', alternatives: ['Later, at home.', 'I am going home afterwards.'], timestamp: '10:43:16', status: 'Needs review' }]);
  };
  const toggleConcept = (label: string) => setSelectedConcepts(prev => prev.includes(label) ? prev.filter(item => item !== label) : [...prev, label]);

  return <Shell><div className="page">
    <Topbar detail="Conversation / local session" />
    <div className="workspace-head"><div><div className="eyebrow">Two-way workspace</div><h1 className="page-title">A conversation in full.</h1><p className="page-subtitle">Detected signs, exact words, and interpretation are kept together. Use demo mode to explore the flow without connecting devices.</p></div><div className="mode-switch" role="tablist" aria-label="Conversation mode"><button className={mode === 'demo' ? 'selected' : ''} onClick={() => { stopCamera(); setMode('demo'); }} role="tab" aria-selected={mode === 'demo'} data-testid="button-mode-demo">Demo mode</button><button className={mode === 'live' ? 'selected' : ''} onClick={() => setMode('live')} role="tab" aria-selected={mode === 'live'} data-testid="button-mode-live">Live devices</button></div></div>
    <div className="workspace">
      <section className="panel conversation-panel" aria-label="Conversation transcript">
        <div className="panel-head"><div className="panel-heading"><AudioLines size={17} /> Conversation thread</div><span className={`live-badge ${mode === 'demo' ? 'demo-badge' : ''}`} data-testid="status-conversation-mode">{mode === 'demo' ? 'Demo mode' : 'Live session'}</span></div>
        <div className="message-list" ref={messageListRef} onScroll={handleScroll}>
          {messages.length === 0 ? <div className="empty-state"><AudioLines size={26} /><h3>A quiet beginning</h3><p>Start with a demo scenario or type a message below.</p></div> : messages.map(message => <MessageBubble key={message.id} message={message} alternative={alternatives[message.id]} onAlternative={value => setAlternatives(prev => ({ ...prev, [message.id]: value }))} onSpeak={speak} />)}
          <div ref={messagesEndRef} key="messages-end-sentinel" style={{ height: 1 }} />
        </div>
        <div className="compose">
          <div className="compose-top"><textarea value={draft} onChange={event => setDraft(event.target.value)} onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); sendText(); } }} placeholder="Type an English reply…" aria-label="Type an English reply" data-testid="input-message" /><button className="button primary" onClick={sendText} disabled={!draft.trim()} aria-label="Send typed reply" data-testid="button-send-message"><Send size={15} /></button></div>
          <div className="compose-actions"><span className="compose-hint">Enter to send · Shift + Enter for a new line</span><button className="button soft small" onClick={() => speak('Your reply is ready to be spoken.')} data-testid="button-speak-test">{speaking ? <Pause size={13} /> : <Volume2 size={13} />} {speaking ? 'Speaking…' : 'Test voice'}</button></div>
        </div>
      </section>
      <aside className="side-stack">
        <section className="panel side-panel"><div className="eyebrow">Input status</div><h3>Connection controls</h3>
          <div className="device-row"><div className="device-info"><Camera size={16} className="device-icon" /><span>Camera for ISL</span></div><button className={`toggle ${camera ? 'on' : ''}`} onClick={camera ? stopCamera : enableCamera} disabled={mode === 'demo'} aria-label={`${camera ? 'Disable' : 'Enable'} camera`} aria-pressed={camera} data-testid="toggle-camera"><i /></button></div>
          <div className="device-row"><div className="device-info"><Mic size={16} className="device-icon" /><span>Microphone</span></div><button className={`toggle ${microphone ? 'on' : ''}`} onClick={() => setMicrophone(!microphone)} aria-label={`${microphone ? 'Disable' : 'Enable'} microphone`} aria-pressed={microphone} data-testid="toggle-microphone"><i /></button></div>
          {mode === 'live' && cameraStatus === 'streaming' && <div style={{ position: 'relative', marginTop: 14 }}><video ref={videoRef} autoPlay playsInline muted aria-label="Live camera preview" data-testid="video-camera-preview" style={{ width: '100%', display: 'block', borderRadius: 10, background: 'hsl(195 24% 14%)' }} /><canvas ref={overlayRef} aria-hidden="true" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }} />{recognizingSessionTokens.length > 0 && <div style={{ position: 'absolute', bottom: 8, left: 8, right: 8, padding: '6px 10px', background: 'rgba(0, 0, 0, 0.75)', color: '#56e0d1', borderRadius: 6, fontSize: 12, fontWeight: 600, backdropFilter: 'blur(4px)' }} data-testid="live-recognizing-preview">Recognizing: {recognizingSessionTokens.join(' → ')}</div>}</div>}
          <div className="availability"><strong>{mode === 'demo' ? 'DEMO MODE is on' : 'LIVE CAMERA'}</strong>{mode === 'demo' ? 'No camera or microphone data is being captured. Turn on Live devices when you are ready.' : cameraStatus === 'disabled' ? 'Camera is off. Enable it to request access.' : cameraStatus === 'requesting' ? 'Requesting camera permission…' : cameraStatus === 'streaming' ? `Camera stream is active. Hand landmarker: ${modelStatus === 'ready' ? handDetected ? `hand detected (${hands.length}/2)` : 'hand not detected' : modelStatus === 'loading' ? 'loading…' : modelStatus === 'error' ? 'error' : 'waiting'}. Inference backend: ${backendStatus === 'connected' ? `Connected (${inferenceHint})` : 'Inference offline (FastAPI backend unavailable at http://localhost:8000)'}.` : cameraStatus === 'unsupported' ? 'This browser does not support camera access.' : cameraStatus === 'denied' ? 'Camera permission was denied. Allow camera access in your browser settings and try again.' : cameraStatus === 'no-device' ? 'No camera device was found.' : cameraStatus === 'unavailable' ? 'The camera is unavailable or already in use by another application.' : 'Unable to start the camera. Please try again.'}</div>
        </section>
        <section className="panel side-panel"><div className="eyebrow">Try a scene</div><h3>Demo scenarios</h3><div className="scenario-row">{['Everyday hello', 'Need assistance', 'Clarify a phrase'].map(item => <button className="scenario-button" key={item} onClick={() => runScenario(item)} data-testid={`button-scenario-${item.toLowerCase().replaceAll(' ', '-')}`}><strong>{item}</strong><span>{item === 'Need assistance' ? 'An urgent request with visible context.' : item === 'Clarify a phrase' ? 'Compare two possible meanings.' : 'A simple opening exchange.'}</span></button>)}</div><div className="availability"><strong>Now showing: {scenario}</strong>{messages[messages.length - 1]?.status || 'No messages yet'} · {messages.length} messages</div></section>
        <section className="panel side-panel"><div className="eyebrow">Concept palette</div><h3>Tap to add context</h3><div className="selected-concepts">{selectedConcepts.map(label => <button className="concept-tag" key={label} onClick={() => toggleConcept(label)} aria-label={`Remove ${label} concept`} data-testid={`button-remove-concept-${label.toLowerCase().replaceAll(' ', '-')}`}>{label} <X size={10} style={{ verticalAlign: 'middle' }} /></button>)}</div><div className="concept-line" style={{ marginTop: 11 }}>{concepts.slice(0, 6).map(concept => <button className="alternative-button" key={concept.id} onClick={() => toggleConcept(concept.label)} data-testid={`button-concept-${concept.id}`}>{concept.label}</button>)}</div></section>
        <section className="panel side-panel debug-panel"><div className="eyebrow">Optional transparency</div><h3>Developer diagnostics</h3><button className="button small" style={{ marginTop: 12, background: 'hsl(195 24% 29%)', color: 'hsl(40 33% 96%)', borderColor: 'hsl(195 24% 35%)' }} onClick={() => setDebug(!debug)} data-testid="button-toggle-diagnostics">{debug ? 'Hide diagnostics' : 'Show diagnostics'} <Terminal size={13} /></button>{debug && <div><div className="debug-line"><span>capture.state</span><b>{mode === 'demo' ? 'sandbox' : cameraStatus}</b></div><div className="debug-line"><span>hand.model</span><b>{mode === 'demo' ? 'inactive' : modelStatus}</b></div><div className="debug-line"><span>hand.detected</span><b>{mode === 'live' && modelStatus === 'ready' ? handDetected ? 'yes' : 'no' : 'unavailable'}</b></div><div className="debug-line"><span>handedness</span><b>{hands.map(hand => hand.handedness).join(', ') || '—'}</b></div><div className="debug-line"><span>landmarks</span><b>{hands.map(hand => hand.landmarks.length).join(', ') || '0'} points/hand</b></div><div className="debug-line"><span>inference.ms</span><b>{modelStatus === 'ready' ? `${inferenceFps} FPS · ${inferenceLatency} ms` : '—'}</b></div><div className="debug-line"><span>inference.backend</span><b>{backendStatus === 'connected' ? 'connected (FastAPI :8000)' : 'Inference offline'}</b></div><div className="debug-line"><span>inference.status</span><b>{inferenceHint}</b></div><div className="debug-line"><span>prediction.live</span><b>{livePrediction ? `${livePrediction.label} (${Math.round(livePrediction.confidence * 100)}%)` : 'none'}</b></div><div className="debug-line"><span>audio.output</span><b>{'speechSynthesis' in window ? 'supported' : 'unavailable'}</b></div></div>}</section>
      </aside>
    </div>
  </div></Shell>;
}

function VocabularyPage() {
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState('All');
  const [playing, setPlaying] = useState<string | null>(null);
  const videoRefs = useRef<Record<string, HTMLVideoElement | null>>({});
  const filters = ['All', 'Conversation', 'Needs', 'Everyday', 'Emergency', 'Time'];
  const visible = useMemo(() => concepts.filter(item => (filter === 'All' || item.category === filter) && `${item.label} ${item.meaning}`.toLowerCase().includes(query.toLowerCase())), [filter, query]);

  const toggleDemo = (concept: SignConcept) => {
    const vid = videoRefs.current[concept.id];
    if (playing === concept.id) {
      vid?.pause();
      setPlaying(null);
    } else {
      // Pause any currently playing video
      if (playing) {
        videoRefs.current[playing]?.pause();
      }
      if (vid) {
        vid.currentTime = 0;
        vid.play().catch(() => {});
      }
      setPlaying(concept.id);
    }
  };

  return <Shell><div className="page">
    <Topbar detail="Reference / Indian Sign Language" />
    <div className="workspace-head"><div><div className="eyebrow">A shared reference</div><h1 className="page-title">Words with a shape.</h1><p className="page-subtitle">Browse the ISL sign videos available in this workspace. Press Preview on any card to watch the reference sign.</p></div><div className="presence"><BookOpen size={14} /> {concepts.length} concepts</div></div>
    <div className="vocab-toolbar"><div className="search-wrap"><Search size={16} /><input className="search-input" type="search" value={query} onChange={event => setQuery(event.target.value)} placeholder="Search a concept or meaning" aria-label="Search vocabulary" data-testid="input-vocabulary-search" /></div><div className="filter-set" role="group" aria-label="Filter vocabulary">{filters.map(item => <button key={item} className={`filter-chip ${filter === item ? 'selected' : ''}`} onClick={() => setFilter(item)} aria-pressed={filter === item} data-testid={`button-filter-${item.toLowerCase()}`}>{item}</button>)}</div></div>
    {visible.length === 0 ? <div className="empty-state"><Search size={26} /><h3>No concepts found</h3><p>Try a different word or clear the {filter} filter.</p><button className="button soft small" style={{ marginTop: 17 }} onClick={() => { setQuery(''); setFilter('All'); }} data-testid="button-clear-vocabulary">Clear search</button></div> : <section className="vocab-grid" aria-label="Vocabulary results">{visible.map(concept => <article className="vocab-card" key={concept.id} data-testid={`card-vocabulary-${concept.id}`}><div className="vocab-top"><span className="category-label">{concept.category}</span>{concept.priority === 'high' && <span className="priority" title="High priority concept" />}</div><div className="demo-art" aria-label={`ISL sign video for ${concept.label}`} style={{ position: 'relative', overflow: 'hidden', background: 'hsl(195 24% 10%)' }}><video ref={el => { videoRefs.current[concept.id] = el; }} src={`/signs/${concept.id}.mp4`} muted playsInline loop onEnded={() => setPlaying(null)} style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }} aria-label={`${concept.label} sign video`} /></div><h3>{concept.label}</h3><p>{concept.meaning}</p><div className="vocab-bottom"><span>{Math.round(concept.confidence * 100)}% reference confidence</span>{concept.demoAvailable ? <button onClick={() => toggleDemo(concept)} data-testid={`button-play-demo-${concept.id}`}>{playing === concept.id ? <><Pause size={11} /> Playing</> : <><Play size={11} /> Preview</>}</button> : <span>Coming with assets</span>}</div></article>)}</section>}
    <div className="availability" style={{ marginTop: 30, maxWidth: 700 }}><strong>About these demonstrations</strong> Real ISL reference videos are shown directly from the local assets. Each clip was recorded for the SignBridge vocabulary set.</div>
  </div></Shell>;
}

function Router() {
  return <Switch><Route path="/" component={HomePage} /><Route path="/conversation" component={ConversationPage} /><Route path="/vocabulary" component={VocabularyPage} /><Route component={NotFound} /></Switch>;
}

function RoutedErrorBoundary({ children }: { children: ReactNode }) {
  const [location] = useLocation();
  return <ErrorBoundary resetKey={location}>{children}</ErrorBoundary>;
}

function App() {
  return <QueryClientProvider client={queryClient}><TooltipProvider><WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}><RoutedErrorBoundary><Router /></RoutedErrorBoundary></WouterRouter><Toaster /></TooltipProvider></QueryClientProvider>;
}

export default App;
