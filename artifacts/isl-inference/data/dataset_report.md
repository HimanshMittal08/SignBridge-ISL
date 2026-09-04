# SignBridge ISL Dataset Availability Report

## Dataset Summary
- **Dataset Source:** `vidit031/isl-isolated-8words` (huggingface_hub)
- **Dataset Location:** `artifacts/isl-inference/scratch/raw_hf_dataset/`
- **Total Product Target Concepts:** 70
- **Available Dataset Concepts:** 8
- **Missing Product Concepts:** 62
- **Total Raw Videos Discovered:** 56
- **Total Usable Videos (MediaPipe landmark extracted):** 56
- **Rejected Videos / Failures:** 0

---

## 70-Target Vocabulary Mapping Breakdown

| Target Concept | Product Category | Dataset Label | Videos Available | Usable Samples | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HELLO** | Greetings & Courtesy | `hello` | 7 | 7 | Available (Insufficient baseline samples) |
| **GOOD_MORNING** | Greetings & Courtesy | — | 0 | 0 | Missing |
| **GOOD_NIGHT** | Greetings & Courtesy | — | 0 | 0 | Missing |
| **THANK_YOU** | Greetings & Courtesy | — | 0 | 0 | Missing |
| **PLEASE** | Greetings & Courtesy | `please` | 7 | 7 | Available (Insufficient baseline samples) |
| **SORRY** | Greetings & Courtesy | — | 0 | 0 | Missing |
| **WELCOME** | Greetings & Courtesy | — | 0 | 0 | Missing |
| **BYE** | Greetings & Courtesy | — | 0 | 0 | Missing |
| **YES** | Basic Responses | `yes` | 7 | 7 | Available (Insufficient baseline samples) |
| **NO** | Basic Responses | `no` | 7 | 7 | Available (Insufficient baseline samples) |
| **OK** | Basic Responses | — | 0 | 0 | Missing |
| **MAYBE** | Basic Responses | — | 0 | 0 | Missing |
| **GOOD** | Basic Responses | — | 0 | 0 | Missing |
| **BAD** | Basic Responses | — | 0 | 0 | Missing |
| **RIGHT** | Basic Responses | — | 0 | 0 | Missing |
| **WRONG** | Basic Responses | — | 0 | 0 | Missing |
| **FOOD** | Food & Drink | — | 0 | 0 | Missing |
| **EAT** | Food & Drink | `eat` | 7 | 7 | Available (Insufficient baseline samples) |
| **DRINK** | Food & Drink | — | 0 | 0 | Missing |
| **WATER** | Food & Drink | `water` | 7 | 7 | Available (Insufficient baseline samples) |
| **TEA** | Food & Drink | — | 0 | 0 | Missing |
| **MILK** | Food & Drink | — | 0 | 0 | Missing |
| **HUNGRY** | Food & Drink | — | 0 | 0 | Missing |
| **THIRSTY** | Food & Drink | — | 0 | 0 | Missing |
| **HELP** | Needs & Help | `help` | 7 | 7 | Available (Insufficient baseline samples) |
| **NEED** | Needs & Help | — | 0 | 0 | Missing |
| **WANT** | Needs & Help | — | 0 | 0 | Missing |
| **DONT_WANT** | Needs & Help | — | 0 | 0 | Missing |
| **SLEEP** | Needs & Help | — | 0 | 0 | Missing |
| **REST** | Needs & Help | — | 0 | 0 | Missing |
| **BATHROOM** | Needs & Help | — | 0 | 0 | Missing |
| **MEDICINE** | Needs & Help | — | 0 | 0 | Missing |
| **GO** | Actions | `go` | 7 | 7 | Available (Insufficient baseline samples) |
| **COME** | Actions | — | 0 | 0 | Missing |
| **STOP** | Actions | — | 0 | 0 | Missing |
| **WAIT** | Actions | — | 0 | 0 | Missing |
| **SIT** | Actions | — | 0 | 0 | Missing |
| **STAND** | Actions | — | 0 | 0 | Missing |
| **GIVE** | Actions | — | 0 | 0 | Missing |
| **TAKE** | Actions | — | 0 | 0 | Missing |
| **OPEN** | Actions | — | 0 | 0 | Missing |
| **CLOSE** | Actions | — | 0 | 0 | Missing |
| **ME** | People / Pronouns | — | 0 | 0 | Missing |
| **YOU** | People / Pronouns | — | 0 | 0 | Missing |
| **WE** | People / Pronouns | — | 0 | 0 | Missing |
| **MOTHER** | People / Pronouns | — | 0 | 0 | Missing |
| **FATHER** | People / Pronouns | — | 0 | 0 | Missing |
| **FRIEND** | People / Pronouns | — | 0 | 0 | Missing |
| **DOCTOR** | People / Pronouns | — | 0 | 0 | Missing |
| **WHAT** | Questions | — | 0 | 0 | Missing |
| **WHERE** | Questions | — | 0 | 0 | Missing |
| **WHO** | Questions | — | 0 | 0 | Missing |
| **WHEN** | Questions | — | 0 | 0 | Missing |
| **WHY** | Questions | — | 0 | 0 | Missing |
| **HOW** | Questions | — | 0 | 0 | Missing |
| **WHICH** | Questions | — | 0 | 0 | Missing |
| **TODAY** | Time & Context | — | 0 | 0 | Missing |
| **TOMORROW** | Time & Context | — | 0 | 0 | Missing |
| **YESTERDAY** | Time & Context | — | 0 | 0 | Missing |
| **NOW** | Time & Context | — | 0 | 0 | Missing |
| **LATER** | Time & Context | — | 0 | 0 | Missing |
| **HOME** | Time & Context | — | 0 | 0 | Missing |
| **EMERGENCY** | Emergency | — | 0 | 0 | Missing |
| **DANGER** | Emergency | — | 0 | 0 | Missing |
| **POLICE** | Emergency | — | 0 | 0 | Missing |
| **HOSPITAL** | Emergency | — | 0 | 0 | Missing |
| **FIRE** | Emergency | — | 0 | 0 | Missing |
| **CALL** | Emergency | — | 0 | 0 | Missing |
| **PAIN** | Emergency | — | 0 | 0 | Missing |
| **SAVE_ME** | Emergency | — | 0 | 0 | Missing |

---

## Signer Distribution
- `USER001` .. `USER006`: 43 clips total (ISL500)
- `INCLUDE_MVI`: 6 clips total (INCLUDE dataset)
- `CISLR`: 7 clips total (CISLR dataset)

## Data Quality & Balance
- **Class Balance:** Perfectly balanced across the 8 available classes (7 clips each).
- **Data Volume:** 7 samples per class is insufficient for production generalization across unseen signers without data augmentation and regularized training.
