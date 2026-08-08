# Dataset audit

Audit updated: 2026-08-08. Counts and terms below come from primary papers, official
project pages, or official repositories where available. “Not verified” is intentional:
restricted or ambiguous resources must be confirmed before a manifest is built.

| Dataset | Official source / access | Languages | Modalities and task | Size / speakers / labels | Metadata, license, and main risks |
|---|---|---|---|---|---|
| ESD | [HLT Singapore repository](https://github.com/HLTSingapore/Emotional-Speech-Data); request form and data-use agreement | English, Mandarin | Audio plus fixed transcripts; utterance-level; **no video** | 350 parallel utterances, 10 native speakers per language, five emotions (`neutral`, `happy`, `angry`, `sad`, `surprise`), commonly 35,000 recordings | Speaker IDs exist. Research-use data agreement; repository code license is not the data license. Good bilingual control, but trimodal claims would be false. Split by speaker. |
| IEMOCAP | [USC SAIL](https://sail.usc.edu/iemocap/); licensed request | English | Audiovisual recordings, speech, face motion, transcripts; dyadic scripted/improvised dialogue annotated at utterance/turn level | About 12 hours, 10 actors in five sessions; categorical and dimensional emotion labels | Speaker/session IDs and corpus metadata exist; country/region/accent not verified. Treat clips independently but group splits by session, speaker, and original recording. Common four-class mappings must be explicit. |
| CREMA-D | [Official repository](https://github.com/CheyneyComputerScience/CREMA-D); public Git LFS | English | Real audio and video; the 12 sentence texts can be reconstructed from official sentence IDs; acted utterance-level clips | 7,442 clips; 91 actors (48 male, 43 female); anger, disgust, fear, happy, neutral, sad; four intensity levels | Actor IDs/demographics exist. Database ODbL and contents DBCL. Speaker split is mandatory. Repeated sentences create lexical duplicate risk. |
| EmotionTalk | [Official Hugging Face dataset](https://huggingface.co/datasets/BAAI/Emotiontalk) and [paper](https://aclanthology.org/2026.findings-acl.440/); gated contact-information agreement | Mandarin Chinese | Audio, text, and video from dyadic conversations; current project extraction uses audio plus its matching transcript only | 23.6 hours, 19,250 utterances, 19 actors, seven utterance-level emotions | CC BY-NC-SA 4.0. Local audit found 16,413 angry/happy/neutral/sad samples. Split connected speaker components and protect original source groups. Language is confounded with corpus in a CREMA-D/EmotionTalk comparison. |
| EmoS | [Official repository](https://github.com/EmotionVerse/EmoS) and ACL paper; upstream data required | English, Mandarin | Static trimodal clips plus streaming video; emotion recognition | 9,403 static samples (5,000 MELD English + 4,403 CH-SIMS v2 Chinese), plus about two streaming hours/50 videos; basic seven and fine-grained labels | Source media are not redistributed; obtain and obey MELD/CH-SIMS terms. English and Mandarin are completely tied to different source corpora. Dialogue/source-video grouping is necessary. |
| MMAFFBen | [Official repository](https://github.com/DavidMChan/MMAFFBen); code Apache-2.0, upstream data terms apply | 35 languages across aggregated tasks | Text/image/video affect benchmark; modalities/languages are not necessarily aligned as one trimodal utterance sample | Multiple constituent datasets and four affect tasks; no single shared speaker/count definition | Useful benchmark/catalog, not a clean bilingual trimodal training corpus. Audit every constituent license and do not infer absent raw streams. |
| M-MELD | [Primary paper](https://arxiv.org/abs/2505.10406); verify release before use | English, Greek, Polish, French, Spanish; **no Mandarin** | Multilingual extension of MELD; conversation/ERC structure | Derived from MELD; source split and exact released assets must be checked | Unsuitable as the primary English–Mandarin dataset and would import conversational assumptions. Source-TV and dialogue leakage remain risks. |
| MELD | [Official repository](https://github.com/declare-lab/MELD) | English | MP4 utterance clips with audio/video/transcript; multi-party dialogue/ERC | 13,708 utterances, 1,433 dialogues, seven emotions; official train/dev/test split | Dialogue, speaker and television episode/source leakage require group-aware analysis. Copyrighted source media and project terms must be reviewed. |
| CASIA Chinese emotional speech | Official release/license page was not verified; do not download from mirrors | Mandarin | Audio-only acted speech; transcripts/release fields not verified | Widely reported as 1,200 utterances, four actors, six emotions, but these values remain provisional until checked against an official distribution | Not trimodal. Accessibility and redistribution terms are unresolved. Do not make it a default data dependency. |
| CHEAVD | [CASIA/Institute of Automation project page](http://www.nlpr.ia.ac.cn/english/irds/Resources/201501/t20150121_138424.html); access terms must be confirmed | Mandarin | Natural emotional audio-visual material; multimodal annotation | Exact usable clip, transcript, speaker, and label counts must be verified from the licensed release | Potential Mandarin source, but availability and split grouping must be established before use. Never conflate it with CASIA emotional speech. |
| SAMSEMO | [Samsung Research project](https://research.samsung.com/research-papers/SAMSEMO-A-Multi-Modal-Multi-Lingual-and-Multi-Label-Emotion-Dataset) | English, German, Spanish, Polish, Korean; **no Mandarin** | Video scenes and multimodal/multilabel annotation | 23,086 scenes in five languages | Annotation/metadata terms are CC BY-NC-SA 4.0; raw media may have separate rights. Not an English–Mandarin solution. |

## Required leakage and confounding controls

1. Split by speaker before class balancing or cache generation.
2. Keep all segments from one original recording/video in one split.
3. Reject reused audio/video paths and sample IDs across splits.
4. Flag exact and near-duplicate transcripts across splits. Repeated acted scripts may be
   legitimate, but must be reported and cannot substitute for acoustic/visual evidence.
5. Report a language-by-corpus contingency table. If a language occurs in only one
   corpus, language and corpus effects are not separately identifiable.
6. Preserve modality availability; no synthetic visual features for audio-only datasets.

## Conclusion

There is no verified drop-in dataset that simultaneously provides unrestricted English
and Mandarin, real audio/text/video, clean speaker IDs, and an unconfounded acquisition
protocol. The initial code therefore consumes a neutral manifest, works without real
data, and makes cross-corpus limitations observable rather than hiding them.
