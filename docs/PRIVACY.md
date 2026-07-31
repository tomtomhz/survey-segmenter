# Privacy and data protection

Survey Segmenter processes survey responses, which in the EU and EEA means personal data under the
GDPR. This document states exactly what the software does with that data, so the answer to "where
does our respondent data go?" is a document rather than a recollection.

## The short version

**Your survey file never leaves the computer it is opened on.** The analysis runs entirely locally,
on `127.0.0.1`. There is no server, no account, no telemetry, and no analytics.

One optional feature transmits anything at all: the Claude interpretation layer. When — and only
when — a user has entered their own Anthropic API key, the tool sends the **aggregate result
summary** to Anthropic. Never an individual's answers.

## What is processed, and where

| Data | Where it is processed | Where it is stored | Leaves the machine? |
|---|---|---|---|
| The uploaded survey file | In memory, locally | Copied into the project folder so you can re-group without re-uploading | No |
| Segment assignments, profiles, typing rule | Locally | `~/.survey_segmenter/projects/` | No |
| Aggregate report digest (segment sizes, mean scores, stability statistics, demographic percentages) | Locally | — | **Only** to Anthropic, and only with a user-supplied API key |
| Anthropic API key | — | `~/.survey_segmenter/config.json`, permissions `600` | No |

## What is never sent to Anthropic

The digest handed to Claude is aggregate by construction. It contains no respondent identifiers, no
individual response rows, and no free-text answers.

This is verified by an automated test rather than asserted in prose: the suite builds a 400-person
dataset, generates the digest, and fails the build if any respondent identifier or free-text answer
appears in the payload. See `test_the_ai_digest_contains_no_individual_respondent_data` in
`tests/test_segment_kmeans.py`.

If no API key is configured, the tool never contacts any network service, and the statistics work
in full.

## GDPR notes for whoever runs the study

- **Lawful basis / purpose limitation.** The tool performs analysis on data you have already
  collected. It does not create a new collection purpose, and it adds no new recipients unless the
  Claude layer is switched on.
- **Data minimisation.** Only aggregate results are ever transmitted, and only on the optional
  path. The design choice to send a digest rather than the dataset was deliberate.
- **Third-country transfer.** Enabling the Claude layer means an aggregate summary is sent to
  Anthropic. Even though it contains no personal data, treat switching it on as a decision to
  document, not a default.
- **Storage limitation and erasure.** Projects persist under `~/.survey_segmenter/projects/`.
  Deleting a project in the app removes all three of its files. To clear everything, delete that
  folder. There is no other copy and no remote backup, so an erasure request is satisfied locally.
- **Pseudonymisation.** The tool identifies respondents only by whatever id column your export
  contains. If you pseudonymise before export, nothing downstream needs the real identity —
  segment assignments and the typing rule work purely on the id you supply.
- **Records of processing.** The report footer stamps the tool version, and each project stores the
  raw input alongside its results, so any published segmentation can be reproduced and evidenced.

## Reporting a problem

If you find a case where respondent-level data leaves the machine, treat it as a security issue
rather than a bug: stop using the affected version and raise it directly with the maintainer.
