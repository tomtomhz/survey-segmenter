# Survey Segmenter — how to install and use

Survey Segmenter turns a survey export into clear customer groups. You drop your file into a
chat-style page (it looks and feels like Claude), and it finds the segments, rates how much to
trust them with a green / amber / red light, **draws the data so you can judge the groups for
yourself**, and — if you add an Anthropic API key — has **Claude read the results and explain what
they mean for your team**, then answers your follow-up questions.

The charts are there on purpose. Clustering always returns the number of groups you ask for, even
when the answers are random, so "we found three segments" can be true and meaningless at the same
time. Looking at the picture is how anyone in the room — not just the person who ran it — can tell
the difference, and it is the check on the write-up when a summary (Claude's or anyone's) reads
more confidently than the data deserves.

**Everything runs on your own computer.** Your survey file never leaves your machine. The only thing
ever sent anywhere is the *aggregate* result summary (group sizes, profiles, confidence — never
anyone's individual answers), and only to Claude, only when you have added your own API key.

You do **not** need Python or any technical setup to run the app. Just download and double-click.

---

## Mac

1. Download **`Survey Segmenter.zip`**.
2. **Move the .zip to your Applications folder before unzipping it, and unzip it there.**

   > **Not on the Desktop or in Documents, if you use iCloud Drive.** macOS syncs those two
   > folders by default, and iCloud writes its own hidden metadata onto everything inside them.
   > On an app bundle that metadata *invalidates the code signature*, and macOS then refuses to
   > launch it — the app simply will not open, and nothing tells you why. It is not damaged; it is
   > being edited by the sync service. Unzipping anywhere outside iCloud avoids it entirely.
   >
   > If it has already happened, do not try to repair that copy. Delete it and unzip again in
   > Applications.

   You will get **`Survey Segmenter.app`**.
3. **The first time only — macOS will block it, and that is expected.** The app is made by your own
   team rather than sold through Apple's App Store, so macOS asks you to approve it once:

   a. Double-click the app. A message appears saying it cannot be opened. Click **Done**.
   b. Open **System Settings → Privacy & Security**, and scroll down to the **Security** section.
   c. You will see *"Survey Segmenter was blocked to protect your Mac"*. Click **Open Anyway**.
   d. Confirm with Touch ID or your password, then click **Open Anyway** once more.

   That is a one-time approval. From then on it opens normally with a double-click.

   > On older macOS versions you may instead be able to right-click the app and choose **Open**.
   > If that option works for you, it does the same thing.
4. Your web browser opens with the **Survey Segmenter** chat page.
5. Attach your survey file — click the **paperclip**, or just **drag the file anywhere onto the
   page** (a `.csv` or `.xlsx` from Google Forms, Typeform, Qualtrics, SurveyMonkey, or a
   spreadsheet). The moment you drop it, it starts analysing.
6. Read the result: how many groups, who they are, and a **confidence rating**. Open **See the data
   yourself** for the charts, and **full statistical report** for all the detail.
7. If you have turned on AI interpretation (see below), **Claude** then writes a plain-language
   readout and recommendations. **Type a question** in the box to ask anything — "which group should
   we target first?", "write a landing-page headline for group 2", "how confident should I be?".
8. Use **+ New** to start over with another file, or **Quit** to close the app.

### What you can do with the result

Under the report you get five things, and they are the point of the whole exercise:

- **See the data yourself.** Four charts, so you never have to take the write-up on trust:
  - **Segment map** — every respondent as a dot, coloured by group. If the colours form their own
    clumps the groups are real. If it is one cloud sliced up like a pie, the method invented them.
    This is the fastest way to catch a wrong conclusion, whoever or whatever drew it.
  - **Who belongs** — how well each individual person fits the group they were put in.
  - **How many groups** — whether the number of groups was an obvious call or a coin flip.
  - **What differs** — the questions the groups actually disagree about, side by side.
- **Take it away.** Download **who is in which group** (a CSV of every person and their group),
  **what defines each group**, and the **scoring rule**. Load the first one straight into your CRM,
  an ad audience, or a mail tool — that is how the segments turn into campaigns.
- **Name the groups.** The automatic labels are built from question codes and are useless in a
  brief. Type names your team recognises, or press **Suggest names with Claude**. The names go into
  the downloads.
- **Score new people.** Upload a file of people who were *not* in the study and the app puts each of
  them into one of your existing groups. Run the survey once, then keep classifying every new
  signup — no re-analysis needed.
- **Group people on different questions.** The app tells you which questions it used. If it set
  aside something you care about, tick it and press **Re-group**.

> If double-clicking ever seems to do nothing, wait ten seconds (the first launch is slow while the
> app starts up), then check your browser.

---

## Turning on Claude's interpretation (optional)

The statistics work on their own. To also have Claude interpret the results and answer questions:

1. Get an Anthropic API key from **console.anthropic.com** (your own account).
2. In the app, click **Settings** (top right), paste the key (it looks like `sk-ant-...`), and click
   **Save key**.
3. That's it — from now on, every analysis is followed by Claude's interpretation, and you can chat.

The key is stored only on your computer (in a small settings file in your home folder) and is used
only to talk to Claude under your own account. You can remove it any time in Settings. If your
organisation sets the `ANTHROPIC_API_KEY` environment variable, the app uses that automatically and
you can skip the Settings step.

---

## Windows

A ready-made Windows app has to be built on a Windows machine. Ask whoever set this up to run the
one-line build there, or use the "run from source" steps below (they work on any computer with
Python).

## Run from source (any computer, for the technically comfortable)

1. Install Python 3 from python.org (if it is not already installed).
2. Open a terminal in this folder and run:
   ```
   pip install ".[ai,excel]"
   segment-kmeans --serve
   ```
   Your browser opens the same chat page as the app. (`ai` adds Claude interpretation; `excel` adds
   `.xlsx` reading. Leave `[ai,excel]` off for just the statistics.)

---

## What file do I give it?

Any survey export with **one row per person** and the answers in columns. It automatically:

- finds the respondent id and ignores timestamps and free-text comments,
- turns "Strongly agree / Agree / ..." answers into numbers,
- sets aside background traits like gender, age, and university (it describes the groups by them
  afterwards; it does not group people on them),
- reads comma **or** semicolon files, Swedish characters (å ä ö), and Excel `.xlsx`,
- chooses the right method for your data and checks how much to trust the result.

If something is wrong with the file, the app tells you in plain language what to fix.

## Is my data private?

Yes. The app runs entirely on your own computer on a local address (`localhost`). **Your survey file
never leaves your machine.** If — and only if — you have added your own API key, the app sends the
*aggregate results* (group sizes, profiles, confidence numbers) to Claude so it can interpret them;
it never sends anyone's individual responses, and nothing is sent at all without a key.
