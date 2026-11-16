# How the evaluation was kept honest

> **In plain words.** A score is only worth something if the test could not have been bent to fit the answer. The
> project used seven habits for this: write the pass mark first; test on a list nobody has looked at; draw fresh
> random layouts blindly; run each counted list once, with no re-rolls; freeze the interface the test uses; log
> every change of rule in public; and have the results checked by someone who did not produce them.

## 7.1 Gates are written before the run

Every phase had a gate, a numeric pass mark, written in the brief or in the task text before the run started. Examples:
"the robot reaches every place 10 out of 10 times"; "at least 7 of 10 runs pass"; "at least 18 of 25 per kind".
A result that misses a gate is reported as a miss. The rule written into the project notes was that thresholds never
move to make a gate pass, and that any change is a dated entry in the deviation log (7.6) with the authority named.

Before the first run, the project also wrote down the stop rules (section 6.4). This matters because the awkward
decisions, such as "should we try one more time", were already made in advance.

## 7.2 A sealed test list

The final test lists were produced by a generator and then sealed. "Sealed" here means an archive whose hash was
committed to the repository before the test phase, kept outside the working tree and not opened by anyone until
phase 8. Dev prompts and dev seeds, the ones used while building, are separate from the test ones, and test material
is never used for development.

There were three sealed sets. Set 1 was used for round 1, set 2 for round 2, and set 3 was held in reserve for
a third round that was never needed. It was never extracted, and the report records that.

## 7.3 Blind fresh seeds

The sealed generator was written before the world got its final shape, so some of its layouts did not fit the world
(a recorded deviation). The fix had to avoid hand picking. So layouts were *realised* on the final world by a program
and the supply continued with fresh random seeds whose hash was committed before any layout was realised. A fixed,
mechanical validity filter (a recorded deviation, written before unsealing) dropped entries that were known to be unfair,
for example a toolbox answer in the office, where the VLM cannot name it. Replacements were the next entries of the same
stream, in order, and the counts of drops were reported. Nobody read entries by hand. The start place cycles through hall,
another room and the answer room, with at most two answer-room starts per kind, and the seed of each planner call is
derived from the task id. Each list was committed before it was run.

## 7.4 Counted-run rules

A counted list is the one that produces the reported numbers. The rules for it were fixed before any VLM output
existed:

- a counted list runs once, under one invocation id, and a ledger entry is written outside the repository;
- if the runner crashes, the partial run is scored as it stands; a resume may run only entries that have no run
  directory yet; nothing is rerun or deleted; a rehearsal flag, a single-entry flag or an existing output directory
  are refused;
- a failure to reach the VLM server is scored as a failure (label `sim`); if a gate fails only because of those, one
  fresh re-count per phase is allowed without using a fix round;
- before a counted start the GPU job must have at least twice the expected run time left;
- a dress rehearsal on the exact counted commit must be clean first: zero replies cut off at the token limit and zero
  tool calls hidden inside the reasoning;
- the runner stops on a dirty working tree or a moved commit;
- every entry's run id, job, node and sim-to-wall speed are recorded.

When a counted list was spread over several cluster jobs, the split was written down before the first run, each job wrote its
own ledger, a single merge step combined them, and the gates refused an unmerged set. A test with a fake robot checks that a
duplicated or a missing entry fails the gate.

## 7.5 The interface freeze

At the end of phase 7 the interface was frozen (section 4.8) and tagged in git. Every counted run in phases 8 and 10 records
the commit it ran on, and a test confirms that the interface file at that commit equals the tag. After the one allowed fix,
a second freeze (version 2) replaced the first. The humanoid test then required a swap check against version 2: a
`git diff` of the agent, the skills, the prompts, the world generators, the interface files and the serving files, against
the tag, must show **0 lines**. It did.

## 7.6 The deviation log

Every time a rule, a parameter or a gate was changed during the build, a row was added to the deviation log (appendix C):
what changed, from what to what, why, and whether the meaning of a gate changed. There are 22 rows. The column "did the gate
semantics change" is the one that matters: for most rows the answer is no (a camera, a tolerance, a map). Two rows are marked
yes: a parser that was too strict was changed after seeing data, with the original failure left in the record as a fail (a recorded deviation),
and a failing precondition was allowed to proceed by the project lead's decision (a recorded deviation). Deviation 15 is also flagged in
the log itself: the ruling that excluded a failed cluster run was made after its result was known, on bug evidence, and the
project lead confirmed it.

## 7.7 Controls, mutants and audits

Some checks are there to prove that other checks work.

- **Negative controls.** A gate that cannot fail is worthless. So the camera crop check must fail on uncropped frames; the
  scorer must fail seven deliberately wrong scripts; a scripted walk into a chair must register a collision; and the humanoid
  self-view control must fail on pitched frames.
- **Mutation tests.** The gate code and the interface freeze have tests that plant bugs and demand that the tests notice:
  17 of 17† planted bugs in the verdict code of the first world gates, 3 of 3† in the freeze.
- **Truth separation.** A test enforces that the agent and the adapters never read the truth file, and a dynamic test checks that
  importing the agent loads neither the truth module nor the scorer.
- **Raw recounts.** Headline results were recomputed from the raw run directories, not from summary tables. The humanoid's score files
  were recounted from all 100 saved run directories.
- **Independent audits.** The task reports were spot-checked by a separate auditor against the saved evidence and git. A number script
  (`showcase/check_numbers.py`) then compares every figure in the showcase documents, including this one, with the audited reports,
  and every number carries a flag for whether its report was audited. That is what the dagger (†) means.
- **Looking at the artefact.** Every visual gate (contact sheets, video, slides, and this PDF) was opened and looked at before it was
  marked passed. Five problems in the video were found only that way.

None of these makes the numbers true. They make it much harder to produce a good-looking number by accident.
