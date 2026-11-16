# The skill API

> **In plain words.** These are the exact definitions the VLM is shown and the runtime enforces: three tools, their arguments,
> their result texts and the rules for what counts as a wrong call. Below them, the seven methods a robot adapter must provide.

## B.1 The tools

All arguments are required, and extra fields are rejected.

| Tool | Arguments | Returns |
|---|---|---|
| `navigate_to` | `place`: one of hall, ward, office, reception, storage | `reached` or `failed: blocked`, plus one photo (number 1) |
| `look_around` | none | `4 photos`, photos numbered 1 to 4 |
| `report` | `answer`, `photo` (an integer of at least 1), `note` (text) | `reported`; ends the run |

The `answer` format depends on the task kind:

| Task kind | `answer` accepts |
|---|---|
| find, conditional | one of the room names (the enum of rooms) |
| check state | `yes` or `no` |
| count | digits only, matching `^(0\|[1-9][0-9]*)$` |

Descriptions the VLM sees: `navigate_to`: "Go to a named place; returns a new photo." `look_around`: "Turn on the spot and take four numbered photos."
`report`: "Give the final answer and end the run. photo = number of the photo (in the current message) that shows the evidence. note = free text,
never scored."

## B.2 Result texts and error texts

| Situation | Text that goes back to the VLM |
|---|---|
| navigation succeeded and within 0.30 m of the place | `reached` |
| anything else for navigation | `failed: blocked` |
| turn-and-photo set done | `4 photos` |
| report accepted | `reported` |
| zero or several tool calls | `error: call exactly one tool per step (got N)` |
| unknown skill | `error: unknown skill 'x'; available: navigate_to, look_around, report` |
| arguments are not valid JSON | `error: arguments of <skill> are not valid JSON` |
| a schema violation, for example an unknown place or a bad answer format | `error: invalid <skill> call at <field>: <reason>` |
| a photo number that is not in the current message | `error: photo N does not exist; the current message shows photos 1..K` |

## B.3 Runtime rules

- At most 12 steps. Running out is a failure.
- Two malformed outputs in a row end the run as a failure. A network failure to the model server is an error step but not a malformed output.
- Errors are results, not crashes. Any exception inside a skill or the adapter ends the run as a crash, logged with its origin.
- `navigate_to` has a time limit of 120 simulated seconds and counts as reached only within 0.30 m of the place.
- `look_around` turns to absolute headings: the place's yaw plus 0, 90, 180 and 270 degrees when the robot stands within 0.30 m of a place.

## B.4 The robot adapter

```python
class Adapter(ABC):
    def go_to(self, x: float, y: float, yaw: float | None, timeout_s: float) -> GoResult   # waits for the goal
    def turn_to(self, yaw: float) -> float        # absolute heading; returns the remaining error
    def photo(self) -> Frame                      # a frame stamped after the robot settled, already cropped
    def pose(self) -> tuple[float, float, float]  # the robot's own estimate: x, y, yaw
    def where(self) -> str                        # the room that estimate falls in
    def stop(self) -> None                        # zero velocity, cancel any goal
    def sim_time(self) -> float                   # the simulator clock

@dataclass(frozen=True)
class GoResult:
    status: str        # "reached" or "blocked"; aborted, cancelled, rejected goals and timeouts all map to "blocked"
    nav_status: str    # the raw navigation status name, or "TIMEOUT"
    sim_s: float       # simulated seconds
    wall_s: float      # wall-clock seconds
```

A frame holds the cropped picture, a matching label image for the scorer (never given to the VLM), and metadata such as the simulator time and the
crop that was applied. A separate class used only on the evaluator side reads the exact pose from the simulator. The planner cannot reach it.
