# Reference voice clips for the model comparison

Two 7–8 second studio clips used as cloning references in `benchmark/`:

| File | Source | Licence |
|---|---|---|
| `natasha.wav` | `intexcp/natasha_sova_ai` (Hugging Face), validation[41] — Sova "Natasha" corpus | CC BY 4.0 |
| `ruslan.wav` | `intexcp/ruslan_sova_ai` (Hugging Face), validation[50] — RUSLAN corpus | CC BY-NC-SA 4.0 |

`refs.json` holds the transcripts (plain and with `+` stress marks).
`stress_manual.json` and `stress_silero.json` are stress-marked versions of the
test phrases. The service itself does not use these files.
