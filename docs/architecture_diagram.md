# FLTS Pipeline Architecture

Below are visual representations (Mermaid + ASCII) of the end-to-end system: preprocessing -> multi-model training -> evaluation/promotion -> inference.

## Mermaid Overview
```mermaid
flowchart LR
    subgraph User/Data Sources
        U[User Trigger]
    end

    U --> P[Preprocess Service\n(config hash, Parquet, claim checks)]

    P -->|Kafka: training-data| TGRU[Trainer GRU]
    P -->|Kafka: training-data| TLSTM[Trainer LSTM]
    P -->|Kafka: training-data| TPRO[Trainer Prophet]

    subgraph MLflow & MinIO
        MLF[(MLflow Tracking + Artifacts)]
        S3[(MinIO Buckets\nmlflow, processed-data, model-promotion, inference-logs)]
    end

    TGRU -->|model-training events\n(run_id, SUCCESS)| EV[Evaluator]
    TLSTM -->|model-training events| EV
    TPRO -->|model-training events| EV

    TGRU --> MLF
    TLSTM --> MLF
    TPRO --> MLF
    P --> S3

    EV -->|Reads metrics & artifacts| MLF
    EV -->|Writes promotion history| S3
    EV -->|Kafka: model-selected| INF[Inference Service]

    INF -->|Download promoted model & scaler| MLF
    INF -->|Structured logs| S3
    INF -->|Predictions| U

    subgraph Topics
        TD[(training-data)]
        MT[(model-training)]
        MS[(model-selected)]
        ID[(inference-data)]
    end

    P --> TD
    TD --> TGRU
    TD --> TLSTM
    TD --> TPRO

    TGRU --> MT
    TLSTM --> MT
    TPRO --> MT
    MT --> EV

    EV --> MS
    MS --> INF

    U -->|ad-hoc inference| INF
    U -->|upload raw data| P

    INF <-->|optional fast path: model-training event| MT
```

## ASCII (High-Level)
```
            +-------------------+
            |   User / Client   |
            +---------+---------+
                      | upload / trigger
                      v
        +-------------+--------------+
        |        Preprocess          |
        |  config hash + Parquet     |
        +------+------+--------------+
               | claim checks (Kafka: training-data)
               v
   +-----------+-----------+--------------------+
   |       Multi-Model Trainers                 |
   |  GRU      LSTM        PROPHET              |
   +-----+-----+-----+-----+-----+--------------+
         |           |            |
         | model-training events  |
         +-----------+------------+
                     v
                +----+------------------+
                |      Evaluator        |
                | waits all model types |
                +----+------------------+
                     | promotion pointer + history
                     v
        +-------------------------------+
        |        MinIO (model-promotion)|
        +-------------------------------+
                     |
           model-selected (Kafka)
                     v
            +--------+---------+
            |      Inference   |
            | 2 workers + queue|
            +--------+---------+
                     | predictions
                     v
            +-------------------+
            |   User / Client   |
            +-------------------+
```

## Concurrency & Caching Notes
- Inference queue: QUEUE_WORKERS=2, bounded by QUEUE_MAXSIZE=40.
- Duplicate prediction suppression: (run_id, payload hash) set.
- Promoted model pointer cached (`current.json`), refreshed on model-selected events.
- Scaler discovery searches `scaler/` artifact path then root of run.

## Promotion Logic
1. Wait until all EXPECTED_MODEL_TYPES SUCCESS for same config_hash.
2. Compute score: 0.5*rmse + 0.3*mae + 0.2*mse (lower wins).
3. Tie-break: latest start_time.
4. Write promotion history + update pointer + emit model-selected.

## Buckets
- processed-data: preprocessed parquet + metadata.
- mlflow: MLflow artifacts (model folders, scaler/, metrics).
- model-promotion: promotion history and current.json pointer.
- inference-logs: structured JSON inference logs.
- inference-txt-logs (legacy/compat retained for now).

## Data/Event Lineage (Claim-Check)
Raw upload -> processed parquet (MinIO) -> small Kafka JSON referencing object -> trainers fetch via gateway -> metrics/artifacts -> evaluator selects -> inference loads promoted model.

---
If you’d like a sequence diagram or a focused one just for promotion or inference, let me know.
