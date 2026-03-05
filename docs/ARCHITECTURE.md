# SciOracle Developer Architecture

This diagram shows the runtime dataflow between agents, proof/energy services, database storage, and OpenClaw integration.

```mermaid
flowchart TD
    U[User / Client] --> API[FastAPI Backend\nbackend/app.py]
    API --> EVAL[EBM Evaluation\nevaluate_energy]
    API --> DB[(SQLite\nmath_knowledge.db)]

    subgraph CORE[Agent Runtime\ncore_agent.py]
      OC[Oracle_Coder]
      SV[Symbolic_Validator]
      ES[EBM_Solver]
      SM[state.json\nSciOracleStateManager]
      OC --> SM
      SV --> SM
      ES --> SM
    end

    SV --> PROOF[Formal Proof Layer\nskills/symbolic_log.py\nSymPy + Z3 subset]
    ES --> EBM[Energy Layer\nebm_math_discovery.py]
    EBM --> DECL[Declaration Gate\ncalibration + repeatability]
    PROOF --> DECL
    DECL --> DB

    DB --> G1[/api/research/graph]
    DB --> G2[/api/research/lineage]

    subgraph BRIDGE[OpenClaw Bridge\nopenclaw_interface.py]
      HB[Heartbeat + State Push]
      CMD[Command Pull\nset_conjecture / set_status / state_patch\nrun_ablation / request_counterexample\npromote_candidate / demote_candidate]
    end

    SM --> HB
    CMD --> SM
```

## Key files

- `core_agent.py`: process orchestration and loop timing.
- `skills/symbolic_log.py`: formal proof checks and counterexample traces.
- `skills/ebm_solve.py`: energy solve + discovery updates.
- `ebm_math_discovery.py`: model, training, declaration logic, DB schema helpers.
- `backend/app.py`: API layer and research endpoints.
- `openclaw_interface.py`: OpenClaw/OpenDeepClaw command and sync bridge.
