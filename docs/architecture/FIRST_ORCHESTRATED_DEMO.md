# First Orchestrated Demo

## Purpose
This document defines the first orchestrated demo for the `neural_kernel` software factory direction.

The goal is to show a clear, believable end-to-end pipeline where specialized assistants transform a client request into structured software project artifacts.

## Demo Goal
Demonstrate how a multi-assistant pipeline can take a business request and produce:

- a structured project brief
- a technical specification
- a project skeleton
- a review report
- a delivery package summary

The first demo is intentionally narrow and human-supervised.

## Selected Demo Scenario
**Internal knowledge assistant for a company**

This scenario is chosen because it is:
- easy to explain
- valuable to businesses
- realistic for a first AI delivery demo
- narrow enough for a structured first orchestration flow

## Example Customer Request
A company says:

> We need an internal assistant that can answer employee questions based on company documents, policies, and process notes. It should have a chat interface, document search, and basic admin controls.

## Demo Pipeline

### Stage 1 — Intake Assistant

#### Input
Raw customer request.

#### Responsibilities
- extract business goal
- identify users
- identify required features
- identify missing details
- structure the request

#### Output
Structured project brief.

#### Example Output
- project type: internal knowledge assistant
- users: employees, internal admins
- core capabilities:
  - document ingestion
  - retrieval/search
  - chat over company knowledge
  - admin controls
- constraints:
  - internal use
  - simple MVP scope
- assumptions:
  - documents are uploaded manually in initial version
  - admin panel can be minimal
- open questions:
  - auth method
  - file size/volume expectations
  - deployment target

---

### Stage 2 — Solution Architect Assistant

#### Input
Structured project brief.

#### Responsibilities
- define architecture
- define main modules
- define delivery phases
- identify risks

#### Output
Technical specification and architecture outline.

#### Example Output
- frontend:
  - simple internal chat UI
  - admin upload page
- backend:
  - document ingestion service
  - retrieval service
  - chat/query service
  - auth layer
- storage:
  - document metadata store
  - vector/retrieval index
- APIs:
  - upload documents
  - query assistant
  - admin document list
- risks:
  - retrieval quality
  - document parsing consistency
  - permissions/auth scope

---

### Stage 3 — Builder Assistant

#### Input
Technical specification.

#### Responsibilities
- generate starter project layout
- create implementation skeleton
- create API stubs
- create data model draft
- create setup scaffolding

#### Output
Project skeleton and starter artifacts.

#### Example Output
- folder structure:
  - `frontend/`
  - `backend/`
  - `docs/`
  - `tests/`
- starter backend modules:
  - `auth.py`
  - `documents.py`
  - `retrieval.py`
  - `chat.py`
- starter frontend pages:
  - chat page
  - admin upload page
- starter README
- `.env.example`
- API route draft

---

### Stage 4 — QA / Review Assistant

#### Input
Generated project skeleton and technical specification.

#### Responsibilities
- identify missing pieces
- identify edge cases
- identify testing gaps
- identify reliability concerns

#### Output
Review report.

#### Example Output
- missing pagination for document list
- unclear error handling for failed uploads
- no document deletion flow yet
- no test skeleton for retrieval logic
- auth assumptions not fully resolved
- recommendation:
  - add upload validation
  - add test plan for retrieval and permissions
  - add admin audit notes

---

### Stage 5 — Delivery Assistant

#### Input
Reviewed artifacts and QA notes.

#### Responsibilities
- package output
- summarize what was created
- describe how to run
- describe known limitations
- prepare handoff notes

#### Output
Delivery summary.

#### Example Output
- project overview
- setup steps
- run instructions
- current MVP scope
- known limitations
- recommended next implementation steps

---

## Final Demo Artifacts
The first orchestrated demo should visibly produce:

1. `project_brief.md`
2. `technical_spec.md`
3. `project_skeleton.md`
4. `review_report.md`
5. `delivery_summary.md`

These can initially be generated as structured markdown artifacts.

## Why This Demo Works
This demo is strong because it shows:
- assistant specialization
- structured handoffs
- artifact-based progress
- realistic business use
- a believable path from request to implementation plan

It avoids the trap of claiming full autonomous software creation.

## Constraints for the First Version
- no claim of full autonomy
- no claim of production-ready end-to-end code generation
- no attempt to support every project type
- keep scope narrow and repeatable
- keep human review in the loop

## Success Criteria
The first orchestrated demo is successful if it demonstrates:

- clear assistant role separation
- meaningful artifact handoff between stages
- a believable business-facing use case
- a repeatable and understandable pipeline
- a clear connection between framework direction and startup direction

## Future Extensions
Later versions can expand toward:
- richer project types
- code artifact generation
- deployment artifact generation
- tool integrations
- workflow memory/state tracking

## Summary
The first orchestrated demo should show a narrow, realistic, role-based AI software delivery flow for an internal knowledge assistant, with structured artifacts produced at every stage.