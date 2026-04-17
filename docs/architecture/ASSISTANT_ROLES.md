# Assistant Roles

## Purpose
This document defines the initial assistant roles for the `neural_kernel` software factory direction.

The goal is to structure the software delivery pipeline around specialized assistants rather than one undifferentiated general assistant.

## Design Principle
Each assistant should have:
- a clear responsibility
- a well-defined input
- a well-defined output
- limited scope
- predictable handoff to the next stage

The system should prefer role clarity over agent complexity.

---

## 1. Intake Assistant

### Role
Convert a raw customer request into a structured project brief.

### Responsibilities
- identify the customer goal
- extract required functionality
- extract constraints
- identify unclear requirements
- identify expected users and usage context
- normalize the request into a reusable format

### Input
- customer prompt
- follow-up clarifications
- any provided business context

### Output
- structured project brief

### Typical Output Fields
- project summary
- target users
- main features
- input/output expectations
- integrations
- constraints
- assumptions
- open questions

### Success Condition
The request becomes clear enough for architectural planning.

---

## 2. Solution Architect Assistant

### Role
Translate the structured brief into a technical design and execution plan.

### Responsibilities
- define system components
- choose implementation approach
- identify major modules
- outline data flow
- define external dependencies
- identify risks and unknowns
- propose execution phases

### Input
- structured project brief

### Output
- technical specification
- architecture outline
- implementation plan

### Typical Output Fields
- architecture summary
- frontend/backend needs
- database needs
- API boundaries
- integration points
- deployment assumptions
- risk notes
- phased delivery plan

### Success Condition
The brief becomes technically actionable.

---

## 3. Builder Assistant

### Role
Generate initial implementation artifacts from the technical specification.

### Responsibilities
- generate project skeleton
- create starter module layout
- draft API contracts
- draft data models
- draft core implementation stubs
- generate setup and README scaffolding

### Input
- technical specification
- architecture outline

### Output
- project skeleton
- implementation stubs
- developer-facing starter artifacts

### Typical Output Fields
- file/folder structure
- module list
- endpoint skeletons
- schema drafts
- starter components
- setup instructions

### Success Condition
A project foundation exists and can be reviewed or extended.

---

## 4. QA / Review Assistant

### Role
Evaluate generated artifacts for quality, consistency, and missing pieces.

### Responsibilities
- identify architectural inconsistencies
- identify missing files or flows
- identify test gaps
- identify edge cases
- identify basic security and reliability concerns
- propose corrective actions

### Input
- generated project artifacts
- technical specification

### Output
- review report
- issue list
- improvement recommendations

### Typical Output Fields
- missing components
- test recommendations
- reliability notes
- code organization issues
- risk flags
- patch suggestions

### Success Condition
The generated project becomes safer and more complete before delivery.

---

## 5. Delivery Assistant

### Role
Package reviewed outputs into a client-ready delivery bundle.

### Responsibilities
- organize final artifacts
- prepare README / handoff notes
- summarize what was built
- explain how to run or deploy
- explain known limitations
- prepare delivery summary

### Input
- reviewed project artifacts
- QA findings
- final approved scope

### Output
- delivery package
- handoff summary
- usage notes

### Typical Output Fields
- project overview
- run instructions
- deployment notes
- limitations
- next-step recommendations

### Success Condition
The project is understandable and transferable to a user or customer.

---

## Human Oversight Role

### Role
Provide decision authority, final review, and accountability.

### Responsibilities
- approve scope
- review assistant outputs
- resolve ambiguity
- enforce quality threshold
- decide what gets delivered

### Why It Matters
The first version of the software factory should not aim for full autonomy.

Human oversight is essential for:
- quality
- trust
- prioritization
- real-world accountability

---

## Role Handoff Sequence

The intended initial pipeline is:

1. Intake Assistant
2. Solution Architect Assistant
3. Builder Assistant
4. QA / Review Assistant
5. Delivery Assistant

This order should remain fixed in the first version to keep orchestration simple and understandable.

---

## Initial Operating Rules

- each role should produce structured outputs
- outputs should be reusable by downstream roles
- avoid vague conversational handoffs
- keep artifact formats simple and auditable
- optimize for repeatability over autonomy
- keep humans in the approval loop

---

## First Demo Role Set
The first orchestrated demo should use exactly these roles:
- Intake Assistant
- Solution Architect Assistant
- Builder Assistant
- QA / Review Assistant
- Delivery Assistant

This is enough to demonstrate a convincing software factory flow without unnecessary complexity.

---

## Summary
The assistant system should start as a role-based pipeline with clean handoffs, limited scope per assistant, and strong human oversight.