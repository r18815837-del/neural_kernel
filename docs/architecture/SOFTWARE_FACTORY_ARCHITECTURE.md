# Software Factory Architecture

## Purpose
This document defines the first architecture for the `neural_kernel` AI software factory direction.

The goal is to move from a framework-centric project toward a system that can help transform client requirements into structured software project outputs.

## High-Level Goal
Build an AI-assisted software delivery pipeline that takes a user request and produces:
- a structured brief
- a technical specification
- a project skeleton
- a review pass
- a delivery package

## Core Idea
The system is not intended to be a fully autonomous replacement for engineering work.

Instead, it acts as a structured multi-assistant pipeline with human oversight, where specialized assistants handle different stages of project definition, generation, review, and delivery.

## System Layers

### 1. Foundation Layer
This is the technology base.

Includes:
- `neural_kernel`
- model/runtime experimentation layer
- reusable AI tooling primitives
- benchmarking and validation foundation

### 2. Orchestration Layer
This layer coordinates multi-step execution.

Responsibilities:
- route requests through assistant stages
- pass artifacts between stages
- preserve shared context
- maintain execution order
- define workflow boundaries

### 3. Assistant Layer
This layer contains specialized role-based assistants.

Initial assistant roles:
- Intake Assistant
- Solution Architect Assistant
- Builder Assistant
- QA / Review Assistant
- Delivery Assistant

### 4. Artifact Layer
This layer contains outputs produced by the pipeline.

Examples:
- requirement brief
- technical specification
- code skeleton
- API contract
- database schema draft
- test plan
- README / handoff notes

### 5. Human Oversight Layer
This layer ensures quality and accountability.

Responsibilities:
- approve scope
- validate architecture decisions
- review generated outputs
- decide what is delivered to the client

## Pipeline Overview

### Stage 1 — Intake
Input:
- customer request
- problem statement
- goals
- constraints

Output:
- structured project brief

### Stage 2 — Solution Design
Input:
- structured project brief

Output:
- technical specification
- architecture outline
- delivery plan

### Stage 3 — Build
Input:
- technical specification

Output:
- project skeleton
- module breakdown
- starter implementation artifacts

### Stage 4 — Review
Input:
- build artifacts

Output:
- quality review
- bug/edge-case notes
- improvement recommendations

### Stage 5 — Delivery
Input:
- reviewed project artifacts

Output:
- final package
- usage notes
- deployment or handoff guidance

## Initial Constraints
- keep the first version human-in-the-loop
- optimize for clarity and repeatability, not full autonomy
- focus on one clear demo scenario first
- prefer structured artifacts over vague conversational output

## Demo Direction
The first orchestrated demo should focus on one narrow and understandable project type.

Recommended direction:
- internal knowledge assistant
or
- simple business support chatbot
or
- document-based internal tool

## Near-Term Outcome
The near-term outcome is not a production AGI platform.

The near-term outcome is a convincing architecture and demo flow showing how specialized assistants can participate in structured software delivery.

## Summary
The software factory should be built as a layered, role-based, human-supervised system that turns client intent into progressively refined software artifacts.