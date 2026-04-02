## prompt_id: coding_003 | experiment: decode_ema_score_averaging_bf390907 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: ema_score_averaging
decode_params: {"ema_beta": 0.9}

### Prompt

Design a simple REST API for a todo list application. Include the endpoints, request/response formats, and provide a basic Python implementation using Flask.

### Generated Text

Here is a design for a simple Todo List REST API, including the endpoint specifications and a complete Python implementation using Flask.

### 1. API Design Overview

The API will follow standard REST conventions using JSON for data exchange. We will use HTTP methods to perform CRUD (Create, Read, Update, Delete) operations on "Todo" resources.

**Base URL:** `http://localhost:5000/api`

#### Endpoints Specification

| Method | Endpoint | Description | Request Body (JSON) | Response (Success) |
| :--- | :--- | :--- | :--- | :--- |
| **GET** | `/todos` | Get all todos | None | `200 OK`: List of todo objects |
| **POST** | `/todos` | Create a new todo | `{ "title": "...", "completed": false }` | `201 Created`: The created todo object |
| **GET** | `/todos/<id>` | Get a specific todo by ID | None | `200 OK`: Single todo object<br>`404 Not Found` |
| **PUT** | `/todos/<id>` | Update an existing todo | `{ "title": "...", "
