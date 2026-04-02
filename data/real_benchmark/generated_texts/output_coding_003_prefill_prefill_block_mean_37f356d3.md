## prompt_id: coding_003 | experiment: prefill_prefill_block_mean_37f356d3 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 64}
decode: sliding_window_score_averaging
decode_params: {"window_size": 1}

### Prompt

Design a simple REST API for a todo list application. Include the endpoints, request/response formats, and provide a basic Python implementation using Flask.

### Generated Text

assistant
<think>

</think>

Here is a complete design and implementation for a simple Todo List REST API using Python and Flask.

### 1. API Design Overview

We will use standard HTTP methods to perform CRUD (Create, Read, Update, Delete) operations on a list of tasks.

**Base URL:** `/api`

| Method | Endpoint | Description | Request Body (JSON) | Response (200 OK) |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/todos` | Get all todos | None | `[{ "id": 1, "title": "...", "completed": false }]` |
| `GET` | `/todos/<id>` | Get a specific todo | None | `{ "id": 1, "title": "...", "completed": false }` |
| `POST` | `/todos` | Create a new todo | `{ "title": "Buy milk" }` | `{ "id": 2, "title": "Buy milk", "completed": false }` |
| `PUT` | `/todos/<id>` | Update a todo | `{ "title": "Buy eggs", "completed": true
