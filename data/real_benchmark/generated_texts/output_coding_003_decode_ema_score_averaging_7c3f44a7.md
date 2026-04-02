## prompt_id: coding_003 | experiment: decode_ema_score_averaging_7c3f44a7 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: ema_score_averaging
decode_params: {"ema_beta": 0.7}

### Prompt

Design a simple REST API for a todo list application. Include the endpoints, request/response formats, and provide a basic Python implementation using Flask.

### Generated Text

Here is a design for a simple Todo List REST API, including the endpoint specifications and a complete Python implementation using Flask.

### 1. API Design Overview

The API will support standard CRUD operations (Create, Read, Update, Delete) for todo items. Each item will have an `id`, `title`, `description`, and a `completed` status.

**Base URL:** `/api/todos`

#### Endpoints Specification

| Method | Endpoint | Description | Request Body (JSON) | Response (Success 200/201) |
| :--- | :--- | :--- | :--- | :--- |
| **GET** | `/api/todos` | Get all todos | None | `[{ "id": 1, "title": "...", ... }]` |
| **POST** | `/api/todos` | Create a new todo | `{ "title": "Buy milk", "description": "Low fat" }` | `{ "id": 3, "title": "Buy milk", ... }` |
| **GET** | `/api/todos/<id>` | Get a specific todo | None | `{ "id": 1, "title": "...", ... }
