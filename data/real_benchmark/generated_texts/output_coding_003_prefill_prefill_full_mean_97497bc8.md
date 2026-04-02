## prompt_id: coding_003 | experiment: prefill_prefill_full_mean_97497bc8 | seed: 42

### MoE Routing

prefill: prefill_full_mean
prefill_params: {}
decode: sliding_window_score_averaging
decode_params: {"window_size": 1}

### Prompt

Design a simple REST API for a todo list application. Include the endpoints, request/response formats, and provide a basic Python implementation using Flask.

### Generated Text

assistant
<think>

</think>

Here is a design for a simple Todo List REST API, followed by a complete, runnable Python implementation using Flask.

### 1. API Design Specification

We will use standard HTTP methods and JSON for data exchange. The resource is `todos`.

#### Endpoints

| Method | Endpoint | Description | Request Body (JSON) | Response (200 OK) |
| :--- | :--- | :--- | :--- | :--- |
| **GET** | `/api/todos` | Get all todos | None | `[{ "id": 1, "title": "...", "completed": false }]` |
| **POST** | `/api/todos` | Create a new todo | `{ "title": "Buy milk" }` | `{ "id": 2, "title": "Buy milk", "completed": false }` |
| **GET** | `/api/todos/<id>` | Get a specific todo | None | `{ "id": 2, "title": "Buy milk", "completed": false }` |
| **PUT** | `/api/todos/<id>` | Update a todo (replace) | `{ "title": "Buy
