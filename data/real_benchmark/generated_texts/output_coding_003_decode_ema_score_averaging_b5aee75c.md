## prompt_id: coding_003 | experiment: decode_ema_score_averaging_b5aee75c | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: ema_score_averaging
decode_params: {"ema_beta": 0.3}

### Prompt

Design a simple REST API for a todo list application. Include the endpoints, request/response formats, and provide a basic Python implementation using Flask.

### Generated Text

Here is a design of a simple REST API for a Todo List application, followed with a basic Python implementation using Flask.

### 1. API Design Specification



**Base URL:** ``http://localhost:5000/api``



**Data Format:** All requests and responses will use JSON (``application/json``).



| Method | Endpoint | Description | Request Body (JSON) | Response Body (JSON) |
| :--- | :--- | :--- | :--- | :--- |
| **GET** | `/todos` | Get all todos | None | List of todo objects |
| **POST** | `/todos` | Create a new todo | `{`"title"`: "string", `"completed"`: boolean}` | Created todo object |
| **GET** | `/todos/<id>` | Get specific todo by ID | None | Single todo object or 404 if not found |
| **PUT** | `/todos/<id>` | Update existing todo | `{`"title"`: "string", `"completed"`: boolean}` | Updated todo object or 404 if not found |
| **DELETE** | `/todos/<id>` | Delete a todo by
