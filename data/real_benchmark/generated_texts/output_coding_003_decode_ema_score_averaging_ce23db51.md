## prompt_id: coding_003 | experiment: decode_ema_score_averaging_ce23db51 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: ema_score_averaging
decode_params: {"ema_beta": 0.5}

### Prompt

Design a simple REST API for a todo list application. Include the endpoints, request/response formats, and provide a basic Python implementation using Flask.

### Generated Text

Here is a design for a simple Todo List REST API, including the endpoint specifications and a complete Python implementation using Flask.

### 1. API Design Specification

The API follows RESTful conventions. Uses standard HTTP methods (GET, POST, PUT, DELETE) to interact with resources (`todos`).. Data is exchanged in JSON format.

#### Base URL
`http://localhost:5000/api`

#### Endpoints

| Method | Endpoint | Description | Request Body (JSON) | Response (200 OK) |
| :--- | :--- | :--- | :--- | :--- |
| **GET** | `/api/todos` | Get all todos | N/A | `[{ "id": 1, "title": "...", "completed": false }]` |
| **POST** | `/api/todos` | Create a new todo | `{ "title": "Buy milk" }` | `{ "id": 3, "title": "Buy milk", "completed": false }` |
| **GET** | `/api/todos/<id>` | Get specific todo by ID | N/A | `{ "id": 1, "title": "...", "completed": false }`
