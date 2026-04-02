## prompt_id: coding_003 | experiment: decode_sliding_window_score_averaging_d4dee748 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: sliding_window_score_averaging
decode_params: {"window_size": 4}

### Prompt

Design a simple REST API for a todo list application. Include the endpoints, request/response formats, and provide a basic Python implementation using Flask.

### Generated Text

Here is a design for a simple REST API for a Todo List application, followed with a a basic Python implementation using Flask.

### 1. API Design Specification



**Overview:**
The API provides to manage CRUD operations (Create, Read, Update, Delete) on Todo items. Each item has an unique ID, title, description, and completion status.



**Base URL:** ``http://localhost:5000/api/todos`



| Method | Endpoint | Description | Request Body (JSON) | Response Body (JSON) | Status Code |
|---|---|---|---|---|---|
| GET | `/api/todos` | Get all todos | None | Array of todo objects | 200 OK |
| POST | `/api/todos` | Create a new todo | `{title: string, description: string}` | Object of created todo | 201 Created |
| GET | `/api/todos/{<id>}` | Get specific todo by id | None | Object of the todo | 200 OK /404 Not Found |
| PUT | `/api/todos/{<id>` | Update existing todo | `{title: string,
