## prompt_id: coding_003 | experiment: prefill_prefill_block_mean_81581d58 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: sliding_window_score_averaging
decode_params: {"window_size": 1}

### Prompt

Design a simple REST API for a todo list application. Include the endpoints, request/response formats, and provide a basic Python implementation using Flask.

### Generated Text

Here is a design for a simple Todo List REST API, followed by a complete Python implementation using Flask.

### 1. API Design Overview

The API will support standard CRUD (Create, Read, Update, Delete) operations for managing tasks. We will use JSON for data exchange and HTTP status codes to indicate success or failure.

**Base URL:** `http://localhost:5000/api`

#### Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **GET** | `/todos` | Retrieve all todos. Supports optional query param `?completed=true/false`. |
| **POST** | `/todos` | Create a new todo. |
| **GET** | `/todos/<id>` | Retrieve a specific todo by ID. |
| **PUT** | `/todos/<id>` | Update an existing todo (replaces the whole object). |
| **PATCH** | `/todos/<id>` | Partially update an existing todo (e.g., just toggle completion). |
| **DELETE** | `/todos/<id>` | Delete a specific todo. |

#### Request/Response Formats

**1. Create Todo (`POST /todos`)**
*
