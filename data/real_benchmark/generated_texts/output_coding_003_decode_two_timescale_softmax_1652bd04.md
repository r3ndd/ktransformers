## prompt_id: coding_003 | experiment: decode_two_timescale_softmax_1652bd04 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: two_timescale_softmax
decode_params: {"mix_lambda": 0.2, "rho": 1.0}

### Prompt

Design a simple REST API for a todo list application. Include the endpoints, request/response formats, and provide a basic Python implementation using Flask.

### Generated Text

Here is a design for a simple REST API for a Todo List application, followed with a basic Python implementation using Flask.

### 1. API Design Specification

**Application Name:** `TodoList API`
**Base URL:** `http://localhost:8080/api/todos`

#### Endpoints Overview

| Method | Endpoint | Description |
| --- | --- | --- |
| **GET** | `/todos` | Retrieve all existing todos |
| **POST** | `/todos` | Create a new todo |
| **PUT** | `/todos/{id}` | Update an existing todo |
| **DELETE** | `/todos/{id}` | Delete an existing todo |

#### Request/Response Formats (JSON)

**Data Model:**
```json
{
  "id": integer,
  "title": string,
  "completed": boolean
}
```

#### Endpoint Details

**1. GET `/todos`**
*   **Description:** Returns a list of all current tasks.
*   **Request:** No parameters.
*   **Response:**
    ```json
    {
      "status": "success",
      "data": [
        {"id":
