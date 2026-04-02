## prompt_id: coding_003 | experiment: decode_two_timescale_softmax_cc6c43c8 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: two_timescale_softmax
decode_params: {"mix_lambda": 0.2, "rho": 4.0}

### Prompt

Design a simple REST API for a todo list application. Include the endpoints, request/response formats, and provide a basic Python implementation using Flask.

### Generated Text

Here is a design for a simple REST API for a Todo List application, followed by a basic implementation using Flask.

### API Design Overview

The API will support CRUD operations (Create, Read, Update, Delete) for todo items. It uses JSON for data exchange and standard HTTP status codes.

#### Endpoints & Specifications

| Method | Endpoint | Description | Request Body | Response Body |
|---|---|---|---|---|
| **GET** | `/todos` | Retrieve all todos | None | `List of objects` |
| **POST** | `/todos` | Create a new todo | `{"title": "string", "completed": boolean}` | `Created object` |
| **GET** | `/todos/{id}` | Get specific todo by ID | None | `Object` or `4404` if not found |
| **PUT** | `/todos/{id}` | Update existing todo | `{"title": "string", "completed": boolean}` | `Updated object` |
| **DELETE** | `/todos/{id}` | Delete specific todo | None | `200` with message or `4404` if not found |

*Note: The
