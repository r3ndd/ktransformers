## prompt_id: coding_003 | experiment: decode_two_timescale_softmax_f375d90f | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: two_timescale_softmax
decode_params: {"mix_lambda": 0.2, "rho": 16.0}

### Prompt

Design a simple REST API for a todo list application. Include the endpoints, request/response formats, and provide a basic Python implementation using Flask.

### Generated Text

Here is a design for a simple REST API for Todo List application, followed by a basic implementation using Flask.

### **API Design**

#### **Endpoints**
1.  **GET `/todos`**: Retrieve all existing todos.
2.  **POST `/todos`**: Create a new todo.
3.  **GET `/todos/{id}`**: Retrieve a specific todo by ID.
4.  **PUT `/todos/{id}`**: Update a specific todo by ID.
5.  **DELETE `/todos/{id}`**: Delete a specific todo by ID.

#### **Request/Response Formats**
-   **Request Body (JSON)**: `{"title": "string", "completed": boolean}`
-   **Response Body (JSON)**: `{"id": integer, "title": "string", "completed": boolean}`

---

### **Python Implementation (Flapp)**

```python
from flask import Flask, jsonify, request, post, get, put, delete
from flask import json

app = Flask()

todos = []

def create_todo():
    """
    POST /todos
    """
    data = request.get_json()
    if not data or 'title' not
