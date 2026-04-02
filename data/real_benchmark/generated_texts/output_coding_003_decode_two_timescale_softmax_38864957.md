## prompt_id: coding_003 | experiment: decode_two_timescale_softmax_38864957 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: two_timescale_softmax
decode_params: {"mix_lambda": 0.2, "rho": 64.0}

### Prompt

Design a simple REST API for a todo list application. Include the endpoints, request/response formats, and provide a basic Python implementation using Flask.

### Generated Text

Here is a design for a simple REST API for a Todo List application, followed by a basic implementation using Flask.

### **API Design**

#### Endpoints
1.  **`GET /todos`**: Retrieve all existing tasks.
2.  **`POST /todos`**: Create a new task.
3.  **`GET /todos/{id}`**: Retrieve a specific task by ID.
4.  **`PUT /todos/{id}`**: Update a specific task.
5.  **`DELETE /todos/{id}`**: Delete a specific task.

#### Request/Response Formats
-   **Request Body (JSON)**: `{"title": "string", "completed": boolean}`
-   **Response Body (JSON)**: `{"id": integer, "title": "string", "completed": boolean}`

### **Python Implementation**

```python
from flask import Flask, jsonify, request
import json

app = Flask(__name__)

# In-memory storage to simulate a database
todos = []

def generate_id():
    return len(todos) + 1

@app.route('/todos')
def get_todos():
    return jsonify(
        todos
    )
