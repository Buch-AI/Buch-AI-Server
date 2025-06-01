# Buch AI MCP Server

This directory contains the Model Context Protocol (MCP) server implementation for Buch AI.

## Configuration

The MCP server configuration is dynamically generated using the `config_generator.py` script. This ensures that:

1. **UV Executable Path**: Automatically detected using `which uv` command
2. **Project Directory**: Uses `PROJECT_ROOT` from the main `config.py` file

### Generated Configuration

The `config.json` file is automatically generated with the following structure:

```json
{
  "mcpServers": {
    "buch-ai": {
      "command": "/path/to/uv",
      "args": [
        "--directory",
        "/path/to/Buch-AI-Server/",
        "run",
        "app/mcp/main.py"
      ],
      "host": "127.0.0.1",
      "port": 8050,
      "timeout": 30000
    }
  }
}
```

### Usage

The configuration is automatically generated when running the MCP server:

```bash
python app/mcp/main.py
```

Or you can generate it manually:

```bash
python app/mcp/config_generator.py
```

## Available Tools

The MCP server provides the following tools:

- `generate_story_string`: Generate a complete story as a single string
- `split_story`: Generate a story split into parts and sub-parts
- `summarise_story`: Summarise an existing story
- `generate_image_prompts`: Generate image prompts for story sections
- `generate_image`: Generate an image based on a text prompt

Each tool supports different model types (`lite`, `standard`, `pro`, `max`) and optional cost centre tracking.

## Features

The FastMCP server exposes the following tools:

### LLM Tools
- **generate_story_string**: Generate a complete story as a single string
- **split_story**: Generate a story split into parts and sub-parts  
- **summarise_story**: Summarise an existing story
- **generate_image_prompts**: Generate image prompts for each section of a story

### Image Tools
- **generate_image**: Generate an image based on a text prompt

## Setup

1. **Install Dependencies**
   ```bash
   pip install -e .
   ```

2. **Set Environment Variables**
   Make sure you have the necessary environment variables set for your LLM and image providers (Vertex AI, Pollinations AI, etc.).

3. **Test the Server**
   ```bash
   python app/mcp/main.py
   ```

## Usage with MCP Clients

### With Claude Desktop

1. Add the server configuration to your Claude Desktop config:
   ```json
   {
     "mcpServers": {
       "buch-ai": {
         "command": "python",
         "args": ["/path/to/your/project/app/mcp/main.py"],
         "env": {
           "PYTHONPATH": "/path/to/your/project"
         }
       }
     }
   }
   ```

2. Restart Claude Desktop and the tools will be available.

### With Other MCP Clients

Use the provided `config.json` in this directory as a reference for configuring other MCP clients.

## Project Structure

```
app/
├── mcp/
│   ├── main.py          # FastMCP server implementation
│   ├── config.json      # MCP client configuration
│   ├── test.py          # Test suite
│   ├── README.md        # This documentation
│   └── __init__.py
├── services/
│   ├── llm/             # LLM service implementations
│   └── image/           # Image service implementations
└── ...
```

## Tool Usage Examples

### Generate a Story
```json
{
  "tool": "generate_story_string",
  "arguments": {
    "prompt": "Write a story about a brave knight",
    "model_type": "lite"
  }
}
```

### Generate Image Prompts
```json
{
  "tool": "generate_image_prompts", 
  "arguments": {
    "story": "Once upon a time, there was a brave knight...",
    "story_parts": ["The knight begins his journey", "The knight faces the dragon"],
    "model_type": "lite"
  }
}
```

### Generate an Image
```json
{
  "tool": "generate_image",
  "arguments": {
    "prompt": "A brave knight in shining armor",
    "width": 720,
    "height": 720
  }
}
```

## Model Types

Available model types for LLM operations:
- `lite`: Fast, basic quality
- `standard`: Balanced speed and quality  
- `pro`: Higher quality, slower
- `max`: Maximum quality, slowest

## Response Format

All responses are JSON strings containing the results and usage information:

```json
{
  "story": "Generated story text...",
  "usage": {
    "embedding_tokens": 0,
    "generation_prompt_tokens": 25,
    "generation_completion_tokens": 150,
    "total_tokens": 175,
    "cost": 0.001
  }
}
```

For images, the response includes base64-encoded image data:

```json
{
  "image_data": "base64-encoded-image-data...",
  "content_type": "image/jpeg",
  "note": "Image data is base64 encoded. Decode to get the actual image bytes."
}
```

## Development

The FastMCP server is built using:
- `FastMCP` for easy MCP server implementation with decorators
- FastAPI services for the actual AI functionality
- Vertex AI for LLM operations
- Pollinations AI for image generation

### Adding New Tools

To add new tools, simply define a function with the `@mcp.tool()` decorator:

```python
@mcp.tool()
def your_new_tool(param1: str, param2: int = 42) -> str:
    """Description of your tool.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
    """
    # Your implementation here
    return "result"
```

FastMCP automatically handles:
- Type validation based on function signatures
- JSON schema generation for parameters
- Tool registration and discovery

## Testing

Run the test suite to validate the FastMCP server:

```bash
python app/mcp/test.py
```

This will verify that all tools are properly configured and can handle the expected request formats.

## Key Advantages of FastMCP

1. **Decorator-based**: Simple `@mcp.tool()` decorators for tool definition
2. **Automatic schema generation**: No need to manually define JSON schemas
3. **Type safety**: Uses Python type hints for validation
4. **Less boilerplate**: Much simpler than low-level MCP server implementation
5. **Built-in documentation**: Docstrings become tool descriptions automatically 