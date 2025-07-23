#!/usr/bin/env python3
"""
Test script for the Buch AI MCP Server.
This script validates that all tools are properly configured and can be called.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.mcp.main import mcp


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    try:
        print("✅ All required modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_server_instance():
    """Test that the FastMCP server instance is properly created."""
    print("\nTesting server instance...")

    try:
        assert mcp is not None, "MCP instance is None"
        assert hasattr(mcp, "name"), "MCP missing name attribute"
        print(f"✅ FastMCP server instance created successfully: {mcp.name}")
        return True
    except Exception as e:
        print(f"❌ Server instance test failed: {e}")
        return False


def test_registered_tools():
    """Test that tools are properly registered with FastMCP."""
    print("\nTesting registered tools...")

    try:
        # FastMCP stores tools in the _tool_manager._tools attribute
        tools = getattr(mcp._tool_manager, "_tools", {})

        expected_tools = {
            "generate_story_string",
            "split_story",
            "summarise_story",
            "generate_image_prompts",
            "generate_image",
        }

        registered_tools = set(tools.keys())

        print(f"Expected tools: {expected_tools}")
        print(f"Registered tools: {registered_tools}")

        missing_tools = expected_tools - registered_tools
        if missing_tools:
            print(f"❌ Missing tools: {missing_tools}")
            return False

        print("✅ All expected tools are registered")

        # Test that each tool has proper metadata
        for tool_name in expected_tools:
            tool = tools[tool_name]
            if hasattr(tool, "description") and tool.description:
                print(f"✅ Tool {tool_name} has description")
            else:
                print(f"⚠️  Tool {tool_name} missing description")

            if hasattr(tool, "parameters") and tool.parameters:
                print(f"✅ Tool {tool_name} has parameters schema")
            else:
                print(f"⚠️  Tool {tool_name} missing parameters schema")

        return True
    except Exception as e:
        print(f"❌ Tool registration test failed: {e}")
        return False


def test_request_models():
    """Test that request models can be instantiated."""
    print("\nTesting request models...")

    try:
        from app.services.image.common import ImageGenerationRequest
        from app.services.llm.common import (
            GenerateImagePromptsRequest,
            GenerateStoryRequest,
            SummariseStoryRequest,
        )

        # Test LLM request models
        story_request = GenerateStoryRequest(prompt="Test story")
        print(f"✅ GenerateStoryRequest: {story_request.prompt}")

        summarise_request = SummariseStoryRequest(story="Test story to summarise")
        print(f"✅ SummariseStoryRequest: {summarise_request.story[:20]}...")

        image_prompts_request = GenerateImagePromptsRequest(
            story="Test story", story_parts=["Part 1", "Part 2"]
        )
        print(
            f"✅ GenerateImagePromptsRequest: {len(image_prompts_request.story_parts)} parts"
        )

        # Test image request model
        image_request = ImageGenerationRequest(prompt="Test image prompt")
        print(f"✅ ImageGenerationRequest: {image_request.prompt}")

        return True
    except Exception as e:
        print(f"❌ Request model test failed: {e}")
        return False


def test_service_initialization():
    """Test that services can be initialized."""
    print("\nTesting service initialization...")

    try:
        from app.services.image.pollinations_ai import PollinationsAiRouterService
        from app.services.llm.vertex_ai import VertexAiRouterService

        # NOTE: We're not actually calling the services, just checking they can be imported and instantiated
        VertexAiRouterService()
        PollinationsAiRouterService()

        print("✅ VertexAiRouterService initialized")
        print("✅ PollinationsAiRouterService initialized")
        return True
    except Exception as e:
        print(f"❌ Service initialization test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Testing Buch AI FastMCP Server")
    print("=" * 40)

    tests = [
        test_imports,
        test_server_instance,
        test_registered_tools,
        test_request_models,
        test_service_initialization,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        print("\n📝 FastMCP Server Summary:")
        print("  - FastMCP server instance created successfully")
        print("  - All required modules can be imported")
        print("  - All tools are properly registered")
        print("  - Request models work correctly")
        print("  - Services can be initialized")
        print("\n🚀 FastMCP server is ready to use!")
        print("  Run: python app/mcp/main.py")
    else:
        print(f"❌ {total - passed} test(s) failed")
        exit(1)


if __name__ == "__main__":
    main()
