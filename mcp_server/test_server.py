#!/usr/bin/env python3
"""
Test script for Blue Bikes MCP Server

This script tests the MCP server by calling each tool and verifying responses.
"""

import sys
from pathlib import Path

parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

import asyncio
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_mcp_server():
    """Test all MCP server tools."""

    server_script = Path(__file__).parent / "bluebikes_mcp_server.py"

    server_params = StdioServerParameters(
        command="python",
        args=[str(server_script)],
    )

    print("🚀 Starting Blue Bikes MCP Server Test")
    print("=" * 70)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            print("\n[1/8] Initializing session...")
            await session.initialize()
            print("✓ Session initialized")

            print("\n[2/8] Listing available tools...")
            tools_response = await session.list_tools()
            tools = tools_response.tools
            print(f"✓ Found {len(tools)} tools:")
            for tool in tools:
                print(f"  - {tool.name}: {tool.description[:60]}...")

            # Test each tool
            print("\n[3/8] Testing: get_summary_stats")
            try:
                result = await session.call_tool("get_summary_stats", {})
                data = json.loads(result.content[0].text)
                print(f"✓ Total stations: {data.get('total_stations', 'N/A')}")
                print(f"  Average demand: {data.get('average_demand', 'N/A')}")
            except Exception as e:
                print(f"✗ Error: {e}")

            print("\n[4/8] Testing: get_top_stations (n=5)")
            try:
                result = await session.call_tool("get_top_stations", {"n": 5})
                data = json.loads(result.content[0].text)
                print(f"✓ Top 5 stations retrieved")
                if "stations" in data:
                    for i, station in enumerate(data["stations"][:3], 1):
                        print(f"  {i}. Station {station['pickup_location_id']}: {station['predicted_demand']} rides")
            except Exception as e:
                print(f"✗ Error: {e}")

            print("\n[5/8] Testing: get_next_hour_predictions")
            try:
                result = await session.call_tool("get_next_hour_predictions", {})
                data = json.loads(result.content[0].text)
                print(f"✓ Predictions for {data.get('total_stations', 0)} stations")
                print(f"  Prediction hour: {data.get('prediction_hour', 'N/A')}")
            except Exception as e:
                print(f"✗ Error: {e}")

            print("\n[6/8] Testing: fetch_predictions (hours=12)")
            try:
                result = await session.call_tool("fetch_predictions", {"hours": 12})
                data = json.loads(result.content[0].text)
                print(f"✓ Fetched {data.get('total_records', 0)} prediction records")
            except Exception as e:
                print(f"✗ Error: {e}")

            print("\n[7/8] Testing: fetch_actual_rides (hours=12)")
            try:
                result = await session.call_tool("fetch_actual_rides", {"hours": 12})
                data = json.loads(result.content[0].text)
                print(f"✓ Fetched {data.get('total_records', 0)} actual ride records")
            except Exception as e:
                print(f"✗ Error: {e}")

            print("\n[8/8] Testing: get_model_metrics (hours=24)")
            try:
                result = await session.call_tool("get_model_metrics", {"hours": 24})
                data = json.loads(result.content[0].text)
                if "metrics" in data:
                    print(f"✓ Model Metrics (24 hours):")
                    print(f"  MAE: {data['metrics'].get('mae', 'N/A')}")
                    print(f"  RMSE: {data['metrics'].get('rmse', 'N/A')}")
                    print(f"  R² Score: {data['metrics'].get('r2_score', 'N/A')}")
                else:
                    print(f"✗ Error: {data.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"✗ Error: {e}")

    print("\n" + "=" * 70)
    print("✅ MCP Server Test Completed")


if __name__ == "__main__":
    print("Blue Bikes MCP Server - Test Suite")
    print("Make sure Hopsworks credentials are set in your environment:")
    print("  - HOPSWORKS_API_KEY")
    print("  - HOPSWORKS_PROJECT_NAME")
    print()

    try:
        asyncio.run(test_mcp_server())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed: {e}")
        sys.exit(1)
