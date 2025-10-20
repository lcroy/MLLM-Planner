# Robot Action Planner - 2-Step Pipeline

A streamlined robot action planning system that combines Vision Language Models (VLM) and Large Language Models (LLM) to generate executable robot action plans from natural language commands and object coordinate data.

## Overview

This system accepts JSON input with pre-defined objects and their coordinates, analyzes images using VLM for spatial relationships, and generates intelligent action plans with actual 3D coordinates.

### Key Features

- **2-Step Pipeline**: VLM analysis → LLM action planning
- **JSON Input**: Accepts objects with 3D coordinates
- **VLM Analysis**: Qwen2.5-VL-72B for spatial relationships and colors
- **LLM Planning**: Llama-3.3-70B for intelligent action planning
- **Real Coordinates**: Outputs plans with actual (x, y, z) coordinates
- **Blocking Detection**: Handles objects that block others
- **Multi-step Tasks**: Supports complex pick-and-place operations

## Architecture (2-Step Pipeline)

```
JSON Input (objects + coordinates)
    ↓
STEP 1: VLM Analysis
    → Analyzes image for spatial relationships
    → Identifies colors and object properties
    → Detects blocking relationships
    ↓
STEP 2: LLM Action Planning
    → Combines VLM analysis + coordinates
    → Generates action plan with actual (x, y, z)
    → Handles multi-step tasks
    ↓
Action Plan Output
```

## Quick Start

### 1. Install Dependencies

```bash
cd /home/ubuntu/projects/vlmplanner
pip install -r requirements.txt
```

### 2. Test the System

```bash
# Test the 2-step pipeline directly
python -c "
from max_planner_no_integration import robot_action_planner_with_objects

result = robot_action_planner_with_objects(
    task_description='pick up the paper',
    image_path='images/scenario.jpeg',
    objects=[
        {'object': 'paper', 'object_center_location': {'x': 38, 'y': 64, 'z': 50}, 'object_color': 'white'},
        {'object': 'banana', 'object_center_location': {'x': 35, 'y': 45, 'z': 18}, 'object_color': 'yellow'}
    ]
)
print('Success:', result['success'])
print('Action Plan:', result['action_plan'])
"
```

### 3. Direct Function Usage

The system works directly with the Python function - no server needed!

## Input Format

```json
{
  "user_message": "pick up the paper",
  "imagePath": "images/scenario.jpeg",
  "objects": [
    {
      "object": "paper",
      "object_center_location": {"x": 38, "y": 64, "z": 50},
      "object_color": "white"
    },
    {
      "object": "banana",
      "object_center_location": {"x": 35, "y": 45, "z": 18},
      "object_color": "yellow"
    }
  ]
}
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_message` | string | Yes | Natural language task description |
| `imagePath` | string | Yes | Path to image file (will be analyzed by VLM) |
| `objects` | array | Yes | List of objects with coordinates and colors |
| `objects[].object` | string | Yes | Object name |
| `objects[].object_center_location` | object | Yes | 3D coordinates (x, y, z) |
| `objects[].object_color` | string | Yes | Object color |

## Output Format

```json
{
  "success": true,
  "detected_objects": ["Paper", "Banana"],
  "spatial_relationships": "### VLM Spatial Analysis:\n...",
  "action_plan": [
    {
      "object": "Paper",
      "object_center_location": {"x": 38, "y": 64, "z": 50},
      "ActionName": "robot_pickUp"
    },
    {
      "object": "Paper",
      "object_center_location": {"x": 38, "y": 64, "z": 50},
      "ActionName": "robot_dropOff"
    },
    {
      "object": "none",
      "object_center_location": {"x": 0, "y": 0, "z": 0},
      "ActionName": "Go_home"
    }
  ],
  "object_coordinates": {
    "Paper": [38, 64, 50],
    "Banana": [35, 45, 18]
  }
}
```

## Processing Pipeline (2 Steps)

### STEP 1: VLM Analysis
The system analyzes the **actual image** to understand:
- Spatial relationships (which objects are on top, left, right, etc.)
- Colors and visual properties  
- Accessibility (which objects are blocked)

**Why VLM?** Coordinates alone can't tell if an object is "resting on top of" vs "floating above" another object. VLM provides this crucial spatial understanding from the actual image.

### STEP 2: LLM Action Planning
The LLM generates the action plan by:
1. Understanding the user's natural language command
2. Analyzing VLM spatial relationships (e.g., which objects block others)
3. Using the **actual coordinates** from JSON
4. Generating action sequence as JSON list: `[{"object": "name", "object_center_location": {"x": x, "y": y, "z": z}, "ActionName": "action"}, ...]`

**That's it!** Just 2 steps: VLM analyzes → LLM plans with coordinates.

## Action Plan Format

Actions are returned as a JSON list of dictionaries:
```json
[
  {
    "object": "ObjectName",
    "object_center_location": {"x": x, "y": y, "z": z},
    "ActionName": "ActionName"
  },
  ...
]
```

### Available Actions

| Action | Description |
|--------|-------------|
| `robot_pickUp` | Robot moves to point (hovering directly above the object), then descends to pick it up using the vacuum gripper |
| `robot_dropOff` | Robot moves to a predefined drop-off location and releases the object held by the vacuum gripper |
| `Go_home` | Moves the robot to a predefined home position using joint motion |

### Example Plans

**Simple pickup:**
```json
[
  {
    "object": "Paper",
    "object_center_location": {"x": 38, "y": 64, "z": 50},
    "ActionName": "robot_pickUp"
  },
  {
    "object": "Paper",
    "object_center_location": {"x": 38, "y": 64, "z": 50},
    "ActionName": "robot_dropOff"
  },
  {
    "object": "none",
    "object_center_location": {"x": 0, "y": 0, "z": 0},
    "ActionName": "Go_home"
  }
]
```

**Pick and place:**
```json
[
  {
    "object": "Pen",
    "object_center_location": {"x": 15, "y": 20, "z": 30},
    "ActionName": "robot_pickUp"
  },
  {
    "object": "Pen",
    "object_center_location": {"x": 25, "y": 25, "z": 15},
    "ActionName": "robot_dropOff"
  },
  {
    "object": "none",
    "object_center_location": {"x": 0, "y": 0, "z": 0},
    "ActionName": "Go_home"
  }
]
```

**Handling blocked objects:**
```json
[
  {
    "object": "USB Drive",
    "object_center_location": {"x": 15, "y": 20, "z": 30},
    "ActionName": "robot_pickUp"
  },
  {
    "object": "USB Drive",
    "object_center_location": {"x": 50, "y": 50, "z": 10},
    "ActionName": "robot_dropOff"
  },
  {
    "object": "Pen",
    "object_center_location": {"x": 15, "y": 20, "z": 10},
    "ActionName": "robot_pickUp"
  },
  {
    "object": "Pen",
    "object_center_location": {"x": 55, "y": 55, "z": 10},
    "ActionName": "robot_dropOff"
  },
  {
    "object": "none",
    "object_center_location": {"x": 0, "y": 0, "z": 0},
    "ActionName": "Go_home"
  }
]
```

## Project Structure

```
vlmplanner/
├── max_planner_no_integration.py    # Core 2-step pipeline (VLM + LLM)
├── planner.py                      # Alternative planner implementation
├── requirements.txt                # Python dependencies
├── README.md                       # This documentation
└── images/
    └── scenario.jpeg               # Sample test image
```

## Configuration

Edit `max_planner_no_integration.py` to configure API keys:

```python
# Online API Configuration
OPENROUTER_API_KEY = "your-api-key-here"
ONLINE_VLM_MODEL = "qwen/qwen2.5-vl-72b-instruct:free"  # VLM for image analysis
ONLINE_LLM_MODEL = "meta-llama/llama-3.3-70b-instruct:free"  # LLM for planning
```

## Testing

### Direct Function Test
```python
from max_planner_no_integration import robot_action_planner_with_objects

result = robot_action_planner_with_objects(
    task_description="pick up the paper",
    image_path="images/scenario.jpeg",
    objects=[
        {"object": "paper", "object_center_location": {"x": 38, "y": 64, "z": 50}, "object_color": "white"}
    ]
)
print(result['action_plan'])
```

### Alternative Implementation Test
```python
from planner import robot_action_planner

result = robot_action_planner(
    task_description="pick up the paper",
    image_path="images/scenario.jpeg",
    objects=[
        {"object": "paper", "object_center_location": {"x": 38, "y": 64, "z": 50}, "object_color": "white"}
    ]
)
print(result['action_plan'])
```

## Example Scenarios

### Scenario 1: Simple Pickup
```json
{
  "user_message": "pick up the paper",
  "imagePath": "images/scenario.jpeg",
  "objects": [
    {"object": "paper", "object_center_location": {"x": 38, "y": 64, "z": 50}, "object_color": "white"}
  ]
}
```
**Result:** 
```json
[
  {
    "object": "Paper",
    "object_center_location": {"x": 38, "y": 64, "z": 50},
    "ActionName": "robot_pickUp"
  },
  {
    "object": "Paper",
    "object_center_location": {"x": 38, "y": 64, "z": 50},
    "ActionName": "robot_dropOff"
  },
  {
    "object": "none",
    "object_center_location": {"x": 0, "y": 0, "z": 0},
    "ActionName": "Go_home"
  }
]
```

### Scenario 2: Pick and Place
```json
{
  "user_message": "pick up the pen and put it on the USB drive",
  "imagePath": "images/scenario.jpeg",
  "objects": [
    {"object": "pen", "object_center_location": {"x": 15, "y": 20, "z": 30}, "object_color": "white"},
    {"object": "USB Drive", "object_center_location": {"x": 25, "y": 25, "z": 15}, "object_color": "blue"}
  ]
}
```
**Result:** Full pick-place-drop sequence with actual coordinates:
```json
[
  {
    "object": "Pen",
    "object_center_location": {"x": 15, "y": 20, "z": 30},
    "ActionName": "robot_pickUp"
  },
  {
    "object": "Pen",
    "object_center_location": {"x": 25, "y": 25, "z": 15},
    "ActionName": "robot_dropOff"
  },
  {
    "object": "none",
    "object_center_location": {"x": 0, "y": 0, "z": 0},
    "ActionName": "Go_home"
  }
]
```

### Scenario 3: Blocked Object (VLM detects pen on top of USB drive)
```json
{
  "user_message": "pick up the USB drive",
  "imagePath": "images/scenario.jpeg",
  "objects": [
    {"object": "pen", "object_center_location": {"x": 15, "y": 20, "z": 30}, "object_color": "white"},
    {"object": "USB Drive", "object_center_location": {"x": 15, "y": 20, "z": 10}, "object_color": "blue"}
  ]
}
```
**Result:** System moves pen first, then picks up USB drive (intelligent blocking detection):
```json
[
  {
    "object": "Pen",
    "object_center_location": {"x": 15, "y": 20, "z": 30},
    "ActionName": "robot_pickUp"
  },
  {
    "object": "Pen",
    "object_center_location": {"x": 50, "y": 50, "z": 10},
    "ActionName": "robot_dropOff"
  },
  {
    "object": "USB Drive",
    "object_center_location": {"x": 15, "y": 20, "z": 10},
    "ActionName": "robot_pickUp"
  },
  {
    "object": "USB Drive",
    "object_center_location": {"x": 55, "y": 55, "z": 10},
    "ActionName": "robot_dropOff"
  },
  {
    "object": "none",
    "object_center_location": {"x": 0, "y": 0, "z": 0},
    "ActionName": "Go_home"
  }
]
```

## Why This Design?

### VLM Provides Spatial Intelligence
Coordinates alone can't tell if objects are:
- Resting on top vs floating above
- Overlapping vs just nearby
- Blocking each other

VLM analyzes the actual image and provides this context.

### Coordinates Provide Precision
VLM can say "pen is to the left" but can't give exact position.  
JSON coordinates provide precise (x, y, z) for robot navigation.

### LLM Provides Intelligence
LLM combines:
- User's natural language command
- VLM's spatial analysis
- Coordinate precision

To create intelligent, executable action plans.

## Dependencies

Install required packages:
```bash
pip install requests
```

Required packages:
- requests - HTTP client library for API calls

## Requirements

- Python 3.7+
- Network connection (for VLM and LLM APIs)
- Valid OpenRouter API key
- Image files in supported formats (JPEG, PNG)

## Troubleshooting

### Import Error
**Problem:** `ModuleNotFoundError: No module named 'max_planner_no_integration'`  
**Solution:** Make sure you're in the correct directory: `cd /home/ubuntu/projects/vlmplanner`

### Image File Not Found
**Problem:** Error: "Image file not found"  
**Solution:** Verify image path is correct and file exists

### VLM Analysis Failed
**Problem:** STEP 1 fails  
**Solution:** 
- Check network connection (VLM is online service)
- Verify API key is valid
- Ensure image file is valid and not corrupted

### Action Plan Has (0, 0, 0) Coordinates
**Problem:** Plan shows zeros instead of actual coordinates  
**Solution:** System has fallback mechanism - check console for warnings

### Processing Timeout
**Problem:** Takes too long or hangs  
**Solution:** 
- VLM + LLM processing takes 10-30 seconds normally
- Check API rate limits
- Verify network connectivity

## Notes

- VLM and LLM are online services (requires network)
- Processing time: ~10-30 seconds per request
- Coordinates use standard 3D coordinate system (x, y, z)
- Object names are converted to Title Case automatically
- The system is designed for research and development use

## License

MIT License - See LICENSE file for details