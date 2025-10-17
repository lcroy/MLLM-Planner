#!/usr/bin/env python3
"""
Enhanced Robot Action Planner - 2-Step Pipeline with Online API

A modular robot action planning system that:
1. Analyzes images with vision model (Qwen2.5-VL) for spatial relationships and colors
2. Generates executable action plans with actual coordinates using LLM (Llama)

Author: Chen Li
"""

import sys
import os
import time
import json
import requests
import io
import contextlib
import base64

# =============================================================================
# CONFIGURATION
# =============================================================================

# Online API Configuration
OPENROUTER_API_KEY = "xxx"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
ONLINE_VLM_MODEL = "qwen/qwen2.5-vl-72b-instruct:free"  # VLM for image analysis
ONLINE_LLM_MODEL = "meta-llama/llama-3.3-70b-instruct:free"  # LLM for action planning
SITE_URL = "http://localhost:5008"  # Your site URL
SITE_NAME = "Robot Action Planner"  # Your site name

# Available Robot Actions
AVAILABLE_ACTIONS = {
    "MovetoObject": "move to the specified object",
    "PickupObject": "pick up the object", 
    "PutDownObject": "put down the object at the current location",
    "OpenGripper": "open the gripper",
    "CloseGripper": "close the gripper",
    "Home": "return the robot to its home position"
}

# =============================================================================
# GLOBAL STATE MANAGEMENT
# =============================================================================

# Online service status
online_vlm_initialized = False

# Cache for image analysis results (avoids reprocessing same image)
cached_image_path = None
cached_vlm_objects = None
cached_vlm_relationships = None

# Pipeline information display
print("="*80)
print("Max - ROBOT PLANNER - SIMPLIFIED 2-STEP PIPELINE")
print("="*80)
print("STEP 1: VLM analyzes image for spatial relationships + colors")
print("STEP 2: LLM generates action plan with actual coordinates")
print("="*80)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def encode_image_to_base64(image_path):
    """
    Encode image to base64 string for API calls.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_online_vlm(image_path, prompt, max_tokens=512):
    """
    Call the online VLM API (Qwen2.5-VL through OpenRouter) using requests with streaming.
    
    Args:
        image_path (str): Path to the image file
        prompt (str): The prompt to send to the API
        max_tokens (int): Maximum tokens in response
        
    Returns:
        str or None: Response text or None if error
    """
    try:
        # Read and encode the image
        base64_image = encode_image_to_base64(image_path)
        data_url = f"data:image/jpeg;base64,{base64_image}"

        print(f"[INFO] Calling VLM API with model: {ONLINE_VLM_MODEL}")
        
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": SITE_URL,
                "X-Title": SITE_NAME,
            },
            data=json.dumps({
                "model": ONLINE_VLM_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url
                                }
                            }
                        ]
                    }
                ],
                "stream": True
            }),
            stream=True
        )
        
        response.raise_for_status()

        # Stream and collect response
        print("[STREAMING VLM RESPONSE]")
        print("-" * 50)
        full_response = ""
        for line in response.iter_lines():
            if line and line.decode('utf-8').startswith("data: ") and line.decode('utf-8') != "data: [DONE]":
                try:
                    chunk = json.loads(line.decode('utf-8')[6:])  # Remove "data: "
                    content = chunk['choices'][0]['delta'].get('content', '')
                    if content:
                        print(content, end='', flush=True)  # Stream to terminal
                        full_response += content
                except:
                    pass
        print()  # New line when done
        print("-" * 50)
        print("[VLM RESPONSE COMPLETE]")

        if len(full_response) > 0:
            return full_response
        else:
            print(f"[ERROR] No response content found from online VLM")
            return None
            
    except Exception as e:
        print(f"[ERROR] Online VLM API call failed: {e}")
        return None

def call_online_llm(prompt, max_tokens=512):
    """
    Call the online LLM API through OpenRouter using requests with streaming.
    
    Args:
        prompt (str): The prompt to send to the API
        max_tokens (int): Maximum tokens in response
        
    Returns:
        str or None: Response text or None if error
    """
    try:
        print(f"[INFO] Calling LLM API with model: {ONLINE_LLM_MODEL}")
        
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": SITE_URL,
                "X-Title": SITE_NAME,
            },
            data=json.dumps({
                "model": ONLINE_LLM_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": True
            }),
            stream=True
        )
        
        response.raise_for_status()

        # Stream and collect response
        print("[STREAMING LLM RESPONSE]")
        print("-" * 50)
        full_response = ""
        for line in response.iter_lines():
            if line and line.decode('utf-8').startswith("data: ") and line.decode('utf-8') != "data: [DONE]":
                try:
                    chunk = json.loads(line.decode('utf-8')[6:])  # Remove "data: "
                    content = chunk['choices'][0]['delta'].get('content', '')
                    if content:
                        print(content, end='', flush=True)  # Stream to terminal
                        full_response += content
                except:
                    pass
        print()  # New line when done
        print("-" * 50)
        print("[LLM RESPONSE COMPLETE]")

        if len(full_response) > 0:
            return full_response
        else:
            print(f"[ERROR] No response content found from online LLM")
            return None
            
    except Exception as e:
        print(f"[ERROR] Online LLM API call failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# MODEL MANAGEMENT
# =============================================================================

def load_vision_model():
    """
    Initialize connection to online vision model service.
    
    Returns:
        bool: True if successful, False otherwise
    """
    global online_vlm_initialized
    
    print("\n INITIALIZING VISION MODEL FOR 2-STEP PIPELINE\n")
    
    # Test API connection
    print(f"[INFO] Testing connection to online VLM service...")
    print(f"[INFO] Model: {ONLINE_VLM_MODEL}")
    print(f"[INFO] API Endpoint: {OPENROUTER_API_URL}")
    
    try:
        # Test the API with a simple request
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": SITE_URL,
            "X-Title": SITE_NAME
        }
        
        # Simple test payload
        test_payload = {
            "model": ONLINE_VLM_MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5
        }
        
        test_response = requests.post(
            OPENROUTER_API_URL, 
            headers=headers, 
            data=json.dumps(test_payload), 
            timeout=10
        )
        
        if test_response.status_code == 200:
            print(f"[INFO] Successfully connected to online VLM service")
        else:
            print(f"[WARNING] API test returned status {test_response.status_code}, but will proceed")
        
    except Exception as e:
        print(f"[WARNING] API connection test failed: {e}, but will proceed anyway")
    
    # Mark as initialized
    online_vlm_initialized = True
    
    print(f"\n Online vision model initialized! Ready for 2-Step pipeline.\n")
    return True

def cleanup_models():
    """Clean up online service status (no actual cleanup needed for online models)."""
    global online_vlm_initialized
    
    print("[INFO] Resetting online service status...")
    
    online_vlm_initialized = False
    
    print("[INFO] Online service status reset.")

# =============================================================================
# 2-STEP PIPELINE FUNCTIONS
# =============================================================================

def step4_action_planning_with_coordinates(task_description, objects_list, scene_description, object_coordinates):
    """
    STEP 2: Generate executable action plan using VLM analysis and actual object coordinates.
    
    Args:
        task_description (str): Original task description
        objects_list (list): List of object names
        scene_description (str): Combined VLM analysis and coordinate data
        object_coordinates (dict): Map of object names to (x, y, z) coordinates
        
    Returns:
        str or None: Final action plan with actual coordinates or None if failed
    """
    print(f"\n[INFO] Planning action sequence for task: {task_description}")
    
    print(f"[INFO] Objects in scene: {', '.join(objects_list)}")
    print(f"[INFO] Coordinates available: {list(object_coordinates.keys())}")
    
    prompt = f"""You are a robot action planner. Create a sequence of robot actions for the given task using the ACTUAL coordinates provided.

TASK: {task_description}
OBJECTS IN SCENE: {', '.join(objects_list)}

SCENE ANALYSIS:
{scene_description}

ROBOT ACTIONS AVAILABLE:
- MovetoObject: move to an object
- PickupObject: pick up an object
- PutDownObject: put down an object
- OpenGripper: open gripper
- CloseGripper: close gripper
- Home: return to home position

CRITICAL PLANNING RULES:
1. If you need to pick up an object that has other objects ON TOP OF it, you MUST first move those blocking objects away.
2. You cannot pick up an object if another object is resting on it.
3. Always check the VLM spatial analysis for blocking relationships.
4. USE THE ACTUAL COORDINATES provided for each object in the action plan.

COORDINATE FORMAT:
Each object has real 3D coordinates. Use them in the format: (ObjectName, (x, y, z), ActionName)

STEP-BY-STEP ANALYSIS REQUIRED:
1. Understand the user's task and identify target objects
2. Check VLM analysis: Is anything ON TOP OF or BLOCKING the target object?
3. If YES: Move blocking objects first (with their actual coordinates)
4. If NO: Pick up target directly (with its actual coordinates)
5. For multi-step tasks (e.g., "pick up X and put it on Y"), plan the full sequence

IMPORTANT: Use the EXACT coordinates provided in the scene description for each object action.

FINAL ACTION PLAN FORMAT (use this exact format with actual coordinates):
plan: (ObjectName, (x, y, z), ActionName) -> (ObjectName, (x, y, z), ActionName) -> ...

Example with actual coordinates:
plan: (Pen, (15, 20, 30), MovetoObject) -> (Pen, (15, 20, 30), PickupObject) -> (USB Drive, (25, 25, 15), MovetoObject) -> (Pen, (25, 25, 15), PutDownObject)

Now generate the action plan for the given task:"""

    try:
        print("[INFO] Running STEP 2 action planning via online LLM... ")
        
        # Call online LLM API
        plan_response = call_online_llm(prompt, max_tokens=1024)
        
        if plan_response is None:
            print("[ERROR] Online API call failed for STEP 2.")
            return None
        
        print(f"[LLM OUTPUT] Action Plan Response:\n{plan_response}")
        
        # Extract plan from response
        final_plan = extract_plan_from_response_with_coords(plan_response, task_description, objects_list, scene_description, object_coordinates)
        
        print(f"[STEP 2 RESULT] Final Action Plan: {final_plan}")
        print(f"[SUCCESS] Action planning completed with actual coordinates")
        
        return final_plan
        
    except Exception as e:
        print(f"[ERROR] Error during STEP 2 inference: {e}")
        return None

def extract_plan_from_response_with_coords(plan_response, task_description, objects_list, scene_description, object_coordinates):
    """
    Extract action plan from LLM response, ensuring actual coordinates are used.
    
    Args:
        plan_response (str): Raw LLM response
        task_description (str): Original task
        objects_list (list): List of objects
        scene_description (str): Scene description
        object_coordinates (dict): Actual coordinates for each object
        
    Returns:
        str: Formatted action plan with actual coordinates
    """
    import re
    
    # Method 1: Look for "plan:" pattern with coordinates
    plan_matches = list(re.finditer(r'plan:\s*\(', plan_response, re.IGNORECASE))
    if plan_matches:
        last_plan_start = plan_matches[-1].start()
        plan_section = plan_response[last_plan_start:]
        
        lines = plan_section.split('\n')
        plan_line = lines[0].strip()
        
        # Check if plan continues on subsequent lines
        for i in range(1, len(lines)):
            line = lines[i].strip()
            if line.startswith('->') or (line and '->' in line):
                plan_line += ' ' + line
            else:
                break
        
        if plan_line and len(plan_line) > 10:
            # Verify it has coordinates (not just (0.0, 0.0))
            if re.search(r'\(\d+(?:\.\d+)?,\s*\d+(?:\.\d+)?,\s*\d+(?:\.\d+)?\)', plan_line):
                print(f"[INFO] Extracted plan with coordinates from LLM response")
                return plan_line
    
    # Method 2: Look for action patterns with coordinates
    action_pattern = r'\([A-Za-z\s]+,\s*\(\d+(?:\.\d+)?,\s*\d+(?:\.\d+)?,\s*\d+(?:\.\d+)?\),\s*[A-Za-z]+\)'
    if re.search(action_pattern, plan_response):
        actions = re.findall(action_pattern, plan_response)
        if actions:
            print(f"[INFO] Found {len(actions)} actions with coordinates")
            return "plan: " + " -> ".join(actions)
    
    # Method 3: Fallback - create plan with actual coordinates
    print("[WARNING] LLM response doesn't include proper coordinates. Creating fallback plan with actual coordinates...")
    
    # Identify target objects from task
    target_objects = []
    task_lower = task_description.lower()
    
    for obj in objects_list:
        if obj.lower() in task_lower:
            target_objects.append(obj)
    
    if not target_objects:
        target_objects = objects_list[:1]  # Use first object as fallback
    
    # Create plan with actual coordinates
    plan_steps = []
    
    for target_obj in target_objects:
        coords = object_coordinates.get(target_obj, (0, 0, 0))
        
        # Check if object is blocked (from scene description)
        is_blocked = False
        blocking_objects = []
        
        scene_lower = scene_description.lower()
        target_lower = target_obj.lower()
        
        for other_obj in objects_list:
            if other_obj != target_obj:
                # Check various blocking patterns
                blocking_patterns = [
                    f"{other_obj.lower()} is on top of {target_lower}",
                    f"{other_obj.lower()} on {target_lower}",
                    f"on top of the {target_lower}",
                ]
                
                if any(pattern in scene_lower for pattern in blocking_patterns):
                    blocking_objects.append(other_obj)
                    is_blocked = True
        
        # Move blocking objects first
        if blocking_objects:
            print(f"[INFO] {target_obj} is blocked by: {', '.join(blocking_objects)}")
            for blocking_obj in blocking_objects:
                block_coords = object_coordinates.get(blocking_obj, (0, 0, 0))
                plan_steps.extend([
                    f"({blocking_obj}, {block_coords}, MovetoObject)",
                    f"({blocking_obj}, {block_coords}, PickupObject)",
                    f"({blocking_obj}, {block_coords}, PutDownObject)"
                ])
        
        # Then handle target object
        plan_steps.extend([
            f"({target_obj}, {coords}, MovetoObject)",
            f"({target_obj}, {coords}, PickupObject)"
        ])
        
        # If task mentions "put on" another object, add that
        if "put" in task_lower and "on" in task_lower:
            for obj in objects_list:
                if obj != target_obj and obj.lower() in task_lower and obj not in target_objects[:-1]:
                    dest_coords = object_coordinates.get(obj, (0, 0, 0))
                    plan_steps.extend([
                        f"({obj}, {dest_coords}, MovetoObject)",
                        f"({target_obj}, {dest_coords}, PutDownObject)"
                    ])
                    break
    
    if plan_steps:
        final_plan = "plan: " + " -> ".join(plan_steps)
        print(f"[INFO] Generated fallback plan with actual coordinates")
        return final_plan
    
    # Last resort
    return f"plan: ({objects_list[0]}, {object_coordinates.get(objects_list[0], (0, 0, 0))}, MovetoObject)"

# =============================================================================
# MAIN PIPELINE FUNCTION
# =============================================================================

def robot_action_planner_with_objects(task_description, image_path, objects, output_format="json", quiet=False, capture_output=False):
    """
    Entry function for robot action planning with pre-defined objects and coordinates.
    This STILL runs VLM analysis to understand spatial relationships and colors,
    then combines VLM results with provided object coordinates for action planning.
    
    Args:
        task_description (str): Description of the task for the robot to perform
        image_path (str): Path to the image file (will be analyzed by VLM)
        objects (list): List of object dictionaries with format:
            [
                {"object": "pen", "object_center_location": {"x": 15, "y": 20, "z": 30}, "object_color": "white"},
                {"object": "USB Drive", "object_center_location": {"x": 25, "y": 25, "z": 15}, "object_color": "blue"}
            ]
        output_format (str): Output format - "json" or "text" (default: "json")
        quiet (bool): Suppress progress output, only return result (default: False)
        capture_output (bool): Capture all terminal output and return as string (default: False)
    
    Returns:
        dict: Result dictionary containing:
            - task_description: The input task
            - image_path: The input image path
            - detected_objects: List of objects with coordinates
            - spatial_relationships: Spatial relationship description from VLM
            - action_plan: Generated action plan with actual coordinates
            - success: Boolean indicating if planning succeeded
            - terminal_output: Captured terminal output (if capture_output=True)
    
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If task description is empty or objects invalid
        RuntimeError: If model loading or planning fails
    """
    # Validate inputs
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found.")
    
    if not task_description.strip():
        raise ValueError("Task description cannot be empty.")
    
    if not objects or not isinstance(objects, list):
        raise ValueError("Objects must be a non-empty list.")
    
    # Initialize output capture
    captured_output = ""
    original_stdout = sys.stdout
    
    try:
        # Set up output capturing
        if quiet:
            output_buffer = io.StringIO()
            sys.stdout = output_buffer
        elif capture_output and not quiet:
            output_buffer = io.StringIO()
            original_stdout = sys.stdout
            
            class TeeOutput:
                def __init__(self, terminal, buffer):
                    self.terminal = terminal
                    self.buffer = buffer
                
                def write(self, text):
                    self.terminal.write(text)
                    self.buffer.write(text)
                    return len(text)
                
                def flush(self):
                    self.terminal.flush()
                    self.buffer.flush()
            
            sys.stdout = TeeOutput(original_stdout, output_buffer)
        
        # Show startup message if not quiet
        if not quiet:
            print("\n Starting Robot Action Planner with Pre-Defined Objects + VLM Analysis")
            print("="*80)
        
        # Initialize online VLM service
        if not load_vision_model():
            if quiet or (capture_output and not quiet):
                sys.stdout = original_stdout
            raise RuntimeError("Failed to initialize online vision service.")
        
        # Extract object names and coordinates from JSON
        object_names = []
        object_coordinates = {}  # Map object name to coordinates
        
        for obj_data in objects:
            obj_name = obj_data.get('object', '').strip().title()
            if obj_name:
                object_names.append(obj_name)
                
                location = obj_data.get('object_center_location', {})
                x = location.get('x', 0)
                y = location.get('y', 0)
                z = location.get('z', 0)
                
                object_coordinates[obj_name] = (x, y, z)
        
        if not object_names:
            raise ValueError("No valid objects found in the provided list.")
        
        print(f"\n[INFO] Received {len(object_names)} objects with coordinates: {', '.join(object_names)}")
        print(f"[INFO] Object coordinates: {object_coordinates}")
        
        # STEP 1: Run VLM analysis on the image to understand spatial relationships
        print("\n" + "="*60)
        print("STEP 1: VLM ANALYSIS FOR SPATIAL RELATIONSHIPS AND COLORS")
        print("="*60)
        
        # Create prompt that includes the known objects
        vlm_prompt = f"""Analyze this image to understand the spatial relationships and visual properties of the objects.

The following objects are present in the scene: {', '.join(object_names)}

For each object, describe:
1. Its color and appearance
2. Its spatial relationship to other objects (on top of, to the left of, to the right of, overlapping, etc.)
3. Whether it's accessible or blocked by other objects

Be precise about spatial relationships, especially:
- Which objects are ON TOP OF other objects
- Which objects are UNDERNEATH other objects
- Which objects are TO THE LEFT or TO THE RIGHT
- Any objects that OVERLAP

Format your response clearly for each object."""

        # Call VLM
        vlm_response = call_online_vlm(image_path, vlm_prompt, max_tokens=1024)
        
        if vlm_response is None:
            print("[ERROR] Failed to get VLM analysis")
            raise RuntimeError("VLM analysis failed")
        
        print(f"[STEP 1 RESULT] VLM Spatial Analysis:\n{vlm_response}")
        
        # STEP 2: Combine VLM analysis with coordinate data and generate action plan
        print("\n" + "="*60)
        print("STEP 2: GENERATING ACTION PLAN WITH VLM + COORDINATES")
        print("="*60)
        
        # Create comprehensive scene description
        combined_description = f"""### VLM Spatial Analysis:
{vlm_response}

### Object Coordinates (from external system):
"""
        for obj_name, coords in object_coordinates.items():
            combined_description += f"- {obj_name}: location ({coords[0]}, {coords[1]}, {coords[2]})\n"
        
        print(f"[INFO] Combined VLM spatial analysis with object coordinates")
        
        # Generate action plan with actual coordinates
        final_plan = step4_action_planning_with_coordinates(
            task_description, 
            object_names, 
            combined_description,
            object_coordinates
        )
        
        # Capture the output if requested
        if quiet:
            captured_output = output_buffer.getvalue()
            sys.stdout = original_stdout
        elif capture_output and not quiet:
            captured_output = output_buffer.getvalue()
            sys.stdout = original_stdout
        
        print("\n" + "="*60)
        print("SIMPLIFIED PIPELINE COMPLETED (VLM → ACTION PLAN)")
        print("="*60)
        
    except Exception as e:
        # Restore stdout in case of error
        if quiet or (capture_output and not quiet):
            captured_output = output_buffer.getvalue() if 'output_buffer' in locals() else ""
            sys.stdout = original_stdout
        cleanup_models()
        raise RuntimeError(f"Planning failed: {e}")
    
    # Check if pipeline succeeded
    if final_plan is None:
        cleanup_models()
        raise RuntimeError("Failed to generate action plan.")
    
    # Prepare result with coordinates
    result = {
        "task_description": task_description,
        "image_path": image_path,
        "detected_objects": object_names,
        "spatial_relationships": vlm_response,
        "action_plan": final_plan,
        "success": final_plan is not None,
        "object_coordinates": object_coordinates
    }
    
    # Add captured output if requested
    if capture_output:
        result["terminal_output"] = captured_output
    
    # Cleanup models
    cleanup_models()
    
    return result
