#!/usr/bin/env python3
"""
Test the planner function directly to see what errors occur
"""

import sys
import os

from max_planner_no_integration import robot_action_planner_with_objects

def planner():
    """Test the planner function directly"""
    print("="*60)
    print("PLANNER Starting...")
    print("="*60)
    
    # Test data
    task_description = "pick up the paper"
    image_path = "images/scenario.jpeg"
    objects = [
        {
            "object": "paper",
            "object_center_location": {"x": 38, "y": 64, "z": 50},
            "object_color": "white"
        },
        {
            "object": "banana", 
            "object_center_location": {"x": 35, "y": 45, "z": 18},
            "object_color": "yellow"
        },
        {
            "object": "game controller",
            "object_center_location": {"x": 15, "y": 18, "z": 24},
            "object_color": "white"
        }
    ]
    
    try:
        print("Calling robot_action_planner_with_objects...")
        result = robot_action_planner_with_objects(
            task_description=task_description,
            image_path=image_path,
            objects=objects,
            output_format="json",
            quiet=False,
            capture_output=False
        )
        
        print("\n" + "="*60)
        print("RESULT")
        print("="*60)
        print(f"Success: {result.get('success', False)}")
        print(f"Action Plan: {result.get('action_plan', 'None')}")
        print(f"Objects: {result.get('detected_objects', [])}")
        print(f"Relationships: {result.get('spatial_relationships', 'None')[:100]}...")
        
        return result.get('success', False)
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Result of Planner")
    print("="*60)
    
    success = planner()
    
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    
    if success:
        print("✓ Planner function works correctly!")
    else:
        print("✗ Planner function has issues!")
    
    print("="*60)
