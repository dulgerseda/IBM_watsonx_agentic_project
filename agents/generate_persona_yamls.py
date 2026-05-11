import json
from pathlib import Path

personas_dir = Path("personas")
agents_dir = Path("agents")

for json_file in personas_dir.glob("*_persona.json"):
    with open(json_file) as f:
        data = json.load(f)
    
    identity = data["agent_identity"]
    meta = data["meta"]
    
    name = f"Persona_{meta['id'].split('_')[2].capitalize()}"
    
    yaml_content = f"""spec_version: v1
kind: native
name: {name}
description: {meta['agent_label']}
instructions: |
  {identity['system_role']}
  
  Style: {identity['style']}
  
  Objective: {identity['objective']}

llm: watsonx/meta-llama/llama-3-3-70b-instruct
style: default
collaborators: []
tools: []
"""
    
    out_file = agents_dir / f"{name.lower()}.yaml"
    with open(out_file, "w") as f:
        f.write(yaml_content)
    print(f"✓ Created {out_file}")

