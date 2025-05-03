#!/usr/bin/env python
"""
Script to list all artifacts in a Weights & Biases project.
"""

import argparse
import wandb
from tabulate import tabulate  # For prettier table output (optional)

def parse_args():
    parser = argparse.ArgumentParser(description="List all artifacts in a W&B project")
    parser.add_argument("--entity", type=str, required=True, 
                        help="W&B entity name (username or team name)")
    parser.add_argument("--project", type=str, required=True,
                        help="W&B project name")
    parser.add_argument("--api_key", type=str, required=True,
                        help="W&B API key")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Optional: Filter artifacts by run ID")
    parser.add_argument("--type", type=str, default=None,
                        help="Optional: Filter artifacts by type (e.g., 'model')")
    return parser.parse_args()

def list_artifacts(entity, project, api_key, run_id=None, artifact_type=None):
    """
    List all artifacts in a W&B project with optional filtering.
    
    Args:
        entity: W&B entity name (username or team name)
        project: W&B project name
        api_key: W&B API key
        run_id: Optional run ID to filter artifacts
        artifact_type: Optional artifact type to filter by
    """
    # Login to W&B
    wandb.login(key=api_key)
    api = wandb.Api()
    
    # Prepare full project path
    project_path = f"{entity}/{project}"
    print(f"Fetching artifacts from project: {project_path}")
    
    # Get all artifacts in the project
    artifacts = []
    
    # If run_id is provided, get artifacts for that specific run
    if run_id:
        try:
            run = api.run(f"{project_path}/{run_id}")
            run_artifacts = run.logged_artifacts()
            print(f"Found {len(run_artifacts)} artifacts in run {run_id}")
            artifacts.extend(run_artifacts)
        except Exception as e:
            print(f"Error fetching run {run_id}: {e}")
            return
    else:
        # Get all artifacts in the project
        try:
            # Get all artifact types
            artifact_types = api.artifact_types(project_path)
            
            # If artifact_type is specified, filter by it
            if artifact_type:
                if artifact_type in artifact_types:
                    artifact_types = [artifact_type]
                else:
                    print(f"Artifact type '{artifact_type}' not found in project")
                    print(f"Available types: {', '.join(artifact_types)}")
                    return
            
            # Fetch all artifacts
            for art_type in artifact_types:
                type_artifacts = api.artifacts(project_path, type=art_type)
                print(f"Found {len(type_artifacts)} artifacts of type '{art_type}'")
                artifacts.extend(type_artifacts)
                
        except Exception as e:
            print(f"Error fetching artifacts: {e}")
            return
    
    # Display artifacts
    if not artifacts:
        print("No artifacts found matching the criteria")
        return
        
    # Prepare table data
    headers = ["Name", "Type", "Version", "Size", "Created", "Run ID"]
    table_data = []
    
    for artifact in artifacts:
        # Get run ID if available
        run_id = "N/A"
        if hasattr(artifact, "logged_by") and artifact.logged_by():
            run_id = artifact.logged_by().id
            
        # Add artifact info to table
        table_data.append([
            artifact.name,
            artifact.type,
            artifact.version,
            f"{artifact.size/(1024*1024):.2f} MB",
            artifact.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            run_id
        ])
    
    # Print table
    try:
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    except NameError:
        # Fallback if tabulate is not installed
        print("\t".join(headers))
        for row in table_data:
            print("\t".join(str(item) for item in row))
            
    print(f"\nTotal artifacts: {len(artifacts)}")

def main():
    args = parse_args()
    list_artifacts(
        entity=args.entity,
        project=args.project,
        api_key=args.api_key,
        run_id=args.run_id,
        artifact_type=args.type
    )

if __name__ == "__main__":
    main()