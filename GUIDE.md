# AI Agent Workflow Guide

## Overview

This document serves as the primary source of truth for AI agents working with this project. It defines the workflow, hierarchy of information, and decision-making process. When instructed to "READ THE GUIDE" or "FOLLOW THE GUIDE," start here and follow the instructions sequentially.

## Core Principles

1. **Hierarchical Information Flow**: All information and instructions flow from a single source of truth.
2. **Specification First**: Every component, feature, or system begins with a specification.
3. **Consistent Structure**: Directory structure mirrors conceptual hierarchy.
4. **Documentation as Code**: Documentation is treated with the same rigor as code.

## Workflow Process

### 1. Initial Orientation

When beginning work on this project, follow these steps in order:

1. Read this GUIDE.md file completely
2. Read project.brain for AI interaction guidelines
3. Read repository.plan for overall repository structure and goals
4. Read the README.md for project overview
5. Read ARCHITECTURE.md for system architecture details

### 2. Task Analysis

When given a specific task:

1. Identify which project the task relates to
2. Locate and read the relevant project.plan file
3. Identify which component or subsystem the task involves
4. Read the corresponding .plan file for that component or subsystem
5. Review existing implementation if applicable

### 3. Implementation Hierarchy

Follow this hierarchy when implementing or modifying features:

1. **Specification**: Begin with the specification document
2. **Planning**: Update or create relevant .plan files
3. **Architecture**: Define the architecture and component relationships
4. **Implementation**: Write the actual code
5. **Documentation**: Update documentation to reflect changes
6. **Testing**: Create or update tests

### 4. Directory Navigation

The repository follows a strict hierarchical structure:

```
Repository Root/
├── GUIDE.md                # This file - start here
├── project.brain           # AI interaction guidelines
├── repository.plan         # Overall repository plan
├── README.md               # Project overview
├── ARCHITECTURE.md         # System architecture
├── docs/                   # README.md shoud link to documentation in this directory
│   ├── project.plan        # project plan
│   ├── structure.plan      # structure plan
│   ├── components.plan     # Component development plan
│   ├── core.plan           # Core features plan
│   └── ...                 # Implementation files
├── [incomplete - please fill in other directories mentioend in style guide or document already existing project ones]
│   ├── [topic].plan      # project plan
│   └── ...                 # Implementation files
└── STYLE-GUIDE/            # Style guide project
    ├── stylebook.plan      # Style guide plan
    └── ...                 # Style guide content
```

### 5. Decision Making Process

When making decisions about implementation:

1. Check if the decision is already specified in a .plan or specification file
2. If not, refer to the parent directory's .plan file
3. If still not specified, refer to the project.plan
4. If still not specified, refer to the STYLE-GUIDE for best practices
5. If still unclear, request clarification from the product owner

### 6. Continuing Work

When instructed to "Following the instructions in GUIDE.md continue work on [x] part of this project":

1. **Identify the Part**: Determine which part of the project [x] refers to:
   - If [x] is a project name, focus on that project
   - If [x] is a component or subsystem, identify which project it belongs to
   - If [x] is a tool or feature, locate its corresponding .plan or specification file

2. **Gather Context**:
   - Read the relevant .plan file for the specified part
   - Review any existing implementation
   - Check for related documentation or specifications

3. **Determine Current Status**:
   - Identify completed tasks in the .plan file
   - Determine which tasks are in progress
   - Identify the next tasks to be completed

4. **Continue Implementation**:
   - Follow the implementation hierarchy (Section 3)
   - Pick up where previous work left off
   - Focus on the next uncompleted tasks in the .plan file

5. **Document Progress**:
   - Update the .plan file with completed tasks
   - Document any decisions or changes made
   - Create or update relevant documentation

Example parts of the project that can be continued:

- [add parts of main project]
- **STYLE-GUIDE/docs**: The style guide documentation

## Project-Specific Guidelines

[fill in]

### STYLE-GUIDE Project

The STYLE-GUIDE documents best practices, coding standards, and development guidelines. Key points:

1. The style guide is the reference for coding standards
2. All code should conform to the style guide
3. The style guide evolves based on project needs
4. Documentation follows the style guide format

## File Types and Their Purpose

- **.md files**: Documentation and specifications
- **.brain files**: AI interaction guidelines
- **.plan files**: Project planning and task tracking
- **.ts/.js files**: Implementation code
- **.css files**: Styling
- **.html files**: Templates and examples

## Updating This Guide

This guide should be updated when:

1. New major components or subsystems are added
2. The workflow process changes
3. New file types or conventions are introduced
4. Project hierarchy is restructured

Updates to this guide should be proposed and reviewed before implementation.

## Next Steps

After reading this guide:

1. Review or create the project.brain file for AI interaction guidelines
2. Review review the repository.plan file for overall project structure
3. Begin task analysis according to Section 2 of this guide