# Schedule Optimization System

## Overview
This is a workforce scheduling optimization system that uses Google OR-Tools to create optimal work schedules for delivery riders. The system minimizes understaffing and overstaffing penalties while respecting worker constraints like maximum hours, rest day preferences, and availability restrictions.

## Project Architecture
- **main.py**: Main orchestrator that implements a two-phase optimization approach
  - Phase 1: Generates a heuristic seed solution (20-second time limit)
  - Phase 2: Deep optimization with warm start (600-second time limit)
- **optimizer.py**: Core optimization logic with constraint programming functions
  - Data loading and validation
  - Constraint application (weekly hours, daily limits, rest days, shift structure)
  - Solution processing and output generation

## Current State
- ✅ All Python dependencies installed (pandas, numpy, ortools, matplotlib)
- ✅ Sample data files created (riders_VEC.csv, demand_VEC.csv) 
- ✅ Code issues fixed and application running successfully
- ✅ Workflow configured to run the optimization system
- ✅ System successfully processes data and finds optimized solutions

## Recent Changes (2024-09-23)
- Set up Python environment with required packages
- Created sample CSV data files with realistic rider and demand data
- Fixed code scoping issues in constraint application functions
- Configured console workflow to run the optimization system
- Verified successful data loading, validation, and optimization execution

## Input Files Required
- **riders_VEC.csv**: Rider information including ID, weekly hours, daily limits, rest preferences, and availability
- **demand_VEC.csv**: Hourly demand data by city, date, and 30-minute time slots

## Output Files Generated
- **horario_VEC.csv**: Final optimized schedule with rider assignments
- **horario_semilla_heuristica.csv**: Initial heuristic seed solution (if generated)

## Running the System
The optimization system runs automatically via the configured workflow. It:
1. Validates input data format and business logic
2. Generates an initial heuristic solution (Phase 1)
3. Performs deep optimization using the seed (Phase 2)
4. Outputs final schedule and displays coverage visualization

## User Preferences
- Console-based application for batch processing
- Comprehensive logging of optimization progress
- Visual coverage graphs showing demand vs assignments
- Spanish language interface for user communications

## Technical Notes
- Uses constraint programming (CP-SAT) solver from OR-Tools
- Implements hierarchical penalty system for demand coverage
- Supports rider preferences for rest days and shift restrictions
- Handles various shift structure rules (minimum shift lengths, maximum shifts per day)
- Time-limited optimization with solution monitoring