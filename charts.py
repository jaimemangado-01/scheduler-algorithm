import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import List, Tuple, Optional
import pandas as pd

def create_penalty_chart(solution_data: List[Tuple[int, float, float]], 
                        phase: str = "optimization") -> go.Figure:
    """Create real-time penalty value chart."""
    if not solution_data:
        # Empty chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers', name='Penalty Value'))
        fig.update_layout(
            title=f"{phase.title()} Progress - Penalty Values",
            xaxis_title="Solution Number",
            yaxis_title="Penalty Value",
            showlegend=True
        )
        return fig
    
    # Extract data
    solution_nums = [item[0] for item in solution_data]
    times = [item[1] for item in solution_data]
    penalties = [item[2] for item in solution_data]
    
    # Create figure with secondary y-axis for time
    fig = go.Figure()
    
    # Add penalty trace
    fig.add_trace(go.Scatter(
        x=solution_nums,
        y=penalties,
        mode='lines+markers',
        name='Penalty Value',
        line=dict(color='blue', width=2),
        marker=dict(size=6),
        hovertemplate='<b>Solution %{x}</b><br>Penalty: %{y:.0f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"{phase.title()} Progress - Penalty Optimization",
        xaxis_title="Solution Number",
        yaxis_title="Penalty Value",
        showlegend=True,
        hovermode='closest',
        height=400
    )
    
    return fig

def create_demand_vs_assignment_chart(demand: np.ndarray, assigned: Optional[np.ndarray] = None,
                                    title: str = "Demand vs Assignments") -> go.Figure:
    """Create demand vs assignment visualization."""
    
    # Create time axis (slots)
    time_slots = list(range(len(demand)))
    
    # Create day dividers (every 48 slots)
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    fig = go.Figure()
    
    # Add demand area chart
    fig.add_trace(go.Scatter(
        x=time_slots,
        y=demand,
        fill='tozeroy',
        fillcolor='rgba(255, 99, 132, 0.3)',
        line=dict(color='rgba(255, 99, 132, 0.8)', width=2),
        name='Demand Required',
        hovertemplate='<b>Slot %{x}</b><br>Demand: %{y}<extra></extra>'
    ))
    
    # Add assignments if provided
    if assigned is not None:
        fig.add_trace(go.Scatter(
            x=time_slots,
            y=assigned,
            line=dict(color='rgba(54, 162, 235, 0.8)', width=3),
            name='Riders Assigned',
            hovertemplate='<b>Slot %{x}</b><br>Assigned: %{y}<extra></extra>'
        ))
        
        # Add difference area (overstaffing/understaffing)
        difference = assigned - demand
        colors = ['rgba(255, 99, 132, 0.2)' if x < 0 else 'rgba(54, 162, 235, 0.2)' for x in difference]
        
        fig.add_trace(go.Scatter(
            x=time_slots,
            y=difference,
            fill='tozeroy',
            fillcolor='rgba(255, 206, 84, 0.2)',
            line=dict(color='rgba(255, 206, 84, 0.8)', width=1),
            name='Difference (Assigned - Demand)',
            hovertemplate='<b>Slot %{x}</b><br>Difference: %{y}<extra></extra>'
        ))
    
    # Add vertical lines for day separators
    for day in range(1, 7):
        fig.add_vline(
            x=day * 48,
            line_dash="dash",
            line_color="gray",
            line_width=1,
            opacity=0.5
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time Slot (30-minute intervals)",
        yaxis_title="Number of Riders",
        showlegend=True,
        hovermode='closest',
        height=500,
        xaxis=dict(
            tickmode='array',
            tickvals=[i * 48 + 24 for i in range(7)],
            ticktext=day_names
        )
    )
    
    return fig

def create_coverage_stats_chart(demand: np.ndarray, assigned: np.ndarray) -> go.Figure:
    """Create coverage statistics visualization."""
    
    # Calculate stats by day
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    daily_demand = []
    daily_assigned = []
    daily_coverage = []
    
    for day in range(7):
        day_demand = np.sum(demand[day*48:(day+1)*48])
        day_assigned = np.sum(assigned[day*48:(day+1)*48])
        coverage = (day_assigned / day_demand * 100) if day_demand > 0 else 0
        
        daily_demand.append(day_demand)
        daily_assigned.append(day_assigned)
        daily_coverage.append(coverage)
    
    # Create subplot figure
    fig = go.Figure()
    
    # Add demand bars
    fig.add_trace(go.Bar(
        x=days,
        y=daily_demand,
        name='Demand',
        marker_color='rgba(255, 99, 132, 0.7)',
        yaxis='y'
    ))
    
    # Add assigned bars
    fig.add_trace(go.Bar(
        x=days,
        y=daily_assigned,
        name='Assigned',
        marker_color='rgba(54, 162, 235, 0.7)',
        yaxis='y'
    ))
    
    # Add coverage line
    fig.add_trace(go.Scatter(
        x=days,
        y=daily_coverage,
        mode='lines+markers',
        name='Coverage %',
        line=dict(color='green', width=3),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    # Update layout with secondary y-axis
    fig.update_layout(
        title="Daily Coverage Statistics",
        xaxis_title="Day of Week",
        yaxis=dict(
            title="Number of Riders",
            side="left"
        ),
        yaxis2=dict(
            title="Coverage Percentage",
            side="right",
            overlaying="y"
        ),
        showlegend=True,
        height=400
    )
    
    return fig

def create_summary_metrics(demand: np.ndarray, assigned: Optional[np.ndarray] = None) -> dict:
    """Calculate summary metrics for the optimization results."""
    total_demand = np.sum(demand)
    
    metrics = {
        "total_demand": int(total_demand),
        "avg_hourly_demand": round(np.mean(demand), 2),
        "peak_demand": int(np.max(demand)),
        "peak_demand_slot": int(np.argmax(demand))
    }
    
    if assigned is not None:
        total_assigned = np.sum(assigned)
        understaffed_slots = np.sum(assigned < demand)
        overstaffed_slots = np.sum(assigned > demand)
        perfect_coverage_slots = np.sum(assigned == demand)
        
        overall_coverage = (total_assigned / total_demand * 100) if total_demand > 0 else 0
        
        metrics.update({
            "total_assigned": int(total_assigned),
            "overall_coverage": round(overall_coverage, 1),
            "understaffed_slots": int(understaffed_slots),
            "overstaffed_slots": int(overstaffed_slots),
            "perfect_coverage_slots": int(perfect_coverage_slots),
            "total_slots": len(demand)
        })
    
    return metrics