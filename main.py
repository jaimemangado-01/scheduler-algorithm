import pandas as pd
import numpy as np
from ortools.sat.python import cp_model
import os
import time

from optimizer_core import *

# --- 1. CONFIGURACIÓN DEL PLANIFICADOR ---
PREVIOUS_WEEK_SCHEDULE_FILE = 'horario_semana_anterior.csv'
FINAL_SCHEDULE_FILE = 'horario_VEC.csv'
SEED_SCHEDULE_FILE = 'horario_semilla_heuristica.csv'

# Tiempos para el flujo "En Frío"
SEED_TIME_LIMIT_SECONDS = 20
MAIN_TIME_LIMIT_SECONDS = 600
UNDERSUPPLY_PENALTY_MULTIPLIER = 1.0
OVERSUPPLY_PENALTY_MULTIPLIER = 2

def generate_heuristic_seed(riders, demand, city_code, date_map):
    """FASE 1: Genera una solución inicial usando la imposición del 70%"""
    print("\n" + "="*40); print("--- FASE 1: GENERANDO SEMILLA HEURÍSTICA ---"); print("="*40)
    
    model = cp_model.CpModel()
    x = { (r, d, s): model.NewBoolVar(f'x_{r}_{d}_{s}') for r in riders.keys() for d in range(NUM_DAYS) for s in range(SLOTS_PER_DAY) }
    
    # Aplicar el conjunto completo de restricciones, usando la versión FORZADA para los descansos
    aux_vars = {"y": None}
    model, aux_vars = apply_weekly_hours(model, x, riders, aux_vars)
    model, aux_vars = apply_daily_max_hours(model, x, riders, aux_vars)
    model, aux_vars = apply_rest_and_work_day_rules_FORCED(model, x, riders, demand, aux_vars)
    model, aux_vars = apply_night_shift_preferences(model, x, riders, aux_vars)
    model, aux_vars = apply_daily_shift_structure_rules(model, x, riders, aux_vars)
    model, aux_vars = apply_final_time_rules(model, x, riders, aux_vars, demand)

    # El objetivo es simplemente encontrar una solución factible que se ajuste lo mejor posible
    assigned = [sum(x[r,d,s] for r in riders.keys()) for d in range(NUM_DAYS) for s in range(SLOTS_PER_DAY)]
    faltan = [model.NewIntVar(0, len(riders), f'f_{i}') for i in range(TOTAL_SLOTS)]; sobran = [model.NewIntVar(0, len(riders), f's_{i}') for i in range(TOTAL_SLOTS)]
    for i in range(TOTAL_SLOTS): model.Add(assigned[i] + faltan[i] - sobran[i] == int(demand[i]))
    model.Minimize(sum(faltan[i] * 1.5 + sobran[i] for i in range(TOTAL_SLOTS)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = SEED_TIME_LIMIT_SECONDS
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print("✅ Semilla heurística encontrada.")
        #process_and_save_schedule(solver, x, riders, SEED_SCHEDULE_FILE.split('.')[0], city_code, date_map)
        solution_dict = {(r,d,s): solver.Value(x.get((r,d,s))) for r in riders for d in range(NUM_DAYS) for s in range(SLOTS_PER_DAY)}
        return solution_dict
    else:
        print("❌ FALLO CRÍTICO: No se pudo generar una semilla inicial. El problema, incluso con la heurística, es infactible.")
        return None

def run_main_optimization(riders, demand, city_code, date_map, warm_start_solution, time_limit):
    """FASE 2: Usa la semilla para una optimización profunda, con penalización jerárquica y reglas flexibles"""
    print("\n" + "="*40); print("--- FASE 2: OPTIMIZACIÓN PROFUNDA (WARM START) ---"); print("="*40)
    model = cp_model.CpModel()
    x = { (r, d, s): model.NewBoolVar(f'x_{r}_{d}_{s}') for r in riders.keys() for d in range(NUM_DAYS) for s in range(SLOTS_PER_DAY) }

    aux_vars = {"y": None}
    model, aux_vars = apply_weekly_hours(model, x, riders, aux_vars)
    model, aux_vars = apply_daily_max_hours(model, x, riders, aux_vars)
    model, aux_vars = apply_rest_and_work_day_rules(model, x, riders, aux_vars)
    model, aux_vars = apply_night_shift_preferences(model, x, riders, aux_vars)
    model, aux_vars = apply_daily_shift_structure_rules(model, x, riders, aux_vars)
    model, aux_vars = apply_final_time_rules(model, x, riders, aux_vars, demand)

    print("   -> Proporcionando semilla al solver para acelerar la búsqueda...")
    for (r, d, s), val in warm_start_solution.items():
        if val == 1 and (r, d, s) in x:
            model.AddHint(x[r, d, s], 1)
    
    day_multipliers_faltan = [1.0, 1.1, 1.2, 1.5, 2.5, 2.8, 3.0]
    penalty_weights_faltan = np.ones(TOTAL_SLOTS)
    for d in range(NUM_DAYS):
        for s in range(SLOTS_PER_DAY):
            if s >= 20*2 and s < 23*2: base_weight = 100 # Cena
            elif s >= 13*2 and s < 15*2: base_weight = 80 # Comida
            elif (s >= 15*2 and s < 20*2) or (s >= 23*2 and s < 24*2): base_weight = 40 # Tarde
            elif s >= 7*2 and s < 13*2: base_weight = 30 # Mañana
            else: base_weight = 10 # Madrugada
            penalty_weights_faltan[d * SLOTS_PER_DAY + s] = base_weight * day_multipliers_faltan[d]

    day_multipliers_sobran = [5.0, 2.8, 2.5, 2.0, 1.8, 1.5, 1.2]
    penalty_weights_sobran = np.ones(TOTAL_SLOTS)
    for d in range(NUM_DAYS):
        for s in range(SLOTS_PER_DAY):
            if s >= 20*2 and s < 23*2: base_weight = 20 # Cena
            elif s >= 13*2 and s < 15*2: base_weight = 20 # Comida
            elif (s >= 15*2 and s < 20*2) or (s >= 23*2 and s < 24*2): base_weight = 50 # Tarde
            elif s >= 7*2 and s < 13*2: base_weight = 80 # Mañana
            else: base_weight = 100 # Madrugada
            penalty_weights_sobran[d * SLOTS_PER_DAY + s] = base_weight * day_multipliers_sobran[d]
    
    assigned_per_slot = [sum(x[r, d, s] for r in riders.keys()) for d in range(NUM_DAYS) for s in range(SLOTS_PER_DAY)]
    faltan = [model.NewIntVar(0, len(riders), f'faltan_{i}') for i in range(TOTAL_SLOTS)]
    sobran = [model.NewIntVar(0, len(riders), f'sobran_{i}') for i in range(TOTAL_SLOTS)]
    for i in range(TOTAL_SLOTS):
        model.Add(assigned_per_slot[i] + faltan[i] - sobran[i] == int(demand[i]))
    
    penalizacion_total_faltan = sum(faltan[i] * UNDERSUPPLY_PENALTY_MULTIPLIER * penalty_weights_faltan[i] for i in range(TOTAL_SLOTS))
    penalizacion_total_sobran = sum(sobran[i] * OVERSUPPLY_PENALTY_MULTIPLIER * penalty_weights_sobran[i] for i in range(TOTAL_SLOTS))
    model.Minimize(penalizacion_total_faltan + penalizacion_total_sobran)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solution_monitor = SolutionMonitorCallback()
    status = solver.Solve(model, solution_monitor)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print(f"\n✅ ÉXITO: Se encontró una solución final y refinada.")
        process_and_save_schedule(solver, x, riders, FINAL_SCHEDULE_FILE.split('.')[0], city_code, date_map)
        assigned_result = np.array([sum(solver.Value(x[r,d,s]) for r in riders.keys()) for d in range(NUM_DAYS) for s in range(SLOTS_PER_DAY)])
        plot_coverage(demand, assigned_result, "Resultado Final del Optimizador")
    else:
        print(f"\n❌❌❌ FALLO CRÍTICO: El modelo final es INFACTIBLE.")

if __name__ == '__main__':
    # Por ahora, solo implementamos el flujo "En Frío"
    if os.path.exists(PREVIOUS_WEEK_SCHEDULE_FILE):
        print(f"Detectado horario anterior: '{PREVIOUS_WEEK_SCHEDULE_FILE}'.")
        print("El flujo de actualización 'En Caliente' aún no está implementado en este orquestador.")
    else:
        # --- Flujo "En Frío" ---
        print("No se detectó horario anterior. Iniciando flujo de creación 'En Frío'.")
        riders, demand, city_code, date_map = load_data()
        if riders:
            seed_solution = generate_heuristic_seed(riders, demand, city_code, date_map)
            if seed_solution:
                run_main_optimization(riders, demand, city_code, date_map, seed_solution, MAIN_TIME_LIMIT_SECONDS)