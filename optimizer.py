import pandas as pd
import numpy as np
from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
import collections
import random

# --- 1. CONFIGURACIÓN GLOBAL ---
RIDERS_FILE = 'riders_VEC.csv'
DEMAND_FILE = 'demand_VEC.csv'
MIN_SHIFT_HOURS = 2
MIN_REST_HOURS_BETWEEN_SHIFTS = 0

NUM_DAYS = 7
SLOTS_PER_DAY = 48
TOTAL_SLOTS = NUM_DAYS * SLOTS_PER_DAY
DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


# --- 2. FUNCIONES DE UTILIDAD ---
from datetime import datetime

def slot_to_time(slot_index):
    """Convierte un índice de slot (0-48) a formato HH:MM."""
    if slot_index == SLOTS_PER_DAY: return "00:00"
    hour = slot_index // 2
    minute = (slot_index % 2) * 30
    return f"{hour:02d}:{minute:02d}"

# En tu archivo optimizer_core.py

def load_data():
    """Carga y preprocesa los datos desde los archivos CSV"""
    try:
        riders_df = pd.read_csv(RIDERS_FILE)
        demand_df = pd.read_csv(DEMAND_FILE)
    except FileNotFoundError as e:
        print(f"❌ Error: No se encontró el archivo {e.filename}.")
        return None, None, None, None

    # --- FASE 1: VALIDACIÓN DE FORMATO Y LIMPIEZA DE RIDERS ---
    print("\n🕵️ Realizando validación de formato de riders...")
    riders_df.dropna(subset=['rider_id'], inplace=True)
    riders_df['rider_id'] = riders_df['rider_id'].astype(str)
    duplicates = riders_df[riders_df.duplicated(subset=['rider_id'], keep=False)]
    if not duplicates.empty:
        print("❌ ERROR: IDs de rider duplicados en 'riders.csv'.")
        print(duplicates.sort_values(by='rider_id'))
        return None, None, None, None
    riders_df['weekly_hours'] = pd.to_numeric(riders_df['weekly_hours'], errors='coerce')
    riders_df['max_daily_hours'] = pd.to_numeric(riders_df['max_daily_hours'], errors='coerce')
    riders_df.dropna(subset=['weekly_hours', 'max_daily_hours'], inplace=True)
    riders_df['preferred_rest_days'] = riders_df['preferred_rest_days'].fillna('')
    if 'acepta_madrugada' in riders_df.columns:
        riders_df['acepta_madrugada'] = riders_df['acepta_madrugada'].fillna(False).astype(bool)
    else:
        print("   -> ADVERTENCIA: No se encontró 'acepta_madrugada' en riders.csv. Se asumirá que nadie acepta turnos de madrugada.")
        riders_df['acepta_madrugada'] = False
    print("✅ Formato de riders correcto.")

    # --- FASE 2: VALIDACIÓN Y LIMPIEZA DE DEMANDA ---
    print("\n🕵️ Realizando validación de formato de demanda...")
    demand_col_name = None
    if 'rider_demand' in demand_df.columns: demand_col_name = 'rider_demand'
    elif 'riders_needed' in demand_df.columns: demand_col_name = 'riders_needed'
    if not demand_col_name:
        print("❌ ERROR en 'demand.csv': No se encontró la columna de demanda. Debe llamarse 'rider_demand' o 'riders_needed'.")
        return None, None, None, None  
    demand_df[demand_col_name] = pd.to_numeric(demand_df[demand_col_name], errors='coerce')
    rows_before = len(demand_df)
    demand_df.dropna(subset=[demand_col_name], inplace=True)
    rows_after = len(demand_df)
    if rows_after < rows_before:
        print(f"   -> ADVERTENCIA: Se eliminaron {rows_before - rows_after} filas de 'demand.csv' por tener valores de demanda no numéricos o vacíos.")
    print("✅ Formato de demanda correcto.")
    
    # --- FASE 3: VALIDACIÓN DE LÓGICA DE NEGOCIO ---
    print("\n🕵️ Realizando validación de lógica de negocio...")
    datos_validos = True
    for index, row in riders_df.iterrows():
        rider_id, weekly_h, daily_max_h = row['rider_id'], row['weekly_hours'], row['max_daily_hours']
        dias_a_trabajar = 5
        if weekly_h > 0:
            max_dias_posibles = int(weekly_h // MIN_SHIFT_HOURS)
            if max_dias_posibles < 5:
                dias_a_trabajar = max_dias_posibles
        else:
            dias_a_trabajar = 0
        if dias_a_trabajar == 0 and weekly_h > 0:
            print(f"❌ ERROR DE LÓGICA en Rider ID '{rider_id}': Tiene {weekly_h}h, pero con turnos de {MIN_SHIFT_HOURS}h, no puede trabajar ningún día.")
            datos_validos = False; continue
        capacidad_maxima = dias_a_trabajar * daily_max_h
        if weekly_h > capacidad_maxima:
            print(f"❌ ERROR DE DATOS en Rider ID '{rider_id}': (Capacidad Máxima Insuficiente)"); datos_validos = False
        capacidad_minima = dias_a_trabajar * MIN_SHIFT_HOURS
        if weekly_h < capacidad_minima:
            print(f"❌ ERROR DE DATOS en Rider ID '{rider_id}': (Capacidad Mínima Excesiva)"); datos_validos = False     
    if not datos_validos:
        print("\n🛑 El script no puede continuar. Revisa las inconsistencias lógicas en 'riders.csv'.")
        return None, None, None, None
    else:
        print("✅ Lógica de negocio consistente.")

    # --- FASE 4: PROCESAMIENTO FINAL ---
    demand_array = np.zeros(TOTAL_SLOTS, dtype=int)
    city_code = demand_df['codigo_ciudad'].iloc[0]
    date_map = {}  
    try:
        demand_df['parsed_date'] = pd.to_datetime(demand_df['day_of_week'], format='%d/%m/%Y')
    except Exception as e:
        print(f"❌ ERROR: El formato de fecha en 'demand.csv' no es 'DD/MM/YYYY'. Error: {e}")
        return None, None, None, None
    for _, row in demand_df.iterrows():
        day_index = row['parsed_date'].weekday()
        date_str = row['day_of_week']
        if day_index not in date_map:
            date_map[day_index] = date_str   
        slot_index = int(row.get('time') or row.get('slot_30min')) - 1
        demand_value = row[demand_col_name]
        global_index = day_index * SLOTS_PER_DAY + slot_index
        demand_array[global_index] = int(demand_value)
    riders_info = riders_df.set_index('rider_id').to_dict('index')
    print(f"✅ Datos cargados y validados para la ciudad '{city_code}'.")
    
    return riders_info, demand_array, city_code, date_map

def plot_coverage(demand, assigned, title):
    """Genera y muestra una gráfica de cobertura."""
    plt.style.use('seaborn-v0_8-whitegrid'); fig, ax = plt.subplots(figsize=(18, 6)); time_axis = np.arange(TOTAL_SLOTS)
    ax.fill_between(time_axis, demand, color="lightcoral", alpha=0.5, label='Demanda Requerida')
    ax.plot(time_axis, assigned, color="darkgreen", linewidth=2, label='Riders Asignados'); ax.set_title(title, fontsize=16, weight='bold')
    ax.set_xlabel("Slot de 30 minutos durante la semana"); ax.set_ylabel("Número de Riders"); ax.legend(); ax.set_xlim(0, TOTAL_SLOTS - 1)
    for day in range(1, NUM_DAYS): ax.axvline(x=day * SLOTS_PER_DAY, color='gray', linestyle='--', linewidth=1)
    print("📊 Mostrando gráfica... Cierra la ventana para terminar."); plt.show()

def process_and_save_schedule(solver, x, riders, filename_prefix, city_code, date_map):
    """Procesa el resultado y lo guarda en el nuevo formato de salida."""
    print(f"📝 Procesando y guardando el horario en '{filename_prefix}.csv'...")
    schedule_data = []
    for r_id in riders.keys():
        for d in range(NUM_DAYS):
            slots_trabajados = [s for s in range(SLOTS_PER_DAY) if solver.Value(x[r_id, d, s]) == 1]
            if not slots_trabajados: continue
            fecha_dia = date_map.get(d, f"Día_{d+1}")
            slots_trabajados.sort()
            start_slot = slots_trabajados[0]
            for i in range(1, len(slots_trabajados)):
                if slots_trabajados[i] != slots_trabajados[i-1] + 1:
                    end_slot = slots_trabajados[i-1] + 1
                    schedule_data.append({
                        'codigo_ciudad': city_code,
                        'courier_ID': r_id,
                        'dia': fecha_dia,
                        'hora_inicio': slot_to_time(start_slot),
                        'hora_final': slot_to_time(end_slot),
                        'accion': 'BOOK'
                    })
                    start_slot = slots_trabajados[i]      
            end_slot = slots_trabajados[-1] + 1
            schedule_data.append({
                'codigo_ciudad': city_code,
                'courier_ID': r_id,
                'dia': fecha_dia,
                'hora_inicio': slot_to_time(start_slot),
                'hora_final': slot_to_time(end_slot),
                'accion': 'BOOK'
            })
    if not schedule_data: print("   -> No se asignaron turnos."); return
    output_df = pd.DataFrame(schedule_data)
    output_df.to_csv(f"{filename_prefix}.csv", index=False); print(f"   -> Horario guardado.")

# --- 3. CLASE CALLBACK ---
class SolutionMonitorCallback(cp_model.CpSolverSolutionCallback):
    """Clase "espía" que se activa cada vez que el solver encuentra una nueva solución."""
    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0
    def on_solution_callback(self):
        self.__solution_count += 1
        print(f"  -> Solución #{self.__solution_count} encontrada! "
              f"Tiempo: {self.WallTime():.2f}s, "
              f"Nueva Penalización: {self.ObjectiveValue():.0f}")
    def solution_count(self):
        return self.__solution_count

# --- 4. FUNCIONES DE RESTRICCIONES ---
def apply_weekly_hours(model, x, riders, aux_vars):
    for r_id, r_info in riders.items():
        model.Add(sum(x[r_id,d,s] for d in range(NUM_DAYS) for s in range(SLOTS_PER_DAY)) == int(r_info['weekly_hours'] * 2))
    return model, aux_vars

def apply_daily_max_hours(model, x, riders, aux_vars):
    for r_id, r_info in riders.items():
        max_slots_diarios = int(r_info['max_daily_hours'] * 2)
        for d in range(NUM_DAYS):
            model.Add(sum(x[r_id, d, s] for s in range(SLOTS_PER_DAY)) <= max_slots_diarios)
    return model, aux_vars

def apply_rest_and_work_day_rules(model, x, riders, aux_vars):
    """Versión FLEXIBLE: El solver elige los mejores días de descanso consecutivos."""
    y = { (r_id, d): model.NewBoolVar(f'y_{r_id}_{d}') for r_id in riders.keys() for d in range(NUM_DAYS) }
    day_abbr_to_index = {name[:3]: i for i, name in enumerate(DAY_NAMES)}
    for r_id, r_info in riders.items():
        dias_a_trabajar = 5
        max_dias_posibles = int(r_info['weekly_hours'] // MIN_SHIFT_HOURS)
        if max_dias_posibles < 5:
            dias_a_trabajar = max_dias_posibles
            print(f"   -> AJUSTE para {r_id}: Se asignarán {dias_a_trabajar} días de trabajo.")
        model.Add(sum(y[r_id, d] for d in range(NUM_DAYS)) == dias_a_trabajar)
        preferred_days_str = r_info.get('preferred_rest_days', '')
        if isinstance(preferred_days_str, str) and '-' in preferred_days_str:
            print(f"   -> Aplicando descanso preferido para {r_id}: {preferred_days_str}")
            day1_abbr, day2_abbr = preferred_days_str.split('-')
            d1_idx, d2_idx = day_abbr_to_index.get(day1_abbr), day_abbr_to_index.get(day2_abbr)
            if d1_idx is not None and d2_idx is not None:
                model.Add(y[r_id, d1_idx] == 0); 
                model.Add(y[r_id, d2_idx] == 0)
        else:
            dias_de_descanso = NUM_DAYS - dias_a_trabajar
            if dias_de_descanso >= 2:
                pares_consecutivos = [(d, d + 1) for d in range(NUM_DAYS - 1)]
                condiciones = []
                for d1, d2 in pares_consecutivos:
                    es_par_de_descanso = model.NewBoolVar(f'descanso_{r_id}_{d1}_{d2}')
                    condiciones.append(es_par_de_descanso)
                    model.AddBoolAnd([y[r_id, d1].Not(), y[r_id, d2].Not()]).OnlyEnforceIf(es_par_de_descanso)
                    model.AddImplication(es_par_de_descanso, y[r_id, d1].Not())
                    model.AddImplication(es_par_de_descanso, y[r_id, d2].Not())
                model.Add(sum(condiciones) >= 1)
        for d in range(NUM_DAYS):
            model.Add(sum(x[r_id, d, s] for s in range(SLOTS_PER_DAY)) > 0).OnlyEnforceIf(y[r_id, d])
            model.Add(sum(x[r_id, d, s] for s in range(SLOTS_PER_DAY)) == 0).OnlyEnforceIf(y[r_id, d].Not())
    aux_vars["y"] = y
    return model, aux_vars

def apply_rest_and_work_day_rules_FORCED(model, x, riders, demand, aux_vars):
    """Usa un pre-análisis de demanda para asignar de forma inteligente los días de descanso"""
    y = { (r_id, d): model.NewBoolVar(f'y_{r_id}_{d}') for r_id in riders.keys() for d in range(NUM_DAYS) }
    day_abbr_to_index = {name[:3]: i for i, name in enumerate(DAY_NAMES)}
    num_total_riders = len(riders)
    total_rest_days_to_assign = num_total_riders * 2 

    # Calcular la "presión" de demanda de cada día
    demand_slots_per_day = [sum(demand[d * SLOTS_PER_DAY : (d + 1) * SLOTS_PER_DAY]) for d in range(NUM_DAYS)]
    total_weekly_demand_slots = sum(demand_slots_per_day) if sum(demand_slots_per_day) > 0 else 1
    day_pressure = [demand_slots / total_weekly_demand_slots for demand_slots in demand_slots_per_day]

    # El día con MENOS presión de demanda es el MEJOR para descansar.
    rest_attractiveness = [(1 - pressure) for pressure in day_pressure]
    total_attractiveness = sum(rest_attractiveness) if sum(rest_attractiveness) > 0 else 1

    # Distribuir el total de días de descanso según la "atracción" de cada día
    target_resting_riders_per_day = [(att / total_attractiveness) * total_rest_days_to_assign for att in rest_attractiveness]

    for d in range(NUM_DAYS):
        print(f"      - {DAY_NAMES[d]}: Demanda Relativa={day_pressure[d]:.1%}. Objetivo descansos: {target_resting_riders_per_day[d]:.1f} riders.")
    
    pares_consecutivos = [(d, d + 1) for d in range(NUM_DAYS - 1)]
    rest_assignments = {pair: [] for pair in pares_consecutivos}
    rider_to_rest_pair = {}

    # Pre-asignar riders con preferencias
    riders_sin_preferencias = list(riders.keys())
    for r_id in list(riders_sin_preferencias):
        r_info = riders[r_id]
        pref_str = r_info.get('preferred_rest_days', '')
        if isinstance(pref_str, str) and '-' in pref_str:
            d1_abbr, d2_abbr = pref_str.split('-')
            d1, d2 = day_abbr_to_index.get(d1_abbr), day_abbr_to_index.get(d2_abbr)
            if d1 is not None and d2 is not None and (d1, d2) in rest_assignments:
                rest_assignments[(d1, d2)].append(r_id)
                rider_to_rest_pair[r_id] = (d1, d2)
                riders_sin_preferencias.remove(r_id)

    for r_id in riders_sin_preferencias:
        bucket_need = {}
        for d1, d2 in pares_consecutivos:
            target = (target_resting_riders_per_day[d1] + target_resting_riders_per_day[d2]) / 2
            need = target - len(rest_assignments.get((d1, d2), []))
            bucket_need[(d1, d2)] = max(0, need)
        
        if not bucket_need: continue
        best_pair = max(bucket_need, key=bucket_need.get)
        rest_assignments[best_pair].append(r_id)
        rider_to_rest_pair[r_id] = best_pair

    for r_id, r_info in riders.items():
        dias_a_trabajar = 5
        if r_info['weekly_hours'] > 0:
            max_dias_posibles = int(r_info['weekly_hours'] // MIN_SHIFT_HOURS)
            if max_dias_posibles < 5:
                dias_a_trabajar = max_dias_posibles
        else:
            dias_a_trabajar = 0
        model.Add(sum(y[r_id, d] for d in range(NUM_DAYS)) == dias_a_trabajar)
        if r_id in rider_to_rest_pair:
            d1, d2 = rider_to_rest_pair[r_id]
            model.Add(y[r_id, d1] == 0)
            model.Add(y[r_id, d2] == 0)
        else:
            if dias_a_trabajar == 0:
                for d in range(NUM_DAYS): model.Add(y[r_id, d] == 0)
            else:
                model.Add(sum(y[r_id, d].Not() for d in range(NUM_DAYS)) == (NUM_DAYS - dias_a_trabajar))
        for d in range(NUM_DAYS):
            model.Add(sum(x[r_id, d, s] for s in range(SLOTS_PER_DAY)) > 0).OnlyEnforceIf(y[r_id, d])
            model.Add(sum(x[r_id, d, s] for s in range(SLOTS_PER_DAY)) == 0).OnlyEnforceIf(y[r_id, d].Not())
            
    aux_vars["y"] = y
    return model, aux_vars

def apply_night_shift_preferences(model, x, riders, aux_vars):
    """Prohíbe trabajar de madrugada a los riders que no lo han aceptado."""
    # Definir qué son los slots de madrugada (ej: de 00:00 a 07:00)
    night_slots = [s for s in range(0, 7 * 2)]
    for r_id, r_info in riders.items():
        if not r_info.get('acepta_madrugada', False):
            for d in range(NUM_DAYS):
                for s in night_slots: model.Add(x[r_id, d, s] == 0)
    return model, aux_vars

def apply_daily_shift_structure_rules(model, x, riders, aux_vars):
    for r_id in riders.keys():
        for d in range(NUM_DAYS):
            for s in range(SLOTS_PER_DAY):
                # Prohíbe R-T-R (turno de 0.5h)
                if s > 0 and s < SLOTS_PER_DAY - 1: model.AddBoolOr([x[r_id, d, s - 1], x[r_id, d, s].Not(), x[r_id, d, s + 1]])
                # Prohíbe R-T-T-R (turno de 1h)
                if s > 0 and s < SLOTS_PER_DAY - 2: model.AddBoolOr([x[r_id, d, s - 1], x[r_id, d, s].Not(), x[r_id, d, s + 1].Not(), x[r_id, d, s + 2]])
                # Prohíbe R-T-T-T-R (turno de 1.5h)
                if s > 0 and s < SLOTS_PER_DAY - 3: model.AddBoolOr([x[r_id, d, s - 1], x[r_id, d, s].Not(), x[r_id, d, s + 1].Not(), x[r_id, d, s + 2].Not(), x[r_id, d, s + 3]])
            
            starts = [model.NewBoolVar(f'start_{r_id}_{d}_{s}') for s in range(SLOTS_PER_DAY)]
            model.Add(starts[0] == x[r_id, d, 0])
            for s in range(1, SLOTS_PER_DAY):
                model.Add(starts[s] == 1).OnlyEnforceIf(x[r_id, d, s]).OnlyEnforceIf(x[r_id, d, s-1].Not())
                model.Add(starts[s] == 0).OnlyEnforceIf(x[r_id, d, s].Not())
                model.Add(starts[s] == 0).OnlyEnforceIf(x[r_id, d, s-1])
            model.Add(sum(starts) <= 2)
            for s in range(1, SLOTS_PER_DAY - 1):
                model.AddBoolOr([x[r_id, d, s - 1].Not(), x[r_id, d, s], x[r_id, d, s + 1].Not()])
    return model, aux_vars

def apply_final_time_rules(model, x, riders, aux_vars, demand):
    for i in range(TOTAL_SLOTS):
        if demand[i] == 0:
            d = i // SLOTS_PER_DAY; 
            s = i % SLOTS_PER_DAY
            for r_id in riders.keys(): 
                model.Add(x[r_id, d, s] == 0)
    slots_de_descanso = int(MIN_REST_HOURS_BETWEEN_SHIFTS * 2)
    if slots_de_descanso <= 0: # Si el descanso es 0, no añadir ninguna restricción
        return model, aux_vars
    for r_id in riders.keys():
        for d in range(NUM_DAYS):
            for s in range(SLOTS_PER_DAY):
                for i in range(1, slots_de_descanso):
                    next_s_raw = s + i
                    day_offset = next_s_raw // SLOTS_PER_DAY
                    if d + day_offset >= NUM_DAYS:
                        continue
                    next_s = next_s_raw % SLOTS_PER_DAY
                    next_d = d + day_offset
                    model.AddBoolOr([x[r_id, d, s].Not(), x[r_id, next_d, next_s].Not()])
    return model, aux_vars