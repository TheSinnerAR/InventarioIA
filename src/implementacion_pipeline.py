import os
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score
import pulp

# Configuración de la conexión a la base de datos
# Ajusta la URL según tu motor (PostgreSQL, MySQL, SQLite, etc.)
DB_URL = os.getenv('DB_URL', 'sqlite:///inventario_full.db')
engine = create_engine(DB_URL)


def load_data(start_date=None, end_date=None):
    """
    Carga datos de ventas e inventario desde la base de datos.
    Parámetros opcionales: rango de fechas para filtrar.
    """
    query = """
    SELECT
        s.fecha,
        s.sku,
        s.unidades_vendidas,
        p.stock_actual,
        pr.descuento,
        pr.tipo_promocion,
        f.tipo_festivo
    FROM ventas s
    JOIN inventario p ON s.sku = p.sku AND s.fecha = p.fecha
    LEFT JOIN promociones pr ON s.fecha = pr.fecha
    LEFT JOIN festivos f ON s.fecha = f.fecha
    WHERE 1=1
    """
    if start_date:
        query += f" AND s.fecha >= '{start_date}'"
    if end_date:
        query += f" AND s.fecha <= '{end_date}'"
    df = pd.read_sql(query, engine)
    return df


def preprocess_data(df):
    """
    Limpieza, imputación de faltantes, generación de variables de calendario.
    """
    df = df.drop_duplicates()
    # Imputar valores faltantes
    df['descuento'] = df['descuento'].fillna(0)
    df['tipo_promocion'] = df['tipo_promocion'].fillna('ninguna')
    df['tipo_festivo'] = df['tipo_festivo'].fillna('normal')

    # Variables de calendario
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['dia_semana'] = df['fecha'].dt.weekday
    df['mes'] = df['fecha'].dt.month
    return df


def train_demand_model(df):
    """
    Entrena un RandomForest para predecir unidades vendidas.
    Devuelve el modelo ajustado.
    """
    features = ['dia_semana', 'mes', 'descuento', 'stock_actual']
    X = df[features]
    y = df['unidades_vendidas']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(random_state=42))
    ])

    param_grid = {
        'rf__n_estimators': [50, 100],
        'rf__max_depth': [10, 20, None]
    }
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_mean_absolute_error')
    grid.fit(X_train, y_train)

    # Evaluación
    preds = grid.predict(X_test)
    print('MAE:', mean_absolute_error(y_test, preds))
    print('RMSE:', mean_squared_error(y_test, preds))

    return grid.best_estimator_


def train_anomaly_detector(df):
    """
    Entrena un IsolationForest para detectar anomalías en movimientos de inventario.
    """
    features = ['unidades_vendidas', 'stock_actual']
    X = df[features]
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X)
    return iso


def optimize_inventory(forecasted_demand, cost_holding, cost_stockout):
    """
    Optimiza cantidades de pedido basadas en demanda pronosticada.
    Usa modelo EOQ extendido con programación lineal.
    forecasted_demand: dict {sku: demanda}
    cost_holding, cost_stockout: dict {sku: costo}
    """
    problem = pulp.LpProblem('Optimizar_Stock', pulp.LpMinimize)
    # Variables de cantidad a pedir
    Q = {sku: pulp.LpVariable(f'Q_{sku}', lowBound=0, cat='Continuous')
         for sku in forecasted_demand}
    
    # Variables auxiliares para modelar el máximo
    shortage = {sku: pulp.LpVariable(f'Shortage_{sku}', lowBound=0, cat='Continuous')
                for sku in forecasted_demand}
    
    # Restricciones para modelar max(forecasted_demand - Q, 0)
    for sku in forecasted_demand:
        problem += shortage[sku] >= forecasted_demand[sku] - Q[sku]
        problem += shortage[sku] >= 0

    # Función objetivo: costos de almacenamiento + faltantes
    holding = pulp.lpSum([cost_holding[sku] * Q[sku] for sku in Q])
    stockout = pulp.lpSum([cost_stockout[sku] * shortage[sku] for sku in Q])
    problem += holding + stockout

    # Restricciones de negocio (opcional)
    # ejs: limite de presupuesto, espacio de almacén, etc.

    problem.solve(pulp.PULP_CBC_CMD(msg=0))
    return {sku: Q[sku].value() for sku in Q}


def main():
    # Cargar y preparar datos
    df = load_data()
    df_clean = preprocess_data(df)

    # Entrenar modelos
    model_demand = train_demand_model(df_clean)
    model_anomaly = train_anomaly_detector(df_clean)

    # Pronóstico de demanda para el próximo día
    features_next = df_clean.copy()  # Generar features para fecha futura...
    forecast = model_demand.predict(features_next[['dia_semana', 'mes', 'descuento', 'stock_actual']])

    # Detección de anomalías
    anomalies = model_anomaly.predict(df_clean[['unidades_vendidas', 'stock_actual']])
    df_clean['anomaly'] = anomalies
    alerts = df_clean[df_clean['anomaly'] == -1]
    
    # Exportar alertas a Excel
    if not alerts.empty:
        alerts_file = 'alertas_anomalias.xlsx'
        alerts.to_excel(alerts_file, index=False)
        print(f'Alertas de anomalías exportadas a: {alerts_file}')
    
    print(f'Se encontraron {len(alerts)} anomalías en los datos')

    # Optimización
    demand_dict = dict(zip(df_clean['sku'], forecast))
    # Suponiendo costos constantes por SKU:
    cost_holding = {sku: 1.0 for sku in demand_dict}
    cost_stockout = {sku: 5.0 for sku in demand_dict}
    
    # Obtener cantidades óptimas
    order_quantities = optimize_inventory(demand_dict, cost_holding, cost_stockout)
    
    # Redondear las cantidades a números enteros
    order_quantities = {sku: round(qty) for sku, qty in order_quantities.items()}
    
    # Convertir el diccionario order_quantities a un DataFrame
    df_orders = pd.DataFrame({
        'sku': list(order_quantities.keys()),
        'cantidad_a_pedir': list(order_quantities.values())
    })
    
    # Calcular el total de unidades a pedir
    total_unidades = df_orders['cantidad_a_pedir'].sum()
    
    # Exportar a Excel el resultado de order_quantities con total
    orders_file = 'cantidades_pedido.xlsx'
    
    # Crear un objeto ExcelWriter para tener más control sobre el Excel
    with pd.ExcelWriter(orders_file, engine='openpyxl') as writer:
        df_orders.to_excel(writer, index=False, sheet_name='Pedidos')
        
        # Obtener la hoja de trabajo para añadir el total
        workbook = writer.book
        worksheet = writer.sheets['Pedidos']
        
        # Añadir el total en la fila siguiente a los datos
        row_total = len(df_orders) + 2  # +2 porque hay una fila de encabezado y queremos dejar una fila en blanco
        worksheet.cell(row=row_total, column=1, value="TOTAL")
        worksheet.cell(row=row_total, column=2, value=total_unidades)
    
    print(f'Cantidades de pedido exportadas a: {orders_file}')
    
    # Mostrar un resumen por consola
    print("\n=== Resumen de Optimización ===")
    print(f"Total SKUs optimizados: {len(order_quantities)}")
    print(f"Cantidad total a pedir: {total_unidades} unidades")
    
    # Mostrar los 5 SKUs con mayor cantidad a pedir
    top_orders = sorted(order_quantities.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 SKUs con mayor cantidad a pedir:")
    for sku, qty in top_orders:
        print(f"SKU: {sku}, Cantidad: {qty}")


if __name__ == '__main__':
    main()