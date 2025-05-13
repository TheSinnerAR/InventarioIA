import os
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
engine = create_engine(DB_URL)

with engine.connect() as conn:
    # Crear tablas
    conn.execute(text('DROP TABLE IF EXISTS ventas, inventario, promociones, festivos'))
    conn.execute(text(
        '''
        CREATE TABLE ventas(
            fecha DATE,
            sku TEXT,
            unidades_vendidas INTEGER
        )
        '''
    ))
    conn.execute(text(
        '''
        CREATE TABLE inventario(
            fecha DATE,
            sku TEXT,
            stock_actual INTEGER
        )
        '''
    ))
    conn.execute(text(
        '''
        CREATE TABLE promociones(
            fecha DATE,
            descuento REAL,
            tipo_promocion TEXT
        )
        '''
    ))
    conn.execute(text(
        '''
        CREATE TABLE festivos(
            fecha DATE,
            tipo_festivo TEXT
        )
        '''
    ))

    # Insertar datos de ejemplo
    start = datetime(2025,3,25)
    skus = ['SKU1', 'SKU2']
    for i in range(7):
        d = (start + timedelta(days=i)).date()
        for sku in skus:
            sold = 20 + hash((d, sku)) % 10
            stock = 100 - hash((sku, d)) % 20
            conn.execute(text(
                'INSERT INTO ventas VALUES (:fecha, :sku, :sold)'
            ), {'fecha': d, 'sku': sku, 'sold': sold})
            conn.execute(text(
                'INSERT INTO inventario VALUES (:fecha, :sku, :stock)'
            ), {'fecha': d, 'sku': sku, 'stock': stock})
    # Promociones y festivos
    promos = [
        {'fecha': datetime(2025,3,27).date(), 'descuento':0.15, 'tipo':'temporada'},
        {'fecha': datetime(2025,3,29).date(), 'descuento':0.20, 'tipo':'liquidacion'},
    ]
    for p in promos:
        conn.execute(text(
            'INSERT INTO promociones VALUES (:fecha, :descuento, :tipo)'
        ), {'fecha':p['fecha'], 'descuento':p['descuento'], 'tipo':p['tipo']})
    holidays = [
        {'fecha': datetime(2025,3,30).date(), 'tipo':'feriado_local'},
        {'fecha': datetime(2025,4,1).date(), 'tipo':'fin_de_semana'}
    ]
    for h in holidays:
        conn.execute(text(
            'INSERT INTO festivos VALUES (:fecha, :tipo)'
        ), {'fecha':h['fecha'], 'tipo':h['tipo']})

    print("Base de datos sembrada con éxito.")