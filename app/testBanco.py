# test_gabbi_connection.py
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

# Senha original
password = "lrc2An*gvNP%00SkW%bY5cFLQV6S0o5v7^"

# URL encode seguro da senha
password_encoded = quote_plus(password)

DATABASE_URL = (
    f"postgresql+psycopg://gabbi_io:{password_encoded}"
    f"@192.168.230.108:5432/gabbi_io"
)

print("Testando conexão com banco Gabbi...\n")

try:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 5}
    )

    with engine.connect() as conn:
        result = conn.execute(text("SELECT version();"))
        version = result.scalar()

        print("✅ Conexão realizada com sucesso!\n")
        print("Versão PostgreSQL:")
        print(version)

        print("\nTestando tabela Article...")

        result = conn.execute(text("""
            SELECT COUNT(*) 
            FROM "Article"
        """))

        total = result.scalar()

        print(f"✅ Total de registros em Article: {total}")

except Exception as e:
    print("❌ Erro ao conectar:")
    print(str(e))