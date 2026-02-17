# Docker Setup Guide

PostgreSQL + pgvector database setup for Arabic Financial RAG System.

## Prerequisites

- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher

### Install Docker (Ubuntu/Debian)

```bash
# Update package index
sudo apt-get update

# Install dependencies
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Verify installation
docker --version
docker compose version
```

## Quick Start

### 1. Start PostgreSQL Container

```bash
cd /home/ahmedsoliman/AI_projects/venv_arabic_rag/database
docker compose up -d
```

This will:
- Pull the `ankane/pgvector:v0.5.1` image
- Create a PostgreSQL container with pgvector extension
- Automatically run `schema.sql` to create tables and indexes
- Start the container in detached mode

### 2. Verify Container Status

```bash
docker compose ps
```

Expected output:
```
NAME                    COMMAND                  SERVICE    STATUS      PORTS
arabic_rag_postgres     "docker-entrypoint.s…"   postgres   Up          0.0.0.0:5432->5432/tcp
```

### 3. Check Health

```bash
docker compose exec postgres pg_isready -U arab_rag -d arab_rag_db
```

Expected output:
```
/var/run/postgresql:5432 - accepting connections
```

## Database Connection Details

- **Host**: `localhost`
- **Port**: `5432`
- **Database**: `arab_rag_db`
- **User**: `arab_rag`
- **Password**: `arab_rag_pass_2024`

### Connection String

```
postgresql://arab_rag:arab_rag_pass_2024@localhost:5432/arab_rag_db
```

## Common Commands

### Start Container

```bash
docker compose up -d
```

### Stop Container

```bash
docker compose down
```

### View Logs

```bash
docker compose logs -f postgres
```

### Access PostgreSQL Shell

```bash
docker compose exec postgres psql -U arab_rag -d arab_rag_db
```

### Restart Container

```bash
docker compose restart
```

### Remove Container and Data

⚠️ **Warning**: This will delete all data!

```bash
docker compose down -v
```

## Manual Schema Setup

If the schema wasn't automatically applied:

```bash
docker compose exec postgres psql -U arab_rag -d arab_rag_db -f /docker-entrypoint-initdb.d/schema.sql
```

## Verify pgvector Extension

```bash
docker compose exec postgres psql -U arab_rag -d arab_rag_db -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

Expected output:
```
 oid  | extname | extowner | extnamespace | extrelocatable | extversion | extconfig | extcondition
------+---------+----------+--------------+----------------+------------+-----------+--------------
 xxxxx| vector  |       10 |         2200 | f              | 0.5.1      |           |
```

## Troubleshooting

### Port Already in Use

If port 5432 is already occupied:

1. Stop the conflicting service:
   ```bash
   sudo systemctl stop postgresql
   ```

2. Or change the port in `docker-compose.yml`:
   ```yaml
   ports:
     - "5433:5432"  # Use external port 5433
   ```

### Permission Denied

If you get permission errors:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

### Container Won't Start

Check logs for errors:
```bash
docker compose logs postgres
```

### Reset Everything

```bash
docker compose down -v
docker compose up -d
```

## Performance Tuning

For production use, adjust PostgreSQL settings in `docker-compose.yml`:

```yaml
environment:
  POSTGRES_SHARED_BUFFERS: 512MB      # 25% of RAM
  POSTGRES_WORK_MEM: 32MB             # For sorting/joins
  POSTGRES_MAINTENANCE_WORK_MEM: 256MB # For index creation
  POSTGRES_EFFECTIVE_CACHE_SIZE: 2GB   # 50-75% of RAM
```

## Data Persistence

Data is stored in a Docker volume: `pgvector_data`

View volume location:
```bash
docker volume inspect database_pgvector_data
```

Backup data:
```bash
docker compose exec postgres pg_dump -U arab_rag arab_rag_db > backup.sql
```

Restore data:
```bash
cat backup.sql | docker compose exec -T postgres psql -U arab_rag -d arab_rag_db
```
