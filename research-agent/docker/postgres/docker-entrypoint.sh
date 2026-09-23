#!/bin/sh
set -eu

PGDATA="${PGDATA:-/var/lib/postgresql/data}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-password}"
POSTGRES_DB="${POSTGRES_DB:-$POSTGRES_USER}"

mkdir -p "$PGDATA" /run/postgresql
chown -R postgres:postgres "$PGDATA" /run/postgresql

if [ ! -s "$PGDATA/PG_VERSION" ]; then
    su-exec postgres initdb \
        --username="$POSTGRES_USER" \
        --auth-local=trust \
        --auth-host=scram-sha-256 \
        --encoding=UTF8

    printf "\nlisten_addresses = '*'\n" >> "$PGDATA/postgresql.conf"
    printf "host all all all scram-sha-256\n" >> "$PGDATA/pg_hba.conf"

    su-exec postgres pg_ctl -D "$PGDATA" -o "-c listen_addresses='localhost'" -w start
    su-exec postgres psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
        -c "ALTER USER \"$POSTGRES_USER\" WITH PASSWORD '$POSTGRES_PASSWORD';"

    if [ "$POSTGRES_DB" != "postgres" ]; then
        su-exec postgres createdb --username "$POSTGRES_USER" "$POSTGRES_DB"
    fi

    for file in /docker-entrypoint-initdb.d/*; do
        [ -e "$file" ] || continue
        su-exec postgres psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -f "$file"
    done

    su-exec postgres pg_ctl -D "$PGDATA" -w stop
fi

exec su-exec postgres "$@"