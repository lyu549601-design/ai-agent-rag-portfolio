FROM python:3.12-alpine

RUN printf "http://mirrors.tuna.tsinghua.edu.cn/alpine/v3.24/main\\nhttp://mirrors.tuna.tsinghua.edu.cn/alpine/v3.24/community\\n" > /etc/apk/repositories \
    && apk add --no-cache postgresql18 postgresql-pgvector su-exec \
    && mkdir -p /var/lib/postgresql/data /docker-entrypoint-initdb.d \
    && chown -R postgres:postgres /var/lib/postgresql /docker-entrypoint-initdb.d

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENV PGDATA=/var/lib/postgresql/data
EXPOSE 5432
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["postgres", "-D", "/var/lib/postgresql/data"]