FROM node:18-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:stable-alpine

COPY --from=builder /app/build /usr/share/nginx/html

RUN addgroup -g 1000 -S appgroup && \
    adduser -u 1000 -S appuser -G appgroup && \
    mkdir -p /var/cache/nginx/client_temp && \
    mkdir -p /tmp/nginx && \
    chown -R 1000:1000 /var/cache/nginx && \
    chown -R 1000:1000 /usr/share/nginx/html && \
    chown -R 1000:1000 /tmp/nginx

RUN echo \
'pid /tmp/nginx.pid;' \
'events {' \
'  worker_connections 1024;' \
'}' \
'http {' \
'  server {' \
'    listen 8000;' \
'    root /usr/share/nginx/html;' \
'    index index.html;' \
'    location / {' \
'      try_files $uri /index.html;' \
'    }' \
'    location ~* \.(js|css|png|jpg|jpeg|gif|svg|ico|woff2?)$ {' \
'      expires 1y;' \
'      access_log off;' \
'      add_header Cache-Control "public";' \
'    }' \
'  }' \
'}' > /etc/nginx/nginx.conf

USER 1000

EXPOSE 8000
CMD ["nginx", "-g", "daemon off;"]