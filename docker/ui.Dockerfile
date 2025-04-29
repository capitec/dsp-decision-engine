FROM node:18-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY src/ ./src/
COPY public/ ./public/
COPY tsconfig*.json ./*.js ./*.json ./

RUN npm run build
# This is useful for testing
# RUN npm i serve
# EXPOSE 3000
# CMD ["npx", "serve", "-s", "build"]

FROM nginx:stable-alpine

COPY --from=builder /app/build /usr/share/nginx/html

RUN addgroup -g 1000 -S appgroup && \
    adduser -u 1000 -S appuser -G appgroup && \
    mkdir -p /var/cache/nginx/client_temp && \
    mkdir -p /tmp/nginx && \
    chown -R 1000:1000 /var/cache/nginx && \
    chown -R 1000:1000 /usr/share/nginx/html && \
    chown -R 1000:1000 /tmp/nginx

COPY nginx.conf /etc/nginx/nginx.conf

USER 1000

EXPOSE 8000
CMD ["nginx", "-g", "daemon off;"]