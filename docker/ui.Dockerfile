FROM node:18-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:stable-alpine

COPY --from=builder /app/build /usr/share/nginx/html

RUN echo 'server {' \
'  listen 8000;' \
'  root /usr/share/nginx/html;' \
'  index index.html;' \
'  location / {' \
'    try_files $uri /index.html;' \
'  }' \
'  location ~* \.(js|css|png|jpg|jpeg|gif|svg|ico|woff2?)$ {' \
'    expires 1y;' \
'    access_log off;' \
'    add_header Cache-Control "public";' \
'  }' \
'}' > /etc/nginx/conf.d/default.conf

EXPOSE 8000
CMD ["nginx", "-g", "daemon off;"]
