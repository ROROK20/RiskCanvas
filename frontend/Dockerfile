# RiskCanvas frontend — Next.js 15. Serves on :3004.
FROM node:20-slim AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install

FROM node:20-slim AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# The engine URL is baked into the bundle at BUILD time (Next inlines
# NEXT_PUBLIC_*; setting it when the container runs has no effect). Pass it as
# a build arg: docker build --build-arg NEXT_PUBLIC_API_URL=https://your-api .
# Unset (the default) builds a demo-mode dashboard that calls no engine.
ARG NEXT_PUBLIC_API_URL=
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
# The public origin for absolute og:image URLs; unset ships no preview image.
ARG NEXT_PUBLIC_SITE_URL=
ENV NEXT_PUBLIC_SITE_URL=$NEXT_PUBLIC_SITE_URL
RUN npm run build

FROM node:20-slim AS runner
WORKDIR /app
ENV NODE_ENV=production
# The official node images ship an unprivileged `node` user (uid 1000); own the
# runtime files with it so the server (and Next's .next cache) run non-root.
COPY --from=builder --chown=node:node /app/package.json ./package.json
COPY --from=builder --chown=node:node /app/node_modules ./node_modules
COPY --from=builder --chown=node:node /app/.next ./.next
COPY --from=builder --chown=node:node /app/public ./public
COPY --from=builder --chown=node:node /app/next.config.mjs ./next.config.mjs
USER node

EXPOSE 3004
CMD ["npm", "run", "start"]
