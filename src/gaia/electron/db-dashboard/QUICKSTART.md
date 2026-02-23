# GAIA DB Dashboard - Quick Start

**Modern React + TypeScript dashboard for real-time GAIA agent monitoring**

---

## Installation

```bash
cd src/gaia/electron/db-dashboard
npm install
```

**Takes:** 3-4 minutes
**Installs:** React 18, TypeScript, Vite, Electron, Tailwind, Recharts, React Query, Framer Motion

---

## Running

### Development Mode (Recommended)

```bash
npm run dev:electron
```

Opens Electron window with:
- Hot Module Replacement (HMR) - instant updates as you edit
- React DevTools support
- Fast refresh

### Production Mode

```bash
npm run build
npm start
```

---

## Features

✅ **Zero flashing** - React Query + placeholderData keeps UI stable
✅ **GitHub-style UI** - Modern dark theme, smooth animations
✅ **Live updates** - Auto-refresh with configurable interval (1s-10s)
✅ **Fast** - Virtualized tables handle 100K+ rows
✅ **Smart caching** - React Query prevents unnecessary refetches
✅ **Smooth charts** - Recharts with transitions

---

## Using with GAIA Code

### Terminal 1: Dashboard
```bash
cd src/gaia/electron/db-dashboard
npm run dev:electron
```

### Terminal 2: GAIA Code
```bash
gaia-code "Your task" --tui simple
```

**Watch:**
- Dashboard shows live stats
- logs.db tab updates smoothly (no flashing)
- Charts update in real-time

---

## Database Location

Default: `~/.gaia/workspace/`

**Databases:**
- memory.db, knowledge.db, tools.db, skills.db, agents.db, plan.db
- logs.db (created on first run)

---

Ready to monitor! 🚀
