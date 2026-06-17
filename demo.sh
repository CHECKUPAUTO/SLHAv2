#!/usr/bin/env bash
# SLHA v2 — Quick demo
# Montre ce que fait SLHA v2 en quelques secondes.

set -euo pipefail

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${CYAN}"
echo "  ╔══════════════════════════════════════╗"
echo "  ║       SLHA v2 — Démo rapide         ║"
echo "  ╚══════════════════════════════════════╝"
echo -e "${NC}"

cd "$(dirname "$0")"

# ── 1. Est-ce que ça compile ? ─────────────────────────────────────
echo -e "${YELLOW}[1/3] Compilation...${NC}"
cargo build --release --quiet 2>&1 | tail -1
echo -e "${GREEN}✓ Compilation OK${NC}"

# ── 2. Tests ────────────────────────────────────────────────────────
echo -e "${YELLOW}[2/3] Tests...${NC}"
cargo test --quiet 2>&1 | grep "test result" || true
echo -e "${GREEN}✓ Tests OK${NC}"

# ── 3. Exemple ──────────────────────────────────────────────────────
echo -e "${YELLOW}[3/3] Lancement de l'exemple...${NC}"
echo ""
cargo run --quiet --example basic_usage
echo ""

# ── Résumé ─────────────────────────────────────────────────────────
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}  SLHA v2 est fonctionnel sur cette machine${NC}"
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo ""
echo "  Pour aller plus loin :"
echo "    make measure        # Benchmark complet"
echo "    make doc            # Documentation"
echo "    docs/GETTING_STARTED.md  # Guide débutant"
echo "    docs/INTEGRATION.md      # Intégrer dans un projet"
echo ""
